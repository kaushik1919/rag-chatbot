"""
Access control and authentication for RAG chatbot.
"""
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    READ_DOCUMENTS = "read_documents"
    UPLOAD_DOCUMENTS = "upload_documents"
    DELETE_DOCUMENTS = "delete_documents"
    QUERY_SYSTEM = "query_system"
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    CONFIGURE_SYSTEM = "configure_system"


class AccessController:
    """Manage user access control and permissions."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.READ_DOCUMENTS,
                Permission.UPLOAD_DOCUMENTS,
                Permission.DELETE_DOCUMENTS,
                Permission.QUERY_SYSTEM,
                Permission.MANAGE_USERS,
                Permission.VIEW_LOGS,
                Permission.CONFIGURE_SYSTEM
            },
            UserRole.USER: {
                Permission.READ_DOCUMENTS,
                Permission.UPLOAD_DOCUMENTS,
                Permission.QUERY_SYSTEM
            },
            UserRole.READONLY: {
                Permission.READ_DOCUMENTS,
                Permission.QUERY_SYSTEM
            },
            UserRole.GUEST: {
                Permission.QUERY_SYSTEM
            }
        }
        
        # User storage
        self.users = {}
        self.sessions = {}
        self.failed_attempts = {}
        
        # Load existing users
        self._load_users()
        
        # Rate limiting
        self.rate_limits = {
            'queries_per_minute': 30,
            'uploads_per_hour': 10,
            'failed_login_attempts': 5,
            'lockout_duration': 300  # 5 minutes
        }
        
        # Session management
        self.session_timeout = 3600  # 1 hour
        
    def _load_users(self) -> None:
        """Load users from storage."""
        users_file = self.config_dir / 'users.json'
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    user_data = json.load(f)
                    self.users = user_data.get('users', {})
                logger.info(f"Loaded {len(self.users)} users")
            except Exception as e:
                logger.error(f"Error loading users: {e}")
                self.users = {}
        
        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _save_users(self) -> None:
        """Save users to storage."""
        users_file = self.config_dir / 'users.json'
        try:
            user_data = {
                'users': self.users,
                'updated': datetime.now().isoformat()
            }
            with open(users_file, 'w') as f:
                json.dump(user_data, f, indent=2)
            logger.info("Saved user data")
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        default_password = secrets.token_urlsafe(16)
        self.create_user(
            username='admin',
            password=default_password,
            role=UserRole.ADMIN,
            email='admin@localhost'
        )
        logger.warning(f"Created default admin user with password: {default_password}")
        logger.warning("Please change the default password immediately!")
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        password_hash, _ = self._hash_password(password, salt)
        return hmac.compare_digest(password_hash, stored_hash)
    
    def create_user(
        self, 
        username: str, 
        password: str, 
        role: UserRole,
        email: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a new user."""
        try:
            # Validate username
            if not username or len(username) < 3:
                raise ValueError("Username must be at least 3 characters")
            
            if username in self.users:
                raise ValueError("Username already exists")
            
            # Validate password
            if not self._validate_password_strength(password):
                raise ValueError("Password does not meet strength requirements")
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            # Create user record
            user_data = {
                'username': username,
                'password_hash': password_hash,
                'salt': salt,
                'role': role.value,
                'email': email,
                'created_at': datetime.now().isoformat(),
                'last_login': None,
                'is_active': True,
                'metadata': metadata or {}
            }
            
            self.users[username] = user_data
            self._save_users()
            
            logger.info(f"Created user: {username} with role: {role.value}")
            return {
                'success': True,
                'username': username,
                'role': role.value
            }
            
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and create session."""
        try:
            # Check if user is locked out
            if self._is_user_locked_out(username):
                return {
                    'success': False,
                    'error': 'Account temporarily locked due to failed login attempts'
                }
            
            # Check if user exists
            if username not in self.users:
                self._record_failed_attempt(username)
                return {
                    'success': False,
                    'error': 'Invalid username or password'
                }
            
            user = self.users[username]
            
            # Check if user is active
            if not user.get('is_active', True):
                return {
                    'success': False,
                    'error': 'Account is disabled'
                }
            
            # Verify password
            if not self._verify_password(password, user['password_hash'], user['salt']):
                self._record_failed_attempt(username)
                return {
                    'success': False,
                    'error': 'Invalid username or password'
                }
            
            # Clear failed attempts
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            session_data = {
                'username': username,
                'role': user['role'],
                'created_at': time.time(),
                'last_activity': time.time()
            }
            
            self.sessions[session_token] = session_data
            
            # Update last login
            user['last_login'] = datetime.now().isoformat()
            self._save_users()
            
            logger.info(f"User {username} authenticated successfully")
            return {
                'success': True,
                'session_token': session_token,
                'user_info': {
                    'username': username,
                    'role': user['role'],
                    'permissions': list(perm.value for perm in self.role_permissions[UserRole(user['role'])])
                }
            }
            
        except Exception as e:
            logger.error(f"Error authenticating user {username}: {e}")
            return {
                'success': False,
                'error': 'Authentication error'
            }
    
    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate session token and return user info."""
        try:
            if not session_token or session_token not in self.sessions:
                return {
                    'valid': False,
                    'error': 'Invalid session token'
                }
            
            session = self.sessions[session_token]
            current_time = time.time()
            
            # Check if session expired
            if current_time - session['last_activity'] > self.session_timeout:
                del self.sessions[session_token]
                return {
                    'valid': False,
                    'error': 'Session expired'
                }
            
            # Update last activity
            session['last_activity'] = current_time
            
            return {
                'valid': True,
                'username': session['username'],
                'role': session['role'],
                'permissions': list(perm.value for perm in self.role_permissions[UserRole(session['role'])])
            }
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return {
                'valid': False,
                'error': 'Session validation error'
            }
    
    def check_permission(self, session_token: str, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        session_info = self.validate_session(session_token)
        
        if not session_info.get('valid', False):
            return False
        
        user_role = UserRole(session_info['role'])
        user_permissions = self.role_permissions.get(user_role, set())
        
        return required_permission in user_permissions
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session."""
        if session_token in self.sessions:
            username = self.sessions[session_token]['username']
            del self.sessions[session_token]
            logger.info(f"User {username} logged out")
            return True
        return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets strength requirements."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        current_time = time.time()
        
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(current_time)
        
        # Remove old attempts (older than lockout duration)
        cutoff_time = current_time - self.rate_limits['lockout_duration']
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff_time
        ]
    
    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = len(self.failed_attempts[username])
        return recent_attempts >= self.rate_limits['failed_login_attempts']
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        return {
            'username': username,
            'role': user['role'],
            'email': user.get('email'),
            'created_at': user['created_at'],
            'last_login': user.get('last_login'),
            'is_active': user.get('is_active', True)
        }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (admin only)."""
        return [self.get_user_info(username) for username in self.users.keys()]
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account."""
        if username in self.users:
            self.users[username]['is_active'] = False
            self._save_users()
            
            # Invalidate all sessions for this user
            sessions_to_remove = [
                token for token, session in self.sessions.items()
                if session['username'] == username
            ]
            
            for token in sessions_to_remove:
                del self.sessions[token]
            
            logger.info(f"Deactivated user: {username}")
            return True
        
        return False
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        active_sessions = len(self.sessions)
        total_users = len(self.users)
        active_users = len([u for u in self.users.values() if u.get('is_active', True)])
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'active_sessions': active_sessions,
            'failed_attempts': len(self.failed_attempts),
            'rate_limits': self.rate_limits
        }
