"""
Content filtering and safety mechanisms for RAG chatbot.
"""
from typing import List, Dict, Any, Optional, Set
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FilterLevel(Enum):
    """Content filtering levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"


class ContentCategory(Enum):
    """Categories of potentially harmful content."""
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    ADULT_CONTENT = "adult_content"
    PERSONAL_INFO = "personal_info"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    HARASSMENT = "harassment"


class ContentFilter:
    """Filter and classify content for safety."""
    
    def __init__(self, filter_level: FilterLevel = FilterLevel.MODERATE):
        self.filter_level = filter_level
        self.blocked_patterns = self._initialize_patterns()
        self.warning_patterns = self._initialize_warning_patterns()
        
    def _initialize_patterns(self) -> Dict[ContentCategory, List[str]]:
        """Initialize patterns for different content categories."""
        patterns = {
            ContentCategory.HATE_SPEECH: [
                r'\b(hate|racist|nazi|supremacist)\b',
                r'\b(discriminat\w+)\b',
                r'\b(slur|offensive|derogatory)\b'
            ],
            ContentCategory.VIOLENCE: [
                r'\b(kill|murder|assassinate|bomb|terrorist|weapon)\b',
                r'\b(violence|violent|attack|assault|harm)\b',
                r'\b(shoot|stab|poison|torture)\b'
            ],
            ContentCategory.ADULT_CONTENT: [
                r'\b(porn|sexual|explicit|nude|adult)\b',
                r'\b(xxx|nsfw|erotic)\b'
            ],
            ContentCategory.PERSONAL_INFO: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card pattern
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
                r'\b\d{3}-\d{3}-\d{4}\b'  # Phone number pattern
            ],
            ContentCategory.SPAM: [
                r'\b(click here|buy now|limited time|act now)\b',
                r'\b(free money|earn \$|make money fast)\b',
                r'\b(winner|congratulations|you\'ve won)\b'
            ],
            ContentCategory.HARASSMENT: [
                r'\b(harass|bully|threaten|intimidate)\b',
                r'\b(stalk|follow|watch|spy)\b'
            ]
        }
        
        # Compile patterns for efficiency
        compiled_patterns = {}
        for category, pattern_list in patterns.items():
            compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]
        
        return compiled_patterns
    
    def _initialize_warning_patterns(self) -> List[re.Pattern]:
        """Initialize patterns that should trigger warnings but not blocking."""
        warning_patterns = [
            r'\b(controversial|sensitive|political)\b',
            r'\b(opinion|belief|controversial)\b',
            r'\b(confidential|private|internal)\b'
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in warning_patterns]
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for safety and appropriateness."""
        try:
            result = {
                'is_safe': True,
                'risk_level': 'low',
                'detected_categories': [],
                'warnings': [],
                'confidence': 1.0,
                'filtered_content': content
            }
            
            if not content or not content.strip():
                return result
            
            detected_issues = []
            
            # Check against blocked patterns
            for category, patterns in self.blocked_patterns.items():
                matches = []
                for pattern in patterns:
                    found_matches = pattern.findall(content)
                    if found_matches:
                        matches.extend(found_matches)
                
                if matches:
                    detected_issues.append({
                        'category': category,
                        'matches': matches,
                        'severity': self._get_category_severity(category)
                    })
            
            # Check warning patterns
            warning_matches = []
            for pattern in self.warning_patterns:
                found_matches = pattern.findall(content)
                if found_matches:
                    warning_matches.extend(found_matches)
            
            if warning_matches:
                result['warnings'].append(f"Potentially sensitive content detected: {warning_matches}")
            
            # Process detected issues
            if detected_issues:
                result['detected_categories'] = [issue['category'].value for issue in detected_issues]
                
                # Determine overall risk level
                max_severity = max(issue['severity'] for issue in detected_issues)
                
                if max_severity >= 8:
                    result['risk_level'] = 'high'
                    result['is_safe'] = False
                elif max_severity >= 5:
                    result['risk_level'] = 'medium'
                    result['is_safe'] = self._should_allow_medium_risk()
                else:
                    result['risk_level'] = 'low'
                    result['warnings'].append("Low-risk content patterns detected")
                
                # Calculate confidence based on number and severity of matches
                total_matches = sum(len(issue['matches']) for issue in detected_issues)
                result['confidence'] = min(1.0, (total_matches * max_severity) / 20)
                
                # Filter content if necessary
                if not result['is_safe']:
                    result['filtered_content'] = self._filter_content(content, detected_issues)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                'is_safe': False,
                'risk_level': 'unknown',
                'detected_categories': [],
                'warnings': [f"Content analysis error: {str(e)}"],
                'confidence': 0.0,
                'filtered_content': ''
            }
    
    def _get_category_severity(self, category: ContentCategory) -> int:
        """Get severity score for content category (1-10)."""
        severity_map = {
            ContentCategory.HATE_SPEECH: 9,
            ContentCategory.VIOLENCE: 9,
            ContentCategory.HARASSMENT: 8,
            ContentCategory.ADULT_CONTENT: 7,
            ContentCategory.PERSONAL_INFO: 8,
            ContentCategory.SPAM: 3,
            ContentCategory.MISINFORMATION: 6
        }
        return severity_map.get(category, 5)
    
    def _should_allow_medium_risk(self) -> bool:
        """Determine if medium risk content should be allowed based on filter level."""
        if self.filter_level == FilterLevel.STRICT:
            return False
        elif self.filter_level == FilterLevel.MODERATE:
            return True  # Allow with warnings
        else:  # RELAXED
            return True
    
    def _filter_content(self, content: str, detected_issues: List[Dict]) -> str:
        """Filter or redact problematic content."""
        filtered_content = content
        
        for issue in detected_issues:
            category = issue['category']
            matches = issue['matches']
            
            for match in matches:
                if category == ContentCategory.PERSONAL_INFO:
                    # Replace with placeholder
                    if '@' in match:
                        filtered_content = filtered_content.replace(match, '[EMAIL_REDACTED]')
                    elif '-' in match and len(match.replace('-', '')) >= 10:
                        filtered_content = filtered_content.replace(match, '[PHONE_REDACTED]')
                    else:
                        filtered_content = filtered_content.replace(match, '[PII_REDACTED]')
                else:
                    # Replace with generic placeholder
                    filtered_content = filtered_content.replace(match, '[CONTENT_FILTERED]')
        
        return filtered_content
    
    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Quick safety check for user queries."""
        analysis = self.analyze_content(query)
        
        return {
            'is_safe': analysis['is_safe'],
            'risk_level': analysis['risk_level'],
            'should_process': analysis['is_safe'] and analysis['risk_level'] != 'high',
            'warnings': analysis['warnings']
        }
    
    def filter_response(self, response: str) -> Dict[str, Any]:
        """Filter generated response content."""
        analysis = self.analyze_content(response)
        
        return {
            'original_response': response,
            'filtered_response': analysis['filtered_content'],
            'was_filtered': response != analysis['filtered_content'],
            'safety_info': {
                'is_safe': analysis['is_safe'],
                'risk_level': analysis['risk_level'],
                'detected_categories': analysis['detected_categories'],
                'warnings': analysis['warnings']
            }
        }
    
    def add_custom_pattern(self, category: ContentCategory, pattern: str) -> None:
        """Add custom filtering pattern."""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            if category not in self.blocked_patterns:
                self.blocked_patterns[category] = []
            self.blocked_patterns[category].append(compiled_pattern)
            logger.info(f"Added custom pattern for {category.value}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern}, error: {e}")
            raise
    
    def update_filter_level(self, new_level: FilterLevel) -> None:
        """Update the content filtering level."""
        self.filter_level = new_level
        logger.info(f"Updated filter level to {new_level.value}")
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get statistics about the content filter."""
        total_patterns = sum(len(patterns) for patterns in self.blocked_patterns.values())
        
        return {
            'filter_level': self.filter_level.value,
            'total_patterns': total_patterns,
            'categories': list(self.blocked_patterns.keys()),
            'warning_patterns': len(self.warning_patterns)
        }
