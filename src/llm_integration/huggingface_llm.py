"""
HuggingFace LLM Integration Module

Provides integration with HuggingFace transformers for language model functionality.
Supports model loading, text generation, quantization, and configuration management.
"""
from typing import Dict, List, Any, Optional, Union
import logging
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMBase(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class HuggingFaceLLM(LLMBase):
    """Hugging Face LLM implementation with support for various open-source models."""
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        'llama2-7b': {
            'model_id': 'meta-llama/Llama-2-7b-chat-hf',
            'max_length': 4096,
            'temperature': 0.7
        },
        'llama2-13b': {
            'model_id': 'meta-llama/Llama-2-13b-chat-hf',
            'max_length': 4096,
            'temperature': 0.7
        },
        'mistral-7b': {
            'model_id': 'mistralai/Mistral-7B-Instruct-v0.1',
            'max_length': 8192,
            'temperature': 0.7
        },
        'falcon-7b': {
            'model_id': 'tiiuae/falcon-7b-instruct',
            'max_length': 2048,
            'temperature': 0.7
        },
        'codellama-7b': {
            'model_id': 'codellama/CodeLlama-7b-Instruct-hf',
            'max_length': 4096,
            'temperature': 0.2
        }
    }
    
    def __init__(
        self, 
        model_name: str = 'mistral-7b',
        device: str = 'auto',
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_auth_token: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_auth_token = use_auth_token
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self._load_model()
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for memory efficiency."""
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_config['model_id']}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['model_id'],
                use_auth_token=self.use_auth_token,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Load model
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_config['model_id'],
                'use_auth_token': self.use_auth_token,
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            if self.device != 'auto' and not (self.load_in_4bit or self.load_in_8bit):
                model_kwargs['device_map'] = self.device
            else:
                model_kwargs['device_map'] = 'auto'
            
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        try:
            if not self.is_model_loaded():
                raise RuntimeError("Model not loaded")
            
            # Use model defaults if not specified
            max_length = max_length or self.model_config['max_length']
            temperature = temperature or self.model_config['temperature']
            
            # Format prompt for the specific model
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate
            with torch.no_grad():
                sequences = self.pipeline(
                    formatted_prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Extract generated text
            generated_text = sequences[0]['generated_text']
            
            # Remove the original prompt from the output
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt according to model requirements."""
        if 'llama' in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        elif 'mistral' in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        elif 'falcon' in self.model_name.lower():
            return f"User: {prompt}\nAssistant:"
        else:
            return prompt
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "model_id": self.model_config['model_id'],
            "device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
            "quantization": {
                "load_in_4bit": self.load_in_4bit,
                "load_in_8bit": self.load_in_8bit
            },
            "parameters": {
                "max_length": self.model_config['max_length'],
                "temperature": self.model_config['temperature']
            },
            "status": "loaded"
        }
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            logger.info("Model unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise
