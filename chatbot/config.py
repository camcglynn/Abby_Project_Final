import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration management for the chatbot"""
    
    def __init__(self):
        """Initialize configuration with environment variables and defaults"""
        # API Keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.abortion_policy_api_key = os.environ.get("ABORTION_POLICY_API_KEY")
        
        # API Endpoints
        self.abortion_policy_base_url = "https://api.abortionpolicyapi.com/v1"
        
        # Model Settings
        self.gpt_model = "gpt-4"  # Default to GPT-4
        self.gpt_temperature = 0.7
        self.gpt_max_tokens = 1200
        
        # Rate Limiting
        self.api_request_delay = 0.4  # seconds between API calls
        
        # Logging
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        
        # Initialize logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_config()
    
    def _setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _validate_config(self):
        """Validate required configuration settings"""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
            logger.warning("OpenAI API key not found in environment variables")
        
        if not self.abortion_policy_api_key:
            missing_keys.append("ABORTION_POLICY_API_KEY")
            logger.warning("Abortion Policy API key not found in environment variables")
        
        if missing_keys:
            logger.warning(f"Missing required API keys: {', '.join(missing_keys)}")
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service"""
        keys = {
            "openai": self.openai_api_key,
            "abortion_policy": self.abortion_policy_api_key
        }
        return keys.get(service, "")
    
    def get_model_settings(self) -> Dict[str, Any]:
        """Get GPT model settings"""
        return {
            "model": self.gpt_model,
            "temperature": self.gpt_temperature,
            "max_tokens": self.gpt_max_tokens
        }
    
    def update_setting(self, key: str, value: Any) -> None:
        """Update a configuration setting"""
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"Updated configuration setting: {key} = {value}")
        else:
            logger.warning(f"Attempted to update unknown configuration setting: {key}")

# Create a global configuration instance
config = Config() 