"""
Model Configuration for the reproductive health chatbot

This module contains configuration settings for the language models used in the chatbot.
"""

# Model Configuration
DEFAULT_MODEL = "gpt-4-mini"  # Use GPT-4 mini for cost optimization

# Model Settings
MODEL_SETTINGS = {
    "gpt-4-mini": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
        "model_type": "chat",
        "pricing": {
            "input": 0.150,      # $0.150 per 1M input tokens
            "output": 0.600,     # $0.600 per 1M output tokens
            "cached_input": 0.075  # $0.075 per 1M cached input tokens
        }
    }
}

# Fallback Model (used if GPT-4 mini is unavailable)
FALLBACK_MODEL = "gpt-3.5-turbo"

# Model Capabilities
MODEL_CAPABILITIES = {
    "gpt-4-mini": {
        "max_context_length": 8192,
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_audio": False
    }
}

# Model Performance Settings
PERFORMANCE_SETTINGS = {
    "max_retries": 3,
    "timeout": 30,  # seconds
    "batch_size": 1,
    "cache_enabled": True,
    "cache_ttl": 3600  # 1 hour in seconds
}

# Token Usage Limits
TOKEN_LIMITS = {
    "max_input_tokens": 4096,
    "max_output_tokens": 2048,
    "max_total_tokens": 6144
}

# Cost Optimization Settings
COST_OPTIMIZATION = {
    "enable_caching": True,
    "cache_threshold": 0.8,  # Cache responses with similarity > 80%
    "max_cache_size": 1000,  # Maximum number of cached responses
    "cache_cleanup_interval": 3600  # Clean up cache every hour
} 