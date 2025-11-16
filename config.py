"""
Production configuration for the Multi-Model Coder API
"""

import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "enabled": True,
        "config": {
            "torch_dtype": "float16",  # Use float16 for better memory efficiency
            "device_map": "auto",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "load_in_8bit": False,  # Set to True if you want to use 8-bit quantization
            "load_in_4bit": False,  # Set to True if you want to use 4-bit quantization
        }
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "enabled": True,
        "config": {
            "torch_dtype": "float16",
            "device_map": "auto", 
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "load_in_8bit": False,
            "load_in_4bit": False,
        }
    }
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8002,
    "workers": 1,  # Keep at 1 for model sharing
    "timeout_keep_alive": 65,
    "max_request_size": 1024 * 1024,  # 1MB max request size
}

# Generation defaults
GENERATION_CONFIG = {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "do_sample": True,
}

# Memory optimization settings
MEMORY_CONFIG = {
    "enable_memory_efficient_attention": True,
    "gradient_checkpointing": False,  # Not needed for inference
    "torch_compile": False,  # Set to True for PyTorch 2.0+ optimization
}

# Environment variables
def get_env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes', 'on')

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

# Override configs with environment variables
API_CONFIG["port"] = get_env_int("API_PORT", API_CONFIG["port"])
API_CONFIG["host"] = os.getenv("API_HOST", API_CONFIG["host"])

# Enable/disable models via environment
MODEL_CONFIGS["qwen"]["enabled"] = get_env_bool("ENABLE_QWEN", True)
MODEL_CONFIGS["deepseek"]["enabled"] = get_env_bool("ENABLE_DEEPSEEK", True)

# Quantization options (for lower memory usage)
if get_env_bool("USE_8BIT"):
    for config in MODEL_CONFIGS.values():
        config["config"]["load_in_8bit"] = True

if get_env_bool("USE_4BIT"):
    for config in MODEL_CONFIGS.values():
        config["config"]["load_in_4bit"] = True