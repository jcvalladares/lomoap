#!/bin/bash

# Production startup script for Multi-Model Coder API

set -e  # Exit on any error

echo "🚀 Starting Multi-Model Coder API"
echo "=================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8+."
    exit 1
fi

if ! command_exists nvidia-smi; then
    echo "⚠️  nvidia-smi not found. Running on CPU (will be slower)."
    export CUDA_VISIBLE_DEVICES=""
else
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
fi

# Set environment variables for optimization
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Optional: Enable specific models only
# export ENABLE_QWEN=true
# export ENABLE_DEEPSEEK=true

# Optional: Use quantization for lower memory usage
# export USE_8BIT=true    # 8-bit quantization
# export USE_4BIT=true    # 4-bit quantization (even lower memory)

# Check available disk space
echo "💾 Checking disk space..."
df -h . | tail -1 | awk '{print "Available disk space: " $4}'

# Check available memory
echo "🧠 Checking system memory..."
free -h | grep "Mem:" | awk '{print "Available RAM: " $7 "/" $2}'

# Start the API with production settings
echo "🎬 Starting API server..."
echo "Access the API at: http://localhost:${API_PORT:-8002}"
echo "API documentation: http://localhost:${API_PORT:-8002}/docs"
echo ""
echo "🔄 Loading models (this may take a few minutes)..."

# Run with uvicorn for production
python -c "
import uvicorn
from main import app
from config import API_CONFIG

print(f'Starting server on {API_CONFIG[\"host\"]}:{API_CONFIG[\"port\"]}')
uvicorn.run(
    app,
    host=API_CONFIG['host'],
    port=API_CONFIG['port'],
    workers=API_CONFIG['workers'],
    timeout_keep_alive=API_CONFIG['timeout_keep_alive'],
    log_level='info'
)
"