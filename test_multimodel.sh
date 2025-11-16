#!/bin/bash

# Multi-Model FastAPI Coder API Test Examples

echo "=== Testing Multi-Model FastAPI Coder API ==="

# Test health endpoint
echo "1. Testing health endpoint:"
curl -X GET "http://localhost:8002/health" \
     -H "accept: application/json"

echo -e "\n\n"

# Test models endpoint
echo "2. Testing models endpoint:"
curl -X GET "http://localhost:8002/models" \
     -H "accept: application/json"

echo -e "\n\n"

# Test Qwen model endpoint
echo "3. Testing Qwen model endpoint:"
curl -X POST "http://localhost:8002/generate/qwen" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a Python function to calculate fibonacci numbers",
       "max_tokens": 150
     }'

echo -e "\n\n"

# Test DeepSeek model endpoint
echo "4. Testing DeepSeek model endpoint:"
curl -X POST "http://localhost:8002/generate/deepseek" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a Python function to sort a list",
       "max_tokens": 150
     }'

echo -e "\n\n"

# Test default endpoint (backward compatibility)
echo "5. Testing default endpoint (should use Qwen):"
curl -X POST "http://localhost:8002/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Hello, explain what you can do",
       "max_tokens": 100
     }'

echo -e "\n\nDone!"