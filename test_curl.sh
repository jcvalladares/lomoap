#!/bin/bash

# FastAPI Qwen Coder API Test Examples

echo "=== Testing FastAPI Qwen Coder API ==="

# Test health endpoint
echo "1. Testing health endpoint:"
curl -X GET "http://localhost:8002/health" \
     -H "accept: application/json"

echo -e "\n\n"

# Test generate endpoint with correct format
echo "2. Testing generate endpoint with correct request:"
curl -X POST "http://localhost:8002/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Write a Python function to calculate fibonacci numbers",
       "max_tokens": 200
     }'

echo -e "\n\n"

# Test with minimal request (only required field)
echo "3. Testing with minimal request (only prompt):"
curl -X POST "http://localhost:8002/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Hello, how are you?"
     }'

echo -e "\n\n"

# Example of request that will cause 422 error
echo "4. Testing invalid request (missing prompt field - expect 422 error):"
curl -X POST "http://localhost:8002/generate" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "This will fail because field name is wrong",
       "max_tokens": 100
     }'

echo -e "\n\nDone!"