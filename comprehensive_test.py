#!/usr/bin/env python3
"""
Test script for the FastAPI Qwen Coder API
This script shows various ways to test the API and handle common errors
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8002"

def wait_for_api():
    """Wait for the API to be ready"""
    print("Waiting for API to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ready":
                    print("✅ API is ready!")
                    return True
                else:
                    print(f"API status: {data.get('status', 'unknown')}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_attempts}: API not ready yet...")
            time.sleep(2)
    
    print("❌ API did not become ready in time")
    return False

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_generate_endpoint():
    """Test the text generation endpoint with correct format"""
    print("\n🔍 Testing generate endpoint...")
    
    # Correct request format
    payload = {
        "prompt": "Write a simple Python function to add two numbers",
        "max_tokens": 150
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request: {json.dumps(payload, indent=2)}")
        response = requests.post(
            f"{API_BASE_URL}/generate",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Generated text: {result['generated_text']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_invalid_requests():
    """Test various invalid requests that might cause 422 errors"""
    print("\n🔍 Testing invalid request formats (expecting 422 errors)...")
    
    test_cases = [
        {
            "name": "Missing prompt field",
            "payload": {"max_tokens": 100}
        },
        {
            "name": "Wrong field name",
            "payload": {"text": "Hello", "max_tokens": 100}
        },
        {
            "name": "Invalid max_tokens type",
            "payload": {"prompt": "Hello", "max_tokens": "invalid"}
        },
        {
            "name": "Empty payload",
            "payload": {}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                headers={"Content-Type": "application/json"},
                json=test_case["payload"]
            )
            print(f"  Status Code: {response.status_code}")
            if response.status_code == 422:
                print("  ✅ Expected 422 error received")
                error_detail = response.json()
                print(f"  Error detail: {error_detail}")
            else:
                print(f"  Unexpected status: {response.status_code}")
                
        except Exception as e:
            print(f"  Request failed: {e}")

def main():
    """Main test function"""
    print("🚀 FastAPI Qwen Coder API Test Suite")
    print("="*50)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ Cannot proceed - API is not ready")
        return
    
    # Run tests
    test_health_endpoint()
    test_generate_endpoint()
    test_invalid_requests()
    
    print("\n" + "="*50)
    print("✅ Test suite completed!")

if __name__ == "__main__":
    main()