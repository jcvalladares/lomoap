#!/usr/bin/env python3
"""
Test script for the Multi-Model Coder API with lazy loading
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8002"

def test_lazy_loading_demo():
    """Demonstrate lazy loading behavior"""
    print("🚀 Multi-Model Coder API - Lazy Loading Demo")
    print("=" * 60)
    
    # 1. Check initial status
    print("\n1. 📊 Initial API status:")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    # 2. List available models
    print("\n2. 📋 Available models:")
    response = requests.get(f"{API_BASE_URL}/models")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    # 3. First request to Qwen (should trigger loading)
    print("\n3. 🔄 First request to Qwen (will trigger model loading):")
    print("Making request...")
    start_time = time.time()
    
    response = requests.post(
        f"{API_BASE_URL}/generate/qwen",
        json={
            "prompt": "Write a simple Python hello world function",
            "max_tokens": 100
        },
        timeout=120  # Allow time for model loading
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! (took {end_time - start_time:.2f} seconds)")
        print(f"Generated text: {result['generated_text'][:200]}...")
        print(f"Model used: {result['model_used']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    
    # 4. Second request to Qwen (should be faster)
    print("\n4. ⚡ Second request to Qwen (should be faster, model already loaded):")
    start_time = time.time()
    
    response = requests.post(
        f"{API_BASE_URL}/generate/qwen",
        json={
            "prompt": "Write a function to calculate factorial",
            "max_tokens": 100
        }
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! (took {end_time - start_time:.2f} seconds)")
        print(f"Generated text: {result['generated_text'][:200]}...")
    
    # 5. Check memory status
    print("\n5. 💾 Memory status after Qwen loading:")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"GPU Memory allocated: {health.get('memory_info', {}).get('gpu_allocated_gb', 'N/A')} GB")
        print(f"Models in memory: {[k for k, v in health['models'].items() if v['in_memory']]}")
    
    # 6. Request to DeepSeek (should unload Qwen and load DeepSeek)
    print("\n6. 🔄 Request to DeepSeek (will unload Qwen and load DeepSeek):")
    start_time = time.time()
    
    response = requests.post(
        f"{API_BASE_URL}/generate/deepseek",
        json={
            "prompt": "Write a binary search algorithm in Python",
            "max_tokens": 150
        },
        timeout=120
    )
    
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! (took {end_time - start_time:.2f} seconds)")
        print(f"Generated text: {result['generated_text'][:200]}...")
        print(f"Model used: {result['model_used']}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    
    # 7. Final memory status
    print("\n7. 💾 Final memory status:")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"GPU Memory allocated: {health.get('memory_info', {}).get('gpu_allocated_gb', 'N/A')} GB")
        print(f"Models in memory: {[k for k, v in health['models'].items() if v['in_memory']]}")
        print("\nModel statuses:")
        for model, status in health['models'].items():
            print(f"  {model}: {status['status']} (in_memory: {status['in_memory']})")

def test_manual_model_management():
    """Test manual preloading and unloading"""
    print("\n\n🔧 Manual Model Management Demo")
    print("=" * 40)
    
    # Preload Qwen
    print("\n1. 📥 Preloading Qwen model:")
    response = requests.post(f"{API_BASE_URL}/models/qwen/preload")
    if response.status_code == 200:
        print("✅ Qwen preloaded")
        print(json.dumps(response.json(), indent=2))
    
    # Unload Qwen
    print("\n2. 📤 Unloading Qwen model:")
    response = requests.post(f"{API_BASE_URL}/models/qwen/unload")
    if response.status_code == 200:
        print("✅ Qwen unloaded")
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    try:
        print("Waiting for API to be ready...")
        time.sleep(2)  # Give API time to start
        
        # Check if API is running
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API not running. Please start the API first with: python main.py")
            exit(1)
            
        test_lazy_loading_demo()
        test_manual_model_management()
        
        print("\n🎉 Demo completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start the API first with: python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")