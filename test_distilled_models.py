#!/usr/bin/env python3
"""
Test script for Distilled Multi-Model Coder API
Demonstrates the performance benefits of using distilled models
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8002"

def test_model_performance(model_endpoint, model_name, prompt, max_tokens=100):
    """Test a specific model and measure performance"""
    print(f"\n🧪 Testing {model_name}")
    print(f"📍 Endpoint: {model_endpoint}")
    print(f"💭 Prompt: '{prompt[:50]}...'")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}{model_endpoint}",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens
            },
            timeout=120
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")
            
            print(f"✅ Success!")
            print(f"⏱️  Time: {duration:.2f} seconds")
            print(f"📝 Generated: {len(generated_text)} characters")
            print(f"🔤 Preview: {generated_text[:100]}...")
            
            return {
                "success": True,
                "duration": duration,
                "length": len(generated_text),
                "model": model_name
            }
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return {"success": False, "error": response.text, "model": model_name}
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Exception after {duration:.2f}s: {e}")
        return {"success": False, "error": str(e), "model": model_name}

def compare_all_models():
    """Compare performance across all distilled models"""
    print("🚀 Distilled Multi-Model Performance Comparison")
    print("=" * 60)
    
    # Test prompts for different scenarios
    test_cases = [
        {
            "name": "Simple Function",
            "prompt": "Write a Python function to calculate the factorial of a number",
            "max_tokens": 150
        },
        {
            "name": "Algorithm Implementation",
            "prompt": "Implement a binary search algorithm in Python with comments",
            "max_tokens": 200
        },
        {
            "name": "Code Explanation",
            "prompt": "Explain how recursion works with a simple example",
            "max_tokens": 120
        }
    ]
    
    # Models to test (in order of expected size)
    models_to_test = [
        ("/generate/qwen-tiny", "Qwen 0.5B (Ultra-tiny)"),
        ("/generate/qwen", "Qwen 1.5B (Distilled)"),
        ("/generate/codegemma", "CodeGemma 2B"),
        ("/generate/deepseek", "DeepSeek Lite Base")
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"🎯 Test Case: {test_case['name']}")
        print(f"{'='*60}")
        
        case_results = []
        
        for model_endpoint, model_name in models_to_test:
            result = test_model_performance(
                model_endpoint, 
                model_name, 
                test_case["prompt"],
                test_case["max_tokens"]
            )
            result["test_case"] = test_case["name"]
            case_results.append(result)
            
            # Small delay between tests
            time.sleep(2)
        
        results.extend(case_results)
        
        # Show case summary
        successful_results = [r for r in case_results if r["success"]]
        if successful_results:
            fastest = min(successful_results, key=lambda x: x["duration"])
            slowest = max(successful_results, key=lambda x: x["duration"])
            
            print(f"\n📊 {test_case['name']} Summary:")
            print(f"🏆 Fastest: {fastest['model']} ({fastest['duration']:.2f}s)")
            print(f"🐌 Slowest: {slowest['model']} ({slowest['duration']:.2f}s)")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("🏁 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        # Group by model
        model_performance = {}
        for result in successful_results:
            model = result["model"]
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(result["duration"])
        
        # Calculate averages
        model_averages = {}
        for model, times in model_performance.items():
            avg_time = sum(times) / len(times)
            model_averages[model] = avg_time
        
        # Sort by performance
        sorted_models = sorted(model_averages.items(), key=lambda x: x[1])
        
        print("📈 Average Response Times:")
        for i, (model, avg_time) in enumerate(sorted_models, 1):
            medal = ["🥇", "🥈", "🥉", "🏅"][min(i-1, 3)]
            print(f"{medal} {i}. {model}: {avg_time:.2f}s")
    
    return results

def check_memory_usage():
    """Check memory usage for each model"""
    print(f"\n{'='*60}")
    print("💾 MEMORY USAGE COMPARISON")
    print(f"{'='*60}")
    
    models = ["qwen-tiny", "qwen", "codegemma", "deepseek"]
    
    for model in models:
        print(f"\n🔍 Checking {model.upper()} memory usage...")
        
        # Check memory before loading
        response = requests.get(f"{API_BASE_URL}/gpu")
        if response.status_code == 200:
            before = response.json()
            
        # Load model
        requests.post(f"{API_BASE_URL}/models/{model}/preload")
        
        # Check memory after loading
        response = requests.get(f"{API_BASE_URL}/gpu")
        if response.status_code == 200:
            after = response.json()
            
            if after["gpu_hardware"].get("cuda_available"):
                for gpu in after["gpu_hardware"]["devices"]:
                    print(f"  GPU {gpu['id']}: {gpu['memory_allocated_gb']}GB allocated")
            
            # Check if model is loaded
            loaded_models = after.get("loaded_models", {})
            if model in loaded_models:
                device_info = loaded_models[model]["device_info"]
                print(f"  📍 Location: {device_info.get('primary_device', 'unknown')}")
        
        # Unload model for next test
        requests.post(f"{API_BASE_URL}/models/{model}/unload")
        time.sleep(1)

def main():
    """Main test function"""
    try:
        # Check if API is running
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API not running. Please start with: python main.py")
            return
        
        # Show API info
        api_info = response.json()
        print("🎯 Connected to:", api_info.get("message", "API"))
        print("📋 Available models:", len(api_info.get("available_endpoints", [])))
        
        # Run tests
        print("\n🧪 Starting comprehensive model testing...")
        
        # Performance comparison
        results = compare_all_models()
        
        # Memory usage comparison  
        check_memory_usage()
        
        print(f"\n🎉 Testing completed!")
        print(f"📊 Total tests run: {len(results)}")
        print(f"✅ Successful tests: {len([r for r in results if r['success']])}")
        
        print(f"\n💡 Distilled Model Benefits:")
        print(f"  🚀 Faster inference times")
        print(f"  💾 Lower memory usage") 
        print(f"  ⚡ Quicker model loading")
        print(f"  🎯 Good performance for code tasks")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please start with: python main.py")
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()