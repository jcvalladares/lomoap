#!/usr/bin/env python3
"""
GPU Monitoring Script for Multi-Model Coder API
This script helps you monitor GPU usage and model locations
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8002"

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_gpu_status():
    """Check detailed GPU status"""
    try:
        response = requests.get(f"{API_BASE_URL}/gpu")
        if response.status_code == 200:
            data = response.json()
            
            print_separator("🖥️  GPU HARDWARE STATUS")
            
            gpu_hardware = data.get("gpu_hardware", {})
            if gpu_hardware.get("cuda_available"):
                print(f"CUDA Available: ✅ Yes")
                print(f"GPU Count: {gpu_hardware.get('gpu_count', 0)}")
                print(f"Current Device: {gpu_hardware.get('current_device', 'N/A')}")
                
                print(f"\n📊 GPU Memory Status:")
                for gpu in gpu_hardware.get("devices", []):
                    status_icon = "🟢" if float(gpu["utilization_percent"]) > 0 else "⚫"
                    print(f"{status_icon} GPU {gpu['id']}: {gpu['name']}")
                    print(f"   Memory: {gpu['memory_allocated_gb']}GB / {gpu['memory_total_gb']}GB ({gpu['utilization_percent']}%)")
                    print(f"   Free: {gpu['memory_free_gb']}GB")
            else:
                print("CUDA Available: ❌ No - Running on CPU")
                print(f"Reason: {gpu_hardware.get('message', 'Unknown')}")
            
            print_separator("🤖 LOADED MODELS")
            
            loaded_models = data.get("loaded_models", {})
            if loaded_models:
                for model_key, info in loaded_models.items():
                    device_info = info.get("device_info", {})
                    primary_device = device_info.get("primary_device", "unknown")
                    
                    # Determine device type and icon
                    if "cuda" in primary_device:
                        device_icon = "🟢 GPU"
                        device_color = primary_device
                    elif "cpu" in primary_device:
                        device_icon = "🔵 CPU"
                        device_color = "CPU"
                    else:
                        device_icon = "❓"
                        device_color = primary_device
                    
                    print(f"{device_icon} {model_key.upper()}: {info['model_name']}")
                    print(f"   Location: {device_color}")
                    print(f"   Status: {info['status']}")
                    
                    if device_info.get("is_distributed"):
                        print(f"   Distributed: ✅ Yes - {device_info.get('devices', [])}")
                    else:
                        print(f"   Distributed: ❌ No")
            else:
                print("No models currently loaded in memory")
            
            print_separator("📈 RECOMMENDATIONS")
            
            recommendations = data.get("recommendations", {})
            models_on_gpu = recommendations.get("models_on_gpu", 0)
            total_models = recommendations.get("total_models_loaded", 0)
            
            print(f"Models on GPU: {models_on_gpu}/{total_models}")
            
            if gpu_hardware.get("cuda_available"):
                if models_on_gpu == 0 and total_models > 0:
                    print("⚠️  Warning: Models loaded but not using GPU")
                elif models_on_gpu > 0:
                    print("✅ Good: Models are utilizing GPU")
                else:
                    print("ℹ️  Info: No models loaded (lazy loading)")
            else:
                print("ℹ️  Info: Running on CPU mode")
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8002")
    except Exception as e:
        print(f"❌ Error: {e}")

def monitor_model_loading():
    """Monitor model loading in real-time"""
    print_separator("🔄 MONITORING MODEL LOADING")
    print("This will monitor GPU usage when you make requests to different models")
    print("In another terminal, try:")
    print("  curl -X POST 'http://localhost:8002/generate/qwen' -d '{\"prompt\":\"Hello\"}' -H 'Content-Type: application/json'")
    print("\nPress Ctrl+C to stop monitoring...\n")
    
    try:
        while True:
            # Get current status
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                
                # Show loaded models and GPU usage
                models = data.get("models", {})
                gpu_info = data.get("gpu_info", {})
                
                loaded_models = [k for k, v in models.items() if v["in_memory"]]
                
                print(f"\r⏰ {time.strftime('%H:%M:%S')} | Loaded: {loaded_models or 'None'}", end="")
                
                if gpu_info.get("cuda_available"):
                    for gpu in gpu_info.get("devices", []):
                        print(f" | GPU{gpu['id']}: {gpu['memory_allocated_gb']}GB", end="")
                
                sys.stdout.flush()
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

def main():
    print("🚀 GPU Monitoring for Multi-Model Coder API")
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_model_loading()
    else:
        check_gpu_status()
        
        print(f"\n💡 Usage:")
        print(f"  python {sys.argv[0]}          - Check current GPU status")
        print(f"  python {sys.argv[0]} monitor  - Monitor GPU usage in real-time")

if __name__ == "__main__":
    main()