#!/usr/bin/env python3
"""
Quick demonstration of distilled models
"""

import json

# Show the model configurations
print("🎯 Distilled Multi-Model Coder API")
print("="*50)

models = {
    "qwen-tiny": {
        "name": "Qwen2.5-Coder-0.5B-Instruct",
        "size": "0.5B parameters",
        "memory": "~1-2GB",
        "speed": "⚡⚡⚡ Ultra-fast",
        "use_case": "Quick code snippets, completions"
    },
    "qwen": {
        "name": "Qwen2.5-Coder-1.5B-Instruct", 
        "size": "1.5B parameters",
        "memory": "~2-3GB",
        "speed": "⚡⚡ Fast",
        "use_case": "Balanced performance (Default)"
    },
    "codegemma": {
        "name": "google/codegemma-2b",
        "size": "2B parameters", 
        "memory": "~3-4GB",
        "speed": "⚡ Good",
        "use_case": "Google's specialized code model"
    },
    "deepseek": {
        "name": "DeepSeek-Coder-V2-Lite-Base",
        "size": "~6B parameters",
        "memory": "~4-6GB", 
        "speed": "🐌 Slower but capable",
        "use_case": "Complex algorithms, detailed code"
    }
}

print("\n📋 Available Distilled Models:")
print("-" * 50)

for key, info in models.items():
    print(f"\n🤖 {key.upper()}:")
    print(f"   Model: {info['name']}")
    print(f"   Size: {info['size']}")
    print(f"   Memory: {info['memory']}")
    print(f"   Speed: {info['speed']}")
    print(f"   Best for: {info['use_case']}")

print(f"\n🚀 Key Benefits of Distilled Models:")
print(f"✅ Faster inference (1-3s vs 5-10s)")
print(f"✅ Lower memory usage (1-4GB vs 6-12GB)")
print(f"✅ Quicker loading (10-30s vs 60-120s)")
print(f"✅ Better interactive experience")
print(f"✅ Can fit multiple models on one GPU")

print(f"\n🧪 Test Commands:")
print("-" * 30)

test_commands = [
    ("Ultra-fast completion", "curl -X POST 'http://localhost:8002/generate/qwen-tiny' -d '{\"prompt\":\"def hello():\"}'"),
    ("Balanced performance", "curl -X POST 'http://localhost:8002/generate/qwen' -d '{\"prompt\":\"implement quicksort\"}'"),  
    ("Google specialized", "curl -X POST 'http://localhost:8002/generate/codegemma' -d '{\"prompt\":\"explain recursion\"}'"),
    ("More capable", "curl -X POST 'http://localhost:8002/generate/deepseek' -d '{\"prompt\":\"design REST API\"}'")
]

for desc, cmd in test_commands:
    print(f"\n{desc}:")
    print(f"  {cmd}")

print(f"\n💡 Recommendation:")
print(f"Start with 'qwen' (1.5B) for most tasks - it offers the best balance")
print(f"of performance and speed. Use 'qwen-tiny' when you need instant responses!")

print(f"\n🎮 Interactive Testing:")
print(f"Run: python test_distilled_models.py")
print(f"Monitor: python gpu_monitor.py")

if __name__ == "__main__":
    print(f"\n📝 Configuration saved to model_configs.json")
    
    # Save configurations to a JSON file for reference
    with open("model_configs.json", "w") as f:
        json.dump(models, f, indent=2)
    
    print(f"✅ Ready to use distilled models!")