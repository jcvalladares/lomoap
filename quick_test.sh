#!/bin/bash

# Quick test script to demonstrate distilled model performance
# Make sure the API is running first: python main.py

echo "🚀 Testing Distilled Models Performance"
echo "======================================"

API_URL="http://localhost:8002"

# Check if API is running
if ! curl -s "$API_URL/" > /dev/null; then
    echo "❌ API not running. Please start with: python main.py"
    exit 1
fi

echo "✅ API is running"

# Test prompt
PROMPT="Write a Python function to check if a number is prime"

echo -e "\n🧪 Testing all models with prompt: '$PROMPT'"
echo "=================================================="

# Test each model
models=("qwen-tiny:Ultra-Fast(0.5B)" "qwen:Balanced(1.5B)" "codegemma:Specialized(2B)" "deepseek:Capable(Lite)")

for model_info in "${models[@]}"; do
    IFS=':' read -r model desc <<< "$model_info"
    
    echo -e "\n🤖 Testing $desc"
    echo "Endpoint: /generate/$model"
    
    start_time=$(date +%s.%3N)
    
    response=$(curl -s -X POST "$API_URL/generate/$model" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 100}" 2>/dev/null)
    
    end_time=$(date +%s.%3N)
    
    if [ $? -eq 0 ] && echo "$response" | grep -q "generated_text"; then
        duration=$(echo "$end_time - $start_time" | bc)
        length=$(echo "$response" | jq -r '.generated_text | length' 2>/dev/null || echo "unknown")
        preview=$(echo "$response" | jq -r '.generated_text' 2>/dev/null | head -c 80 || echo "...")
        
        echo "⏱️  Time: ${duration}s"
        echo "📏 Length: $length characters"  
        echo "📝 Preview: $preview..."
    else
        echo "❌ Failed or loading (first request loads model)"
    fi
    
    # Small delay between tests
    sleep 1
done

echo -e "\n📊 Summary:"
echo "==========="
echo "🏆 qwen-tiny: Fastest loading and inference"
echo "⚖️  qwen: Best balance of speed and quality (recommended)"
echo "🎯 codegemma: Google's specialized approach"
echo "🧠 deepseek: More capable for complex tasks"

echo -e "\n💡 Tips:"
echo "- First request to each model will be slower (model loading)"
echo "- Subsequent requests will be much faster"  
echo "- Use 'qwen' for most tasks"
echo "- Use 'qwen-tiny' when you need instant responses"

echo -e "\n🔧 Advanced testing:"
echo "python test_distilled_models.py    # Comprehensive performance test"
echo "python gpu_monitor.py             # Monitor GPU usage"