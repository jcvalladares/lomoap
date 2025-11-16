# Distilled Models Guide

## 🎯 What Are Distilled Models?

Distilled models are smaller, faster versions of large models that maintain most of the performance while using significantly less resources.

## 📊 Model Comparison

| Model | Size | Memory Usage | Speed | Use Case |
|-------|------|--------------|-------|----------|
| **Qwen 0.5B** | 0.5B params | ~1-2GB | ⚡⚡⚡ Fastest | Quick code snippets, simple tasks |
| **Qwen 1.5B** | 1.5B params | ~2-3GB | ⚡⚡ Fast | Balanced performance/speed |
| **CodeGemma 2B** | 2B params | ~3-4GB | ⚡ Good | Google's specialized code model |
| **DeepSeek Lite Base** | ~2-6B params | ~4-6GB | 🐌 Slower | More complex code generation |

## 🚀 Benefits of Distilled Models

### ⚡ **Speed Improvements**
- **Loading Time:** 10-30 seconds vs 60-120 seconds
- **Inference Time:** 1-3 seconds vs 5-10 seconds  
- **Memory Bandwidth:** Less data to move around

### 💾 **Memory Efficiency**
- **GPU Memory:** 1-4GB vs 6-12GB
- **System RAM:** 2-6GB vs 10-20GB
- **Multiple Models:** Can fit 2-3 small models vs 1 large model

### 🏃 **Better User Experience**
- **Instant Responses:** Faster generation for interactive use
- **Less Waiting:** Quick model switching
- **Lower Hardware Requirements:** Works on smaller GPUs

## 🧪 API Endpoints

### **Ultra-Fast (Qwen 0.5B)**
```bash
curl -X POST "http://localhost:8002/generate/qwen-tiny" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a hello world function", "max_tokens": 100}'
```

### **Balanced (Qwen 1.5B) - Default**
```bash
curl -X POST "http://localhost:8002/generate/qwen" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Implement quicksort algorithm", "max_tokens": 200}'
```

### **Google Specialized (CodeGemma 2B)**
```bash
curl -X POST "http://localhost:8002/generate/codegemma" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain recursion with example", "max_tokens": 150}'
```

### **More Capable (DeepSeek Lite)**
```bash
curl -X POST "http://localhost:8002/generate/deepseek" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Design a REST API structure", "max_tokens": 250}'
```

## 🎯 When to Use Which Model

### 🏃 **Qwen 0.5B (Ultra-Fast)**
✅ **Best for:**
- Code completion
- Simple functions
- Quick snippets
- Interactive coding assistance

❌ **Not ideal for:**
- Complex algorithms
- Detailed explanations
- Large code blocks

### ⚖️ **Qwen 1.5B (Balanced) - Recommended Default**
✅ **Best for:**
- General code generation
- Function implementations
- Code explanations
- Balanced performance/quality

✅ **Good for most use cases**

### 🎯 **CodeGemma 2B (Specialized)**
✅ **Best for:**
- Google-style code patterns
- Well-documented code
- Educational explanations
- Code with good practices

### 🧠 **DeepSeek Lite (Capable)**
✅ **Best for:**
- Complex algorithms
- System design
- Detailed implementations
- When quality > speed

## 📈 Performance Testing

Run comprehensive performance tests:

```bash
# Test all models with various prompts
python test_distilled_models.py

# Monitor GPU usage during tests
python gpu_monitor.py monitor
```

## 💡 Optimization Tips

### 1. **Choose the Right Model for Your Task**
```python
# For quick completion
response = requests.post("/generate/qwen-tiny", json={"prompt": "def hello():", "max_tokens": 50})

# For balanced tasks  
response = requests.post("/generate/qwen", json={"prompt": "implement binary search", "max_tokens": 150})

# For complex algorithms
response = requests.post("/generate/deepseek", json={"prompt": "design distributed system", "max_tokens": 300})
```

### 2. **Preload Your Most-Used Model**
```bash
# Preload the model you use most often
curl -X POST "http://localhost:8002/models/qwen/preload"
```

### 3. **Adjust max_tokens Based on Model**
```python
# Smaller models work well with fewer tokens
tiny_request = {"prompt": "hello world", "max_tokens": 50}

# Larger models can handle more tokens effectively  
complex_request = {"prompt": "system design", "max_tokens": 400}
```

### 4. **Monitor Performance**
```bash
# Check which model is currently loaded
curl "http://localhost:8002/health"

# Check detailed GPU usage
curl "http://localhost:8002/gpu"
```

## 🔧 Configuration Options

You can further optimize by modifying the model configurations in `main.py`:

```python
# For even faster loading (with some quality trade-off)
"config": {
    "torch_dtype": torch.float16,
    "device_map": "auto", 
    "load_in_8bit": True,  # Enable 8-bit quantization
    "low_cpu_mem_usage": True
}
```

## 🎯 Expected Performance

### **Loading Times**
- Qwen 0.5B: ~10-15 seconds
- Qwen 1.5B: ~15-25 seconds  
- CodeGemma 2B: ~20-30 seconds
- DeepSeek Lite: ~30-45 seconds

### **Generation Speed** (approximate)
- Qwen 0.5B: ~50-100 tokens/second
- Qwen 1.5B: ~30-60 tokens/second
- CodeGemma 2B: ~20-40 tokens/second  
- DeepSeek Lite: ~15-30 tokens/second

### **Memory Usage**
- Qwen 0.5B: ~1-2GB GPU memory
- Qwen 1.5B: ~2-3GB GPU memory
- CodeGemma 2B: ~3-4GB GPU memory
- DeepSeek Lite: ~4-6GB GPU memory

## 🚀 Getting Started

1. **Start the API:**
   ```bash
   python main.py
   ```

2. **Try the fastest model:**
   ```bash
   curl -X POST "http://localhost:8002/generate/qwen-tiny" \
        -d '{"prompt": "def factorial(n):"}'
   ```

3. **Compare with balanced model:**
   ```bash
   curl -X POST "http://localhost:8002/generate/qwen" \
        -d '{"prompt": "def factorial(n):"}'
   ```

The distilled models give you the best of both worlds: good performance with much faster response times! 🎉