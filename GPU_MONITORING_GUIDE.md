# GPU Monitoring Guide

## 🎯 How to Know if Your Model is Loaded on GPU

You now have comprehensive GPU monitoring built into your API! Here's how to use it:

## 📊 Check GPU Status

### 1. Basic Health Check (includes GPU info)
```bash
curl -s "http://localhost:8002/health" | python -m json.tool
```

This will show you:
- ✅ Which models are loaded
- 🖥️ GPU memory usage per device
- 📍 Where each model is located (GPU/CPU)
- 💾 System memory information

### 2. Detailed GPU Status
```bash
curl -s "http://localhost:8002/gpu" | python -m json.tool
```

This gives you:
- 🔧 GPU hardware details (name, memory, utilization)
- 📍 Exact device location for each model
- 💡 Recommendations for optimization

### 3. Models Overview
```bash
curl -s "http://localhost:8002/models" | python -m json.tool
```

## 🔍 What to Look For

### Model is on GPU ✅
```json
{
  "device_info": {
    "primary_device": "cuda:0",
    "devices": ["cuda:0"],
    "is_distributed": false
  }
}
```

### Model is on CPU ⚠️
```json
{
  "device_info": {
    "primary_device": "cpu",
    "devices": ["cpu"],
    "is_distributed": false
  }
}
```

### Multi-GPU Distribution 🚀
```json
{
  "device_info": {
    "primary_device": "cuda:0",
    "devices": ["cuda:0", "cuda:1"],
    "is_distributed": true,
    "device_map": {"layer.0": "cuda:0", "layer.1": "cuda:1"}
  }
}
```

## 🧪 Test Model Loading

### Step 1: Check initial status (no models loaded)
```bash
curl -s "http://localhost:8002/gpu"
```

### Step 2: Load a model by making a request
```bash
curl -X POST "http://localhost:8002/generate/qwen" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello world", "max_tokens": 50}'
```

### Step 3: Check GPU status again (model now loaded)
```bash
curl -s "http://localhost:8002/gpu"
```

You should see:
- 📈 GPU memory usage increased
- 🎯 Model location shows `cuda:0` (or similar)
- ✅ Model status is "ready"

### Step 4: Switch to another model
```bash
curl -X POST "http://localhost:8002/generate/deepseek" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a function", "max_tokens": 50}'
```

### Step 5: Check GPU status (model swapped)
```bash
curl -s "http://localhost:8002/gpu"
```

You should see:
- 🔄 Previous model unloaded
- 📍 New model loaded on GPU
- 💾 Similar memory usage (one model at a time)

## 🚨 Troubleshooting

### Model Loading on CPU Instead of GPU
If your model loads on CPU when GPU is available:

1. **Check CUDA availability:**
   ```bash
   curl -s "http://localhost:8002/gpu" | grep "cuda_available"
   ```

2. **Check GPU memory:**
   - Make sure you have enough free GPU memory
   - Each model needs ~6-8GB GPU memory

3. **Force GPU loading:**
   - The API uses `device_map="auto"` which should prefer GPU
   - If CPU loading persists, check GPU drivers/CUDA installation

### GPU Memory Issues
If you get CUDA out of memory errors:

1. **Check memory before loading:**
   ```bash
   curl -s "http://localhost:8002/gpu"
   ```

2. **Unload other models:**
   ```bash
   curl -X POST "http://localhost:8002/models/qwen/unload"
   ```

3. **Use smaller models or quantization (modify config.py):**
   - Set `USE_8BIT=true` or `USE_4BIT=true`

## 💡 Pro Tips

1. **Monitor in real-time:**
   ```bash
   python gpu_monitor.py monitor
   ```

2. **Preload models:**
   ```bash
   curl -X POST "http://localhost:8002/models/qwen/preload"
   ```

3. **Check memory after operations:**
   ```bash
   # Before
   curl -s "http://localhost:8002/gpu" | grep memory_allocated
   
   # Load model
   curl -X POST "http://localhost:8002/generate/qwen" -d '{"prompt":"test"}'
   
   # After
   curl -s "http://localhost:8002/gpu" | grep memory_allocated
   ```

## 🎯 Success Indicators

✅ **Model successfully loaded on GPU:**
- `primary_device` shows `cuda:X`
- GPU memory usage increased
- Model status is `ready`
- Generation is fast (~1-3 seconds)

❌ **Model loaded on CPU (slower):**
- `primary_device` shows `cpu`
- GPU memory unchanged
- Generation is slower (10+ seconds)

The API automatically handles device placement, but this monitoring helps you verify everything is working optimally!