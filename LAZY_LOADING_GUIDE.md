# Multi-Model Coder API - Lazy Loading Strategy

## 🎯 Problem Solved
Instead of loading both models at startup (which would consume ~14-20GB RAM), we implemented **lazy loading** with **model swapping** to optimize memory usage.

## 🚀 How It Works

### 1. **Lazy Loading**
- API starts instantly (no model loading at startup)
- Models are loaded only when first requested
- Subsequent requests to the same model are fast (already in memory)

### 2. **Model Swapping**
- Only **one model in memory at a time**
- When you request a different model, the current model is unloaded first
- This keeps memory usage to ~7-10GB instead of 14-20GB

### 3. **Smart Caching**
- Once loaded, models stay in memory for fast responses
- Automatic cleanup when switching models
- Manual preload/unload endpoints for control

## 📊 Memory Strategy Comparison

| Strategy | Startup Time | Memory Usage | First Request | Subsequent Requests |
|----------|---------------|--------------|---------------|-------------------|
| **Load All at Startup** | 2-5 minutes | 14-20GB | Fast (~1s) | Fast (~1s) |
| **Lazy Loading** | Instant | 7-10GB | Slow (~30s) | Fast (~1s) |
| **No Caching** | Instant | Minimal | Very Slow (always loading) | Very Slow |

## 🔧 API Endpoints

### Generation Endpoints
- `POST /generate/qwen` - Generate with Qwen model
- `POST /generate/deepseek` - Generate with DeepSeek model  
- `POST /generate` - Generate with default model (Qwen)

### Management Endpoints
- `GET /health` - Check API and model status
- `GET /models` - List available models and their status
- `POST /models/{model}/preload` - Manually preload a model
- `POST /models/{model}/unload` - Manually unload a model

## 📝 Usage Examples

### Basic Generation
```bash
# First request to Qwen (will load model - takes ~30 seconds)
curl -X POST "http://localhost:8002/generate/qwen" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a Python function", "max_tokens": 150}'

# Second request to Qwen (fast - model already loaded)
curl -X POST "http://localhost:8002/generate/qwen" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a sorting algorithm", "max_tokens": 150}'

# Request to DeepSeek (will unload Qwen and load DeepSeek)
curl -X POST "http://localhost:8002/generate/deepseek" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain recursion", "max_tokens": 150}'
```

### Manual Model Management
```bash
# Preload a model (useful for warming up)
curl -X POST "http://localhost:8002/models/qwen/preload"

# Check status
curl "http://localhost:8002/health"

# Unload a model (free memory)
curl -X POST "http://localhost:8002/models/qwen/unload"
```

## 💡 Best Practices

### For Development
1. **Start with lazy loading** - instant startup
2. **Use preload endpoint** to warm up the model you're testing
3. **Monitor memory usage** via `/health` endpoint

### For Production
1. **Preload your primary model** during deployment
2. **Use a reverse proxy** with proper timeouts for first requests
3. **Monitor memory and swap patterns** based on usage

### For Multiple Models
1. **Profile your usage patterns** - do users switch between models often?
2. **Consider dedicated instances** if you need both models simultaneously
3. **Use the unload endpoint** to free memory during low usage periods

## 🎉 Benefits

✅ **Instant startup** - API ready in seconds, not minutes
✅ **Memory efficient** - Only one model loaded at a time  
✅ **Flexible** - Load models on demand
✅ **Production ready** - Proper error handling and status monitoring
✅ **User friendly** - Clear status messages and loading indicators

This approach gives you the best of both worlds: fast startup and efficient memory usage!