# LLM Model Inventory & Configuration Guide

## Your Available Models (RTX4090 Optimized)

This document maps all available models in your `/mnt/LLM` storage and how to use them with pipecat.

### Storage Summary
```
/mnt/LLM/
├── gguf/               (18GB) - Ollama GGUF models + manifests
├── lmstudio/models/    (420GB) - LM Studio GGUF models
├── cache/huggingface/  (17GB) - HuggingFace format models
├── cache/transformers/ - Transformers library cache
├── models/             (449MB) - Other model formats
└── asr/                (617MB) - Speech recognition models
```

---

## Quick Reference: Recommended Models for Pipecat

### ⚡ Fastest (Real-Time Optimized - 1-2s latency)

| Model | Size | Format | Latency | Quality | Use Case |
|-------|------|--------|---------|---------|----------|
| **TinyLlama-1.1B-Chat** | 1.1B | HF Cache | ~50ms | Basic Q&A | Testing, fast responses |
| **Mistral-7B** | 7B | HF Cache | ~100ms | Good | Balanced |
| **Llama-3.1-8B-Instruct** | 8B | HF Cache | ~120ms | Excellent | Production recommended |

### ⚖️ Balanced (Recommended for Production)

| Model | Size | Format | Source | Tokens/sec | VRAM |
|-------|------|--------|--------|------------|------|
| **Llama-2-7b-hf** | 7B | HF Cache | `/mnt/LLM/cache/huggingface` | 80-100 | 14GB |
| **Qwen2.5-14B-Instruct** | 14B | GGUF | `/mnt/LLM/lmstudio` | 40-60 | 28GB |
| **Ministral-8B-Instruct** | 8B | GGUF | `/mnt/LLM/lmstudio` | 100+ | 16GB |

### 🚀 High-Quality (Complex reasoning)

| Model | Size | Format | Specialized |
|-------|------|--------|------------|
| **Qwen3-32B** | 32B | GGUF | General purpose (RTX4090 can handle w/ quantization) |
| **Ollama-3-32B-Think** | 32B | GGUF | Reasoning-heavy tasks |
| **DeepSeek-Coder-V2-Lite** | 8B | GGUF | Code generation |

### 👁️ Vision Models (Multimodal)

| Model | Size | Capability |
|-------|------|-----------|
| **Llama-3.2-11B-Vision-Instruct** | 11B | Image understanding + chat |
| **Pixtral-12b** | 12B | Vision-specialized from Mistral |

---

## Detailed Model Listings

### In HuggingFace Cache (`/mnt/LLM/cache/huggingface/`)

These models are in native HF format, ideal for vLLM:

```
✓ TinyLlama/TinyLlama-1.1B-Chat-v1.0          (1.1B, fastest)
✓ meta-llama/Llama-2-7b-hf                    (7B, recommended base)
✓ Systran/faster-whisper-large-v3             (ASR, already cached)
✓ jonatasgrosman/wav2vec2-large-xlsr-53-portuguese  (Speech features)
✓ oliverguhr/fullstop-punctuation-multilang-large   (Text processing)
```

**Use with vLLM:**
```bash
docker-compose up --build
# or modify environment in docker-compose.yml:
# LLM_MODEL=meta-llama/Llama-2-7b-hf
```

### In LM Studio (`/mnt/LLM/lmstudio/models/`)

These are GGUF-quantized models, ideal for Ollama or direct inference:

#### By Organization

**Bartowski (Lightweight/Optimized):**
```
✓ bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF    (8B, code)
✓ bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF       (14B, reasoning)
✓ bartowski/Ministral-8B-Instruct-2410-GGUF         (8B, balanced)
✓ bartowski/mistral-community_pixtral-12b-GGUF      (12B, vision)
```

**LM Studio Community (Diverse Models):**
```
✓ lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF
✓ lmstudio-community/gemma-3-27B-it-qat-GGUF       (27B, Google model)
✓ lmstudio-community/gpt-oss-20b-GGUF              (20B, open model)
✓ lmstudio-community/gpt-oss-120b-GGUF             (120B, multi-part)
✓ lmstudio-community/Llama-3.3-70B-Instruct-GGUF   (70B, very large)
✓ lmstudio-community/Magistral-Small-2509-GGUF    (Small, multimodal)
✓ lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
✓ lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF
✓ lmstudio-community/Olmo-3-32B-Think-GGUF        (32B, reasoning)
✓ lmstudio-community/Qwen2.5-Coder-14B-Instruct-GGUF
✓ lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF
✓ lmstudio-community/Qwen3-32B-GGUF               (32B, latest Qwen)
✓ lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF
✓ lmstudio-community/Seed-OSS-36B-Instruct-GGUF
✓ lmstudio-community/Hermes-4-70B-GGUF            (70B, very large)
```

**Leafspark (Vision):**
```
✓ leafspark/Llama-3.2-11B-Vision-Instruct-GGUF    (11B, vision)
```

**Qwen (Official):**
```
✓ Qwen/Qwen2.5-14B-Instruct-GGUF
✓ Qwen/Qwen2.5-72B-Instruct-GGUF                  (72B, very large)
✓ Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
✓ Qwen/Qwen2.5-Coder-32B-Instruct-GGUF
```

**Other:**
```
✓ mrm8488/phi-4-14B-grpo-gsm8k-3e-q4
✓ RichardErkhov/mistralai_-_Mixtral-8x7B-Instruct-v0.1-gguf
```

---

## How to Use Different Models with Pipecat

### Option 1: Use vLLM with HuggingFace Format Models

**Best for:** Fastest setup, best quality, automatic quantization

Edit `docker-compose.gpu.yml`:
```yaml
vllm:
  environment:
    - MODEL=meta-llama/Llama-2-7b-hf    # Change this line
```

Or via environment variable:
```bash
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 docker-compose up
```

**Recommended models:**
```
meta-llama/Llama-2-7b-hf              (7B, balanced)
TinyLlama/TinyLlama-1.1B-Chat-v1.0    (1.1B, fastest)
mistralai/Mistral-7B-v0.1             (7B, excellent)
```

### Option 2: Use Ollama with GGUF Models

**Best for:** Lightweight, quantized, minimal latency

Start with Ollama profile:
```bash
docker-compose --profile ollama up --build
```

Pull a model:
```bash
docker exec ollama ollama pull mistral
```

Available GGUF models from your storage:
```bash
# Small (fast, suitable for voice)
docker exec ollama ollama pull bartowski/Ministral-8B-Instruct-2410-GGUF

# Medium (balanced)
docker exec ollama ollama pull lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF

# Large (slow, high quality)
docker exec ollama ollama pull lmstudio-community/Llama-3.3-70B-Instruct-GGUF
```

### Option 3: Direct GGUF Inference (LM Studio API)

**Best for:** Direct control, custom quantization

```bash
docker-compose --profile lmstudio up --build
```

API endpoint: `http://localhost:8090/v1`

---

## Model Comparison for Real-Time Voice AI

| Metric | TinyLlama | Mistral-7B | Llama-2-7B | Qwen-14B | Llama-70B |
|--------|-----------|-----------|-----------|----------|-----------|
| Inference Speed (tokens/sec) | 150+ | 100+ | 80 | 40 | 10 (quantized) |
| Context Length | 2048 | 8192 | 4096 | 32K | 4096 |
| VRAM Required (fp16) | 2GB | 14GB | 14GB | 28GB | 140GB |
| Quality (1-10) | 5 | 8 | 8 | 8.5 | 9.5 |
| Voice Response Time* | 300ms | 500ms | 600ms | 1000ms | 2000ms+ |
| Recommended Use | Testing | Production | Production | Complex | Offline only |

*Total time: STT + LLM + TTS

---

## Optimal Configurations for RTX4090

### Configuration 1: Blazing Fast (Preferred for Live Voice)
```
Model: TinyLlama-1.1B-Chat-v1.0
Speed: 50ms LLM latency
Quality: Basic
Use: Live testing, rapid prototyping
```

**Enable:**
```yaml
vllm:
  command: >
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    --dtype float16
    --gpu-memory-utilization 0.5
```

### Configuration 2: Balanced (Recommended for Production)
```
Model: Llama-2-7b-hf or Mistral-7B
Speed: 100-120ms LLM latency
Quality: Excellent
Use: Production voice agents
```

**Default in docker-compose.yml**

### Configuration 3: High Quality (Reasoning Heavy)
```
Model: Qwen2.5-14B-Instruct (GGUF) or via vLLM
Speed: 40-60ms tokens/sec (need quantization)
Quality: Excellent reasoning
Use: Complex domain tasks
```

**Enable:**
```bash
# vLLM
LLM_MODEL=Qwen/Qwen2.5-14B-Instruct docker-compose up

# Or Ollama
docker exec ollama ollama pull lmstudio-community/Qwen3-32B-GGUF
```

---

## Model Selection Decision Tree

```
Start here:
├─ Is this for TESTING?
│  └─ Use: TinyLlama-1.1B-Chat
│
├─ Need PRODUCTION VOICE AI?
│  └─ Is response quality critical?
│     ├─ NO: Use Mistral-7B or Llama-3.1-8B
│     └─ YES: Use Qwen2.5-14B or Llama-2-13B
│
├─ Need CODE GENERATION?
│  └─ Use: DeepSeek-Coder-V2-Lite (8B)
│
├─ Need REASONING/THINKING?
│  └─ Use: Olmo-3-32B-Think or Qwen3-32B
│
└─ Need VISION (image understanding)?
   └─ Use: Llama-3.2-11B-Vision or Pixtral-12b
```

---

## Performance Tuning Parameters

### For Faster Inference
```yaml
vllm:
  environment:
    - GPU_MEMORY_UTILIZATION=0.95     # Increase to 0.95
    - DTYPE=float16                    # Keep for speed
    - MAX_MODEL_LEN=2048               # Reduce from 4096
```

### For Better Quality
```yaml
vllm:
  environment:
    - GPU_MEMORY_UTILIZATION=0.9
    - DTYPE=bfloat16                   # Higher precision
    - MAX_MODEL_LEN=4096               # Full context
```

### For Memory-Constrained (if needed)
```yaml
vllm:
  environment:
    - GPU_MEMORY_UTILIZATION=0.8
    - DTYPE=float16
    - MAX_MODEL_LEN=2048
    - QUANTIZATION=awq               # Add quantization
```

---

## Troubleshooting Model Loading

### Model not found error
```bash
# Check if model is cached
ls -la /mnt/LLM/cache/huggingface/hub/ | grep -i "model-name"

# Pre-download if missing
docker run --rm \
  -v /mnt/LLM/cache/huggingface:/root/.cache/huggingface \
  huggingface/transformers:latest \
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')"
```

### CUDA out of memory with large models
```bash
# Use smaller model
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 docker-compose up

# Or reduce max length
# Change MAX_MODEL_LEN from 4096 to 1024 in docker-compose.yml
```

### Slow inference
```bash
# Check GPU utilization
nvidia-smi dmon -s pucvmet

# If low, increase batch size or GPU memory utilization
# If high but slow, switch to faster model (TinyLlama)
```

---

## Environment Variables for docker-compose

Create a `.env` file:
```bash
# Speech Services
DEEPGRAM_API_KEY=your_key
OPENAI_API_KEY=your_key
CARTESIA_API_KEY=your_key

# HuggingFace (for gated models)
HUGGING_FACE_HUB_TOKEN=your_token

# LLM Selection (optional, overrides docker-compose)
LLM_MODEL=meta-llama/Llama-2-7b-hf
LLM_BACKEND=vllm  # or ollama
```

---

## Testing Models Before Production

```bash
# 1. Start docker-compose
docker-compose up -d

# 2. Test vLLM API
curl http://localhost:8000/v1/models

# 3. Quick inference test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Say hi!"}],
    "max_tokens": 50
  }'

# 4. Monitor performance
nvidia-smi -l 1  # 1 second updates

# 5. Stop when done
docker-compose down
```

---

## Summary: Quick Commands

```bash
# Use default (vLLM with Llama-2-7b)
docker-compose up --build

# Use TinyLlama (fastest)
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 docker-compose up

# Use Ollama instead
docker-compose --profile ollama up

# View available models in Ollama
docker exec ollama ollama list

# Pull a model in Ollama
docker exec ollama ollama pull mistral

# Test the bot
curl http://localhost:8000/v1/models  # Check vLLM
curl http://localhost:11434/tags      # Check Ollama
```

---

## Storage Usage by Model Category

```
HuggingFace Cache (17GB total):
  └─ Active: TinyLlama (300MB), WhisperV3 (3GB), PointPillars, etc.

GGUF Models (420GB total):
  ├─ Small (1-8B):   ~80GB
  ├─ Medium (14-32B): ~180GB  
  ├─ Large (70B+):   ~150GB
  └─ Multimodal:      ~10GB

Total Used: 437GB
Available: ~600GB more on /mnt/LLM if needed
```

---

## Next Steps

1. **Start Simple:** Run with default Llama-2-7b
2. **Benchmark:** Test different models to find your sweet spot
3. **Optimize:** Adjust GPU_MEMORY_UTILIZATION and MAX_MODEL_LEN
4. **Integrate:** Modify bot.py to use your selected model
5. **Deploy:** Use Pipecat CLI for cloud deployment

Happy experimenting! 🚀
