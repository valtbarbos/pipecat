# Running SOTA Pipecat Locally with Docker & GPU

This guide details the local Docker implementation for running a fully GPU-accelerated Pipecat conversational AI agent. The setup is optimized for **ultra-low latency**, privacy, and "State Of The Art" (SOTA) agentic capabilities.

## Architecture

The system uses `docker-compose` to orchestrate three services with full NVIDIA GPU passthrough and host-mode networking for seamless WebRTC performance.

### 1. **Pipecat Service (`pipecat`)**
- **Runtime**: Python 3.12 on **CUDA 12.1 Devel** (Ubuntu 22.04).
- **Core Strategy**: High-concurrency, low-latency pipeline using `asyncio`.
- **Key Features**:
  - **STT**: `Whisper Large V3 Turbo` (Local GPU) for near-instant, punctuated transcription.
  - **VAD (Real-time Feedback)**: `SileroVAD` tuned to **0.3s** to provide immediate transcription feedback on the UI.
  - **Turn Analysis**: `LocalSmartTurnAnalyzerV3` provides semantic conversation control; it prevents the bot from interrupting you even during long thinking pauses.
  - **LLM Client**: `OpenAILLMService` (connects to local Ollama `gemma3:27b`).
  - **TTS Client**: `XTTSService` (connects to local XTTS service).
  - **Memory**: `Mem0MemoryService` (ready for persistent user context).

### 2. **LLM Service (`ollama`)**
- **Role**: Provides the "brain" using **Gemma 3 27B**.
- **Hardware**: Full GPU acceleration via `nvidia-container-toolkit`.

### 3. **TTS Service (`xtts`)**
- **Role**: Coqui XTTS v2 for natural, expressive voice synthesis.

---

## Prerequisites

- **Docker Desktop** (or Engine) with NVIDIA Container Toolkit.
- **NVIDIA GPU**: RTX 3090/4090 recommended for SOTA model performance.
- **CUDA 12.x** compatible host drivers.

## Installation & Start

1.  **Build and Start**:
    The Dockerfile automatically configures **cuDNN 9** and Python 3.12 dependencies for GPU STT:
    ```bash
    docker compose up -d --build
    ```

2.  **Download Models**:
    *   **Ollama**: Pull the SOTA model:
        ```bash
        docker exec -it pipecat-ollama-1 ollama pull gemma3:27b
        ```

3.  **Access the Dashboard**:
    Open **[http://localhost:7860/client](http://localhost:7860/client)**.

---

## Real-time Latency Optimizations

The current `bot.py` is tuned for a "premium" conversational feel:
- **Immediate Transcription**: By setting VAD `stop_secs=0.3`, words appear in the UI almost as soon as you say them.
- **Semantic Silence**: We use a `user_turn_end_timeout` of **0.5s**. This means the bot will only think for half a second after you stop speaking before the `LocalSmartTurnAnalyzerV3` decides if your sentence was complete or just a pause.

---

## Troubleshooting

### CUDA / cuDNN Errors
If you see `libcudnn_ops.so` or `cudnnCreateTensorDescriptor` errors:
1. Ensure your host NVIDIA drivers are up to date.
2. The `Dockerfile` uses `nvidia/cuda:12.1.1-devel-ubuntu22.04` and installs `libcudnn9-cuda-12`. This is required for `faster-whisper`.
3. Check GPU status in the container:
   ```bash
   docker exec -it pipecat-pipecat-1 nvidia-smi
   ```

### No Transcription / Agent Silent
1. Check Ollama is running and model is pulled: `docker logs pipecat-ollama-1`.
2. Check XTTS initialization: `docker logs pipecat-xtts-1`.
3. Check Pipecat logs for WebRTC connection status:
   ```bash
   docker logs -f pipecat-pipecat-1
   ```

### Network Issues
WebRTC requires `network_mode: host`. If running on Mac/Windows, ensure Docker Desktop is configured to allow host networking or use specific port mappings (though `host` is preferred for WebRTC).

