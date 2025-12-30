# Running Pipecat Locally with Docker & GPU

This guide details the local Docker implementation for running a fully GPU-accelerated Pipecat conversational AI agent. The setup is designed for privacy and performance, running all inference (ASR, LLM, TTS) locally on your machine.

## Architecture

The system uses `docker-compose` to orchestrate three separate services, all communicating over the host network for optimal WebRTC performance.

### 1. **Pipecat Service (`pipecat`)**
- **Role**: The core application logic and audio pipeline orchestrator.
- **Runtime**: Python 3.12.
- **Components**:
  - **Transport**: `SmallWebRTCTransport` (Default WebRTC implementation).
  - **VAD**: `SileroVADAnalyzer` (Voice Activity Detection).
  - **STT**: `WhisperSTTService` (local `faster-whisper` inference).
  - **LLM Client**: `OpenAILLMService` (connects to the local Ollama service).
  - **TTS Client**: `XTTSService` (connects to the local XTTS service).
- **Networking**: `network_mode: host` (Crucial for WebRTC connectivity).

### 2. **LLM Service (`ollama`)**
- **Role**: Provides large language model inference via an OpenAI-compatible API.
- **Image**: `ollama/ollama:latest`
- **Model**: Default is `tinyllama` (configurable).
- **Port**: `11434`
- **Hardware**: NVIDIA GPU acceleration.

### 3. **TTS Service (`xtts`)**
- **Role**: Provides high-quality text-to-speech synthesis using Coqui XTTS v2.
- **Image**: `ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121`
- **Port**: `8000` (mapped from container port `80`).
- **Hardware**: NVIDIA GPU acceleration.

---

## Prerequisites

- **Docker Desktop** (or Engine) installed.
- **NVIDIA GPU** with drivers installed.
- **NVIDIA Container Toolkit** installed (to allow Docker to access the GPU).

## Quick Start

1.  **Start the Stack**:
    Run the following command in the project root to build and start all services:
    ```bash
    docker compose up --build -d
    ```

2.  **Wait for Initialization**:
    *First-time startup will be slower as models are downloaded.*
    - **XTTS**: Downloads ~2GB model. Check status: `docker compose logs -f xtts`
    - **Ollama**: Downloads the `tinyllama` model.
    - **Whisper**: Downloads the STT model inside the Pipecat container when the bot first initializes.

3.  **Connect**:
    Open your browser to:
    **[http://localhost:7860/client](http://localhost:7860/client)**

    Click **Connect** and allow microphone access. Say "Hello" to start the conversation!

---

## Configuration Details

### `docker-compose.yml`

The compose file defines the interactions and resource allocations. Key configurations:

- **GPU Reservation**: All services request `capabilities: [gpu]` to ensure access to the NVIDIA driver.
- **Volumes**:
  - `~/.cache/huggingface`: Mounted to persist Whisper models, preventing re-downloads.
  - `/mnt/LLM/gguf`: Mounted to persist Ollama models.
  - `.:/app`: The local directory is mounted to `/app` in the Pipecat container, enabling **hot-reloading** or immediate code changes without rebuilding the image.
- **Networking**: The `pipecat` service uses `network_mode: host` to bypass Docker NAT, which is often problematic for WebRTC UDP packets.

### `bot.py` Pipeline

The `bot.py` script constructs the processing pipeline:

```python
pipeline = Pipeline([
    transport.input(),   # Mic input
    stt,                 # Whisper STT (Local)
    tma_in,              # Context Aggregator
    llm,                 # Ollama LLM (Local via HTTP)
    tma_out,             # Context Aggregator
    tts,                 # XTTS TTS (Local via HTTP)
    transport.output(),  # Speaker output
])
```

## Operations

### Changing the LLM Model
1.  **Pull the new model** using the running Ollama container:
    ```bash
    docker exec -it pipecat-ollama-1 ollama pull mistral
    ```
2.  **Update `docker-compose.yml`**:
    Change `LLM_MODEL=tinyllama` to `LLM_MODEL=mistral`.
3.  **Restart**:
    ```bash
    docker compose up -d
    ```

### Changing the Voice
Edit `bot.py` and change the `voice_id` in the `XTTSService` initialization:

```python
tts = XTTSService(
    voice_id="Claribel Dervla", # Change this string
    base_url="http://localhost:8000",
    aiohttp_session=session,
)
```
*Note: You can query available speakers from the XTTS server API if needed.*

### Viewing Logs
To debug issues or see the conversation flow:

```bash
# All logs
docker compose logs -f

# Specific service logs
docker compose logs -f pipecat
docker compose logs -f xtts
```

## Troubleshooting

- **"StartFrame not received yet"**:
    This error appears if the Pipecat bot starts before the XTTS or Ollama services are fully ready.
    **Fix**: Wait a moment for the other services to initialize (check logs), then restart the bot:
    `docker compose restart pipecat`

- **Audio not working?**:
    Ensure you are using `localhost` or accessing the machine via an IP that is reachable. Since `network_mode: host` is used, the container shares the host's network stack. Check your firewall settings if connecting from a different machine.
