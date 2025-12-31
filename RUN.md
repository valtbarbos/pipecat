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
  - **LLM Client**: `OpenAILLMService` (connects to local Ollama `qwen2.5:14b`).
  - **TTS Clients**:
    - `XTTSService`: Connects to local Coqui XTTS service.
    - `ChatterBox`: High-performance streaming TTS integration via the `ResembleAITTSService` protocol (pointing to our local `chatterbox` service).
  - **Dynamic Configuration**: `TTS_SERVICE` allows switching between providers at runtime.
  - **Memory**: `Mem0MemoryService` (ready for persistent user context).
  - **Externalized Instructions**: `SYSTEM_PROMPT_PATH` points to a Markdown file (e.g., `instructions.md`) for easy persona management.

### 2. **LLM Service (`ollama`)**
- **Role**: Provides the "brain" using **Qwen 2.5 14B**.
- **Hardware**: Full GPU acceleration via `nvidia-container-toolkit`.

### 3. **TTS Services**
- **XTTS (`xtts`)**: Coqui XTTS v2 for deep, high-quality voice cloning and expressive synthesis.
- **ChatterBox (`chatterbox`)**: Optimized real-time TTS server providing ultra-low latency streaming audio over WebSockets.

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
        docker exec -it pipecat-ollama-1 ollama pull qwen2.5:14b
        ```

3.  **Access the Dashboard**:
    Open **[http://localhost:7860/client](http://localhost:7860/client)**.

4.  **Managing the Persona**:
    Edit `instructions.md` to change the bot's behavior without restarting the container (the bot reads this file on every new session).

---

## Real-time Latency Optimizations

The current `bot.py` is tuned for a "premium" conversational feel:
- **Immediate Transcription**: By setting VAD `stop_secs=0.3`, words appear in the UI almost as soon as you say them.
- **Semantic Silence**: We use a `user_turn_end_timeout` of **0.7s**. This means the bot will wait a bit longer after you stop speaking before the `LocalSmartTurnAnalyzerV3` decides if your sentence was complete or just a pause, reducing false interruptions.

---

## TTS Voice Options

The local XTTS service comes with 52 pre-defined "Studio" voices. You can change the bot's voice by updating the `TTS_VOICE` environment variable in `docker-compose.yml`.

### Featured Voices
- **Claribel Dervla** (Default): A versatile North American accent. Described as professional, clear, and relatable. Can range from authoritative and sincere to warm and conversational.
- **Daisy Studious**: A clear, articulate voice suitable for educational or instructional content.
- **Gracie Wise**: A friendly and approachable tone with a natural flow.
- **Damien Black**: Noted for its excellent naturalness and narration quality, often preferred for long-form content like audiobooks.
- **Tammie Ema**: A bright and engaging female voice.

### How to Change the Voice
1. Open `docker-compose.yml`.
2. Find the `pipecat` service environment section.
3. Change `TTS_VOICE` to any of the following names:
   > `Ana Florence`, `Andrew Chipper`, `Annmarie Nele`, `Asya Anara`, `Badr Odhiambo`, `Baldur Sanjin`, `Barbora MacLean`, `Brenda Stern`, `Camilla Holmström`, `Chandra MacFarland`, `Claribel Dervla`, `Craig Gutsy`, `Daisy Studious`, `Damien Black`, `Damjan Chapman`, `Dionisio Schuyler`, `Eugenio Mataracı`, `Ferran Simen`, `Filip Traverse`, `Gilberto Mathias`, `Gitta Nikolina`, `Gracie Wise`, `Henriette Usha`, `Ige Behringer`, `Ilkin Urbano`, `Kazuhiko Atallah`, `Kumar Dahl`, `Lidiya Szekeres`, `Lilya Stainthorpe`, `Ludvig Milivoj`, `Luis Moray`, `Maja Ruoho`, `Marcos Rudaski`, `Narelle Moon`, `Nova Hogarth`, `Rosemary Okafor`, `Royston Min`, `Sofia Hellen`, `Suad Qasim`, `Szofi Granger`, `Tammie Ema`, `Tammy Grit`, `Tanja Adelina`, `Torcull Diarmuid`, `Uta Obando`, `Viktor Eka`, `Viktor Menelaos`, `Vjollca Johnnie`, `Wulf Carlevaro`, `Xavier Hayasaka`, `Zacharie Aimilios`, `Zofija Kendrick`

4. Restart the stack:
   ```bash
    docker compose restart pipecat
    ```

## Selecting the TTS Service

You can switch between the two available TTS engines via environment variables in `.env` or `docker-compose.yml`:

- `TTS_SERVICE=xtts` (Default): Uses the Coqui XTTS server. Reliable, high-quality, many voices.
- `TTS_SERVICE=chatterbox`: Uses the ChatterBox server via its streaming WebSocket endpoint. **Recommended for the lowest latency.**

If using `chatterbox`, the following variables also apply:
- `CHATTERBOX_URL`: The WebSocket endpoint (default: `ws://localhost:8004/stream`).
- `TTS_VOICE`: The name of the voice file in the chatterbox `voices/` directory (e.g., `Emily`, `Adrian`).

## Latency vs Quality Configuration

You can tune the trade-off between TTS speed and naturalness using `TTS_TEXT_AGGREGATOR`:

- `sentence` (Default): Standard sentence buffering. Waits for the *next* sentence's first character to confirm the current one ended. **Better Stability**, natural-sounding speech.
- `space_aware`: Streaming is optimized for speed. Starts speaking on the first space/newline after a sentence end. **Lower Latency**, but might chop sentences if the LLM pauses mid-thought.
- `none`: No aggregation. Streams tokens directly to TTS. **Experimental**, might cause "robotic" or jittery audio artifacts.

To change it, update your `.env` or `docker-compose.yml`:
```yaml
TTS_TEXT_AGGREGATOR=sentence
```

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
2. Check your selected TTS service:
   - For XTTS: `docker logs pipecat-xtts-1`
   - For ChatterBox: `docker logs pipecat-chatterbox-1`
3. Check Pipecat logs for WebRTC connection status and TTS initialization:
   ```bash
   docker logs -f pipecat-pipecat-1
   ```

### Network Issues
WebRTC requires `network_mode: host`. If running on Mac/Windows, ensure Docker Desktop is configured to allow host networking or use specific port mappings (though `host` is preferred for WebRTC).

