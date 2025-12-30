# Running SOTA Pipecat Locally with Docker & GPU

This guide details the local Docker implementation for running a fully GPU-accelerated Pipecat conversational AI agent. The setup is designed for privacy, performance, and "State Of The Art" (SOTA) agentic capabilities, including **Long-term Memory** and **Tool Usage**.

## Architecture

The system uses `docker-compose` to orchestrate three separate services, all communicating over the host network for optimal WebRTC performance.

### 1. **Pipecat Service (`pipecat`)**
- **Role**: The core application logic and audio pipeline orchestrator.
- **Runtime**: Python 3.12.
- **Entry Point**: `bot.py` (The SOTA implementation).
- **Components**:
  - **Transport**: `SmallWebRTCTransport` (Default WebRTC implementation).
  - **VAD**: `SileroVADAnalyzer` (Aggressive interruption detection).
  - **Turn Analysis**: `LocalSmartTurnAnalyzerV3` (Context-aware turn detection).
  - **Memory**: `Mem0MemoryService` (Persistent user memory).
  - **Tools**: `MCPClient` (Model Context Protocol) or native function calling.
  - **STT**: `WhisperSTTService` (local `faster-whisper` inference).
  - **LLM Client**: `OpenAILLMService` (connects to local Ollama `gemma3:27b`).
  - **TTS Client**: `XTTSService` (connects to local XTTS service).
  - **Frontend Integration**: `RTVIProcessor` (Real-Time Voice Interface).
- **Networking**: `network_mode: host` (Crucial for WebRTC connectivity).

### 2. **LLM Service (`ollama`)**
- **Role**: Provides large language model inference.
- **Model**: `gemma3:27b`.
- **Port**: `11434`
- **Hardware**: NVIDIA GPU acceleration (RTX 3090/4090 recommended).

### 3. **TTS Service (`xtts`)**
- **Role**: High-quality text-to-speech synthesis (Coqui XTTS v2).
- **Port**: `8000`.

---

## Prerequisites

- **Docker Desktop** (or Engine) installed.
- **NVIDIA GPU** with drivers and Container Toolkit installed.
- **(Optional) Mem0 API Key**: For cloud-based memory (or configure local vector DB).

## Quick Start

1.  **Configure Environment**:
    Edit `docker-compose.yml` if you have a Mem0 API key:
    ```yaml
    environment:
      - MEM0_API_KEY=your_key_here
    ```

2.  **Build and Start**:
    Since SOTA features require specific dependencies, build the image:
    ```bash
    docker compose up -d --build
    ```

3.  **Download Models (First Run)**:
    *   **XTTS**: Downloads ~2GB model (check `docker compose logs -f xtts`).
    *   **Ollama**: You *must* pull the SOTA model manually if not already present:
        ```bash
        docker exec -it pipecat-ollama-1 ollama pull gemma3:27b
        ```

4.  **Connect**:
    Open your browser to:
    **[http://localhost:7860/client](http://localhost:7860/client)**

    Say "Hello" to start the conversation! Try interrupting the bot or asking it to remember your name.

---

## Configuration Details

### `bot.py` Pipeline

The SOTA pipeline integrates memory and flow control:

```python
pipeline = Pipeline([
    transport.input(),             # 1. Mic input
    rtvi,                          # 2. RTVI (Real-time visualization)
    stt,                           # 3. Whisper STT (Local)
    context_aggregator.user(),     # 4. Smart Turn Management
    memory,                        # 5. Mem0 (Long-term Memory)
    llm,                           # 6. Gemma 3 27B (via Ollama)
    tts,                           # 7. XTTS (Local)
    transport.output(),            # 8. Speaker output
    context_aggregator.assistant() # 9. Context tracking
])
```

### Operations

#### Using Memory (Mem0)
If `MEM0_API_KEY` is set, the bot will use Mem0 to store interactions.
*   **Test**: Say "My name is [Name]". Restart the bot. Ask "What is my name?".
*   **Local Mode**: To run Mem0 entirely locally (no API key), uncomment the `local_config` section in `sota_bot.py`.

#### Changing the Model
1.  **Pull**: `docker exec -it pipecat-ollama-1 ollama pull llama3`
2.  **Update**: Change `docker-compose.yml`:
    ```yaml
    - LLM_MODEL=llama3
    ```
3.  **Restart**: `docker compose restart pipecat`

#### Viewing Logs
```bash
docker compose logs -f pipecat
```
