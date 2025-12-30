FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the application code
COPY . .

# Install dependencies (ensure pipecat-ai is installed with necessary extras)
# Using pip for simplicity in this specific Dockerfile as we might not have a full uv.lock setup in root matching this exactly
RUN pip install -e ".[daily,openai,silero,webrtc,runner,whisper,local-smart-turn-v3]" && pip install faster-whisper

CMD ["python", "bot.py"]
