FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install python and system dependencies
RUN apt-get update && apt-get install -y software-properties-common wget curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-full \
    build-essential \
    git \
    portaudio19-dev \
    libgl1 \
    libglib2.0-0 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install pip for python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy the application code
COPY . .

# Install dependencies
# We install faster-whisper specifically as it's required for local gpu stt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -e ".[daily,openai,silero,webrtc,runner,whisper,local-smart-turn-v3,mem0,mcp,noisereduce,rnnoise]" && \
    python -m pip install --no-cache-dir faster-whisper && \
    python -m pip install --no-cache-dir -e "./whisker/pipecat"

# Set library path for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CMD ["python", "bot.py", "-t", "webrtc", "--host", "0.0.0.0", "--port", "7860"]
