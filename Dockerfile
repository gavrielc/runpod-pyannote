# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies for audio processing
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.11 is the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Install pip dependencies in multiple steps for better caching
COPY builder/requirements.txt /requirements.txt

# Install PyTorch first (biggest dependency)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

CMD python3 -u /handler.py
