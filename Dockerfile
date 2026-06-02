# Use NVIDIA's official CUDA 13.0 development image
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Prevent apt-get from prompting for timezone info
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.10, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 to python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# 1. Install PyTorch FIRST so FlashAttention can compile against it
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu130

# 2. Compile FlashAttention so the 10-minute build is cached permanently
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ninja flash-attn>=2.0.0 --no-build-isolation

# 3. Copy requirements and install the remaining fast dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]