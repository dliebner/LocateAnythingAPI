# Use official PyTorch devel image (includes CUDA 13.0, nvcc, and PyTorch 2.8.0)
FROM pytorch/pytorch:2.8.0-cuda13.0-cudnn9-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# 1. Compile FlashAttention FIRST so the 10-minute build is cached permanently
# --no-build-isolation forces pip to use the pre-installed PyTorch to compile the CUDA kernels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ninja flash-attn>=2.0.0 --no-build-isolation

# 2. Copy requirements and install fast dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]