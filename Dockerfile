FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CUDA 12.1 for cross-compatibility)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]