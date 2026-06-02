# LocateAnything API

A FastAPI microservice wrapping NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) spatial grounding model, built for programmatic use in automated pipelines and AI agents.

## Overview

LocateAnything-3B is a 3B-parameter vision-language model that returns bounding boxes for objects, UI elements, and text given a natural language prompt. This repo exposes it over HTTP with two endpoints: an OpenAI-compatible `/v1/chat/completions` and a telemetry-rich `/api/inference` for dataset generation pipelines.

**Features:**
- `/v1/chat/completions` — drop-in compatibility with OpenAI vision tool integrations
- `/api/inference` — returns tokens/sec, boxes/sec, and decoding mode fallback stats alongside results
- Dynamic image resizing (`short_size`) to avoid VRAM OOM on large images
- Exposes all three LocateAnything decoding modes: `hybrid`, `fast` (MTP), `slow` (AR)

---

## Requirements

- NVIDIA GPU (~12GB VRAM minimum)
- Docker + Docker Compose with NVIDIA Container Toolkit (Linux) or WSL2 GPU passthrough (Windows)
- Hugging Face account with a Read-access token

---

## Setup

```bash
git clone https://github.com/dliebner/LocateAnythingAPI.git
cd LocateAnythingAPI
```

**Replace:**
```markdown
Create a `.env` file:
```env
HF_TOKEN=hf_your_token_here
API_PORT=8000  # Defaults to 8000
```

Build and run:
```bash
docker-compose up --build
```

First boot downloads PyTorch dependencies and the ~6GB model weights from Hugging Face. Subsequent starts are fast.

---

## API

*(Note: The examples below use port `8000`. If you changed `API_PORT` in your `.env` file, update the URLs accordingly.)*

### `POST /v1/chat/completions`

OpenAI-compatible. Accepts standard parameters (`temperature`, `max_tokens`, `top_p`) plus custom fields: `task`, `model_mode`, `short_size`, and `top_k`.

```python
import requests, base64

with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "nvidia/LocateAnything-3B",
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "task": "detect",        # detect | ground_multi | ground_gui | ocr | point
    "model_mode": "hybrid",  # hybrid | fast | slow
    "short_size": 1024,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "gem, clover, ring, bat"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
}

response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
print(response.json()["choices"][0]["message"]["content"])
# <ref>bat</ref><box><100><200><150><250></box>...
```

### `POST /api/inference`

Returns the raw output string plus generation stats. Useful for labeling pipelines where you want to track throughput or detect AR fallbacks.

```python
import requests, base64

with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "image_b64": img_b64,
    "prompt": "gem, clover, ring, bat",
    "task": "detect",
    "mode": "hybrid",
    "short_size": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 4096
}

response = requests.post("http://localhost:8000/api/inference", json=payload)
data = response.json()

print(data["raw_text"])  # bounding box string
print(data["stats"])     # {"tps": "84.3", "bps": "13.7", "switch_to_ar": "1", ...}
```

---

## License

**This repo (Apache 2.0):** The API wrapper code is licensed under Apache 2.0.

**Model weights (NVIDIA Non-Commercial):** The weights are downloaded at runtime from Hugging Face and are governed by the [NVIDIA Model License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE).