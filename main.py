import os
import io
import base64
import time
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer

app = FastAPI(title="LocateAnything Dual-API Microservice")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = None
processor = None
tokenizer = None

@app.on_event("startup")
def load_model():
    global model, processor, tokenizer
    model_id = "nvidia/LocateAnything-3B"
    token = os.environ.get("HF_TOKEN")
    
    print(f"Loading {model_id} into VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        _attn_implementation="sdpa",
        trust_remote_code=True,
        token=token
    ).to(device).eval()
    print("✅ Model loaded successfully!")

# --- HELPER FUNCTIONS ---
def format_prompt(task_type: str, category: str) -> str:
    task_map = {
        "ground_gui": "GUI", "ground_multi": "Grounding",
        "detect": "Detection", "ocr": "OCR", "point": "Pointing"
    }
    mapped_task = task_map.get(task_type, "Detection")
    
    cats = "</c>".join(c.strip() for c in category.split(",") if c.strip())
    if mapped_task == "Detection": return f"Locate all the instances that matches the following description: {cats}."
    elif mapped_task == "Grounding": return f"Locate all the instances that match the following description: {cats}."
    elif mapped_task == "OCR": return "Detect all the text in box format."
    elif mapped_task == "GUI": return f"Locate the region that matches the following description: {cats}."
    elif mapped_task == "Pointing": return f"Point to: {cats}."
    return f"Locate all the instances that matches the following description: {cats}."

def _parse_out_info_dict(out_info: str) -> dict:
    """Parses the raw telemetry string into a JSON dictionary."""
    stats = {}
    if not out_info: return stats
    cleaned = re.sub(r"^[Ss]tast?ic\s*[Ii]nfo\s*,?\s*", "", out_info.strip())
    for part in cleaned.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            stats[k.strip()] = v.strip()
    return stats

@torch.no_grad()
def generate_core(image: Image.Image, task: str, prompt: str, mode: str, short_size: int, temp: float):
    """Shared inference logic used by both API endpoints."""
    # 1. Resize Image (OOM Protection)
    w, h = image.size
    if min(w, h) > short_size:
        scale = short_size / min(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    # 2. Build Prompt
    final_prompt = format_prompt(task, prompt)
    hf_messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": final_prompt}]}]

    # 3. Process Inputs
    text = processor.py_apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(hf_messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

    # 4. Generate (verbose=True is required to get the rich telemetry tuple)
    result = model.generate(
        pixel_values=inputs["pixel_values"].to(dtype),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tokenizer,
        max_new_tokens=4096,
        generation_mode=mode,
        temperature=temp,
        do_sample=True if temp > 0 else False,
        verbose=True 
    )

    # 5. Unpack Tuple
    output_text, token_sequence, out_info = "", [], ""
    if isinstance(result, tuple) and len(result) >= 3:
        output_text, token_sequence, out_info = result
    else:
        output_text = result

    return output_text, token_sequence, _parse_out_info_dict(out_info)


# ==========================================
# ENDPOINT 1: OPENAI COMPATIBLE (For ML Clicker)
# ==========================================
class ChatCompletionRequest(BaseModel):
    model: str = "nvidia/LocateAnything-3B"
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.1
    task: str = "detect" 
    model_mode: str = "hybrid"
    short_size: int = 1024

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not model: raise HTTPException(status_code=500, detail="Model not loaded yet.")
    try:
        prompt_text = ""
        image = None
        for msg in req.messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "text": prompt_text = item["text"]
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            header, encoded = url.split(",", 1)
                            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        if not image: raise HTTPException(status_code=400, detail="No image provided.")

        output_text, _, _ = generate_core(image, req.task, prompt_text, req.model_mode, req.short_size, req.temperature)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENDPOINT 2: RICH INFERENCE (For Offline Scripts)
# ==========================================
class RichInferenceRequest(BaseModel):
    image_b64: str
    prompt: str
    task: str = "detect"
    mode: str = "hybrid"
    short_size: int = 1024
    temperature: float = 0.1

@app.post("/api/inference")
def rich_inference(req: RichInferenceRequest):
    if not model: raise HTTPException(status_code=500, detail="Model not loaded yet.")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        
        output_text, token_sequence, stats = generate_core(image, req.task, req.prompt, req.mode, req.short_size, req.temperature)

        return {
            "success": True,
            "raw_text": output_text,
            "stats": stats,
            "token_sequence": token_sequence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
