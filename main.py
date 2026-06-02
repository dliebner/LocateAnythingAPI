import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import io
import base64
import time
import re
import torch
import threading
import traceback

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import tempfile
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer

if not torch.cuda.is_available():
    raise RuntimeError("🚨 CUDA is not available! The container cannot see your GPU. Please check your Docker GPU passthrough settings.")

device = "cuda"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = None
processor = None
tokenizer = None
inference_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, tokenizer
    model_id = "nvidia/LocateAnything-3B"
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is missing! LocateAnything is a gated model. Please provide a valid token in your .env file.")
        
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

    # --- MONKEYPATCH ---
    # The custom Qwen2 LM in LocateAnything throws NotImplementedError for flash_attention_2.
    # We load the model in "sdpa" mode to satisfy Qwen2, but monkeypatch the Vision Encoder's 
    # attention router to use FlashAttention-2 anyway
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if 'LocateAnything' in mod_name and 'modeling_vit' in mod_name:
            if hasattr(mod, 'VL_VISION_ATTENTION_FUNCTIONS') and hasattr(mod, 'multihead_attention'):
                mod.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = mod.multihead_attention
                print("🔧 Successfully monkeypatched Vision Encoder to use FlashAttention-2!")

    print("✅ Model loaded successfully!")
    yield

app = FastAPI(title="LocateAnything Dual-API Microservice", lifespan=lifespan)

# Enable CORS for web-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Used by Docker/K8s to check if the model is ready to receive traffic."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading into VRAM.")
    return {"status": "ready", "model": "nvidia/LocateAnything-3B"}

# --- HELPER FUNCTIONS ---
def format_prompt(task_type: str, category: str) -> str:
    task_map = {
        "ground_gui": "GUI", "ground_multi": "Grounding",
        "detect": "Detection", "ocr": "OCR", "point": "Pointing"
    }
    mapped_task = task_map.get(task_type, "Detection")
    
    cats = "</c>".join(c.strip() for c in category.split(",") if c.strip())
    
    if mapped_task in ["Detection", "Grounding"]: 
        return f"Locate all the instances that matches the following description: {cats}."
    elif mapped_task == "OCR": 
        return "Detect all the text in box format."
    elif mapped_task == "GUI": 
        return f"Locate the region that matches the following description: {cats}."
    elif mapped_task == "Pointing": 
        return f"Point to: {cats}."
        
    return f"Locate all the instances that matches the following description: {cats}."

def _parse_out_info_dict(out_info: str) -> dict:
    """Parses the raw telemetry string into a JSON dictionary."""
    stats = {}
    if not out_info or not isinstance(out_info, str): return stats
    cleaned = re.sub(r"^(?:[Ss]tatistics?|[Ss]tatic)\s*[Ii]nfo\s*:?,?\s*", "", out_info.strip())
    for part in cleaned.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            stats[k.strip()] = v.strip()
    return stats

def decode_base64_image(encoded_str: str) -> Image.Image:
    """Decodes a base64 string robustly, stripping data URI prefixes and adding padding."""
    if "," in encoded_str:
        encoded_str = encoded_str.split(",", 1)[1]
    encoded_str = re.sub(r"\s+", "", encoded_str)
    encoded_str += "=" * ((4 - len(encoded_str) % 4) % 4)
    try:
        image_bytes = base64.b64decode(encoded_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

def decode_base64_video(encoded_str: str) -> str:
    """Decodes base64 video and saves it to a temp file, returning the absolute file path."""
    if "," in encoded_str:
        encoded_str = encoded_str.split(",", 1)[1]
    encoded_str = re.sub(r"\s+", "", encoded_str)
    encoded_str += "=" * ((4 - len(encoded_str) % 4) % 4)
    try:
        video_bytes = base64.b64decode(encoded_str)
        fd, path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(video_bytes)
            return path
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            if os.path.exists(path):
                os.remove(path)
            raise
    except Exception as e:
        raise ValueError(f"Invalid base64 video data: {e}")

@torch.no_grad()
def generate_core(media_content: List[Dict[str, Any]], task: str, prompt: str, mode: str, short_size: int, temp: float, max_tokens: int = 4096, top_p: float = 0.9, top_k: int = 50):
    """Shared inference logic used by both API endpoints."""
    with inference_lock:
        # 1. Resize Images (Skip for video payloads)
        for item in media_content:
            if item["type"] == "image":
                image = item["image"]
                w, h = image.size
                target_short = None
                
                if short_size and short_size > 0:
                    target_short = int(short_size)
                elif min(w, h) > 1024:
                    target_short = None
                    
                if target_short is not None:
                    if w <= h:
                        new_w = target_short
                        scale_factor = new_w / w
                        new_h = int(h * scale_factor)
                    else:
                        new_h = target_short
                        scale_factor = new_h / h
                        new_w = int(w * scale_factor)
                    item["image"] = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 2. Build Prompt Sequence
        final_prompt = format_prompt(task, prompt)
        hf_messages = [{"role": "user", "content": media_content + [{"type": "text", "text": final_prompt}]}]

        # 3. Process Inputs 
        text = processor.py_apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(hf_messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(device)

        # Token Guardrail (Prevents 16k context explosion and NoneType telemetry crash)
        seq_len = inputs["input_ids"].shape[1]
        if seq_len > 15800:
            raise ValueError(f"Payload Too Large: Sequence requires {seq_len} tokens (Limit: 16384). Please reduce short_size, lower video length, or batch frames.")

        # Cast any pixel values to correct dtype safely
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        if "pixel_values_videos" in inputs:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)

        # 4. Construct Generation Kwargs (Safely unpacks all vision/video inputs)
        gen_kwargs = {
            **inputs,
            "tokenizer": tokenizer,
            "max_new_tokens": max_tokens,
            "generation_mode": mode,
            "repetition_penalty": 1.1,
            "use_cache": True,
            "verbose": True,
            "do_sample": True,
            "temperature": temp if temp > 0 else 0.1,
            "top_p": top_p,
            "top_k": top_k
        }

        # 5. Generate
        try:
            with torch.inference_mode():
                result = model.generate(**gen_kwargs)

            # 6. Unpack Tuple Safely
            output_text, token_sequence, out_info = "", [], ""
            if isinstance(result, (tuple, list)):
                output_text = result[0] if len(result) > 0 else ""
                token_sequence = result[1] if len(result) > 1 else []
                out_info = result[2] if len(result) > 2 else ""
                
                # patch for slow mode telemetry
                if mode == "slow" and token_sequence:
                    if isinstance(token_sequence, tuple):
                        token_sequence = list(token_sequence)
                    token_sequence[-1] = ("ar", token_sequence[-1][1])
            else:
                output_text = result

            # Ensure token_sequence is JSON serializable
            def make_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                return obj

            token_sequence = make_serializable(token_sequence)

            return output_text, token_sequence, _parse_out_info_dict(out_info)
        finally:
            pass # Let PyTorch manage the memory pool automatically for higher TPS


# ==========================================
# ENDPOINT 1: OPENAI COMPATIBLE (For ML Clicker)
# ==========================================
class ChatCompletionRequest(BaseModel):
    model: str = "nvidia/LocateAnything-3B"
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 4096
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    task: str = "detect" 
    model_mode: str = "hybrid"
    short_size: int = 1024

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not model: raise HTTPException(status_code=503, detail="Model is still loading or unavailable.")
    
    temp_files = []
    try:
        prompt_text = ""
        media_content = []
        
        for msg in req.messages:
            content = msg.get("content")
            if isinstance(content, str):
                prompt_text += content + "\n"
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text": 
                        prompt_text += item.get("text", "") + "\n"
                    
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("http"):
                            raise HTTPException(status_code=400, detail="Only base64 data URIs are currently supported.")
                        elif url:
                            try:
                                img = decode_base64_image(url)
                                media_content.append({"type": "image", "image": img})
                            except ValueError as e:
                                raise HTTPException(status_code=400, detail=str(e))
                                
                    elif item.get("type") == "video_url":
                        url = item.get("video_url", {}).get("url", "")
                        if url.startswith("http"):
                            raise HTTPException(status_code=400, detail="Only base64 data URIs are currently supported.")
                        elif url:
                            try:
                                vid_path = decode_base64_video(url)
                                temp_files.append(vid_path)
                                # Qwen2-VL resolves local video paths natively via file:// URI
                                media_content.append({"type": "video", "video": f"file://{vid_path}"})
                            except ValueError as e:
                                raise HTTPException(status_code=400, detail=str(e))
        
        prompt_text = prompt_text.strip()
        if not media_content: raise HTTPException(status_code=400, detail="No valid base64 media provided.")

        temp = req.temperature if req.temperature is not None else 0.1
        max_tokens = req.max_tokens if req.max_tokens is not None else 4096
        top_p = req.top_p if req.top_p is not None else 0.9
        top_k = req.top_k if req.top_k is not None else 50
        
        output_text, _, _ = generate_core(media_content, req.task, prompt_text, req.model_mode, req.short_size, temp, max_tokens, top_p, top_k)

        created_time = int(time.time())
        return {
            "id": f"chatcmpl-{created_time}",
            "object": "chat.completion",
            "created": created_time,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass


# ==========================================
# ENDPOINT 2: RICH INFERENCE (For Offline Scripts)
# ==========================================
class RichInferenceRequest(BaseModel):
    image_b64: Optional[Union[str, List[str]]] = None
    video_b64: Optional[str] = None
    prompt: str
    task: str = "detect"
    mode: str = "hybrid"
    short_size: int = 1024
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 50

@app.post("/api/inference")
async def rich_inference(req: RichInferenceRequest):
    if not model: raise HTTPException(status_code=503, detail="Model is still loading or unavailable.")
    
    temp_files = []
    try:
        media_content = []
        
        # 1. Handle Video
        if req.video_b64:
            try:
                vid_path = decode_base64_video(req.video_b64)
                temp_files.append(vid_path)
                media_content.append({"type": "video", "video": f"file://{vid_path}"})
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
                
        # 2. Handle Image(s) / Frame Sequence
        if req.image_b64:
            images_to_process = req.image_b64 if isinstance(req.image_b64, list) else [req.image_b64]
            for b64 in images_to_process:
                try:
                    img = decode_base64_image(b64)
                    media_content.append({"type": "image", "image": img})
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                    
        if not media_content:
            raise HTTPException(status_code=400, detail="No valid base64 image_b64 or video_b64 provided.")
            
        output_text, token_sequence, stats = generate_core(
            media_content, req.task, req.prompt, req.mode, req.short_size, 
            req.temperature, req.max_tokens, req.top_p, req.top_k
        )

        return {
            "success": True,
            "raw_text": output_text,
            "stats": stats,
            "token_sequence": token_sequence
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
