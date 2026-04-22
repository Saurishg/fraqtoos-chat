#!/usr/bin/env python3
"""
FraqtoOS Chat — Tailscale chatbot.
Access: http://100.67.1.60:8080
Supports: Ollama models + Claude API + FLUX.1-schnell image generation
"""
import json, os, requests, base64, uuid, time
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

load_dotenv("/home/work/fraqtoos-chat/.env")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA        = "http://localhost:11434"
COMFYUI       = "http://localhost:8188"
STATIC        = "/home/work/fraqtoos-chat/static"
CLAUDE_MODELS = {"claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"}

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(f"{STATIC}/index.html")


@app.post("/chat")
async def chat(req: Request):
    data     = await req.json()
    model    = data.get("model", "phi4")
    messages = data.get("messages", [])[-12:]
    system   = data.get("system", "")
    if model in CLAUDE_MODELS:
        return StreamingResponse(claude_stream(model, messages, system), media_type="text/plain")
    return StreamingResponse(ollama_stream(model, messages, system), media_type="text/plain")


@app.post("/imagine")
async def imagine(req: Request):
    """Generate image via ComfyUI FLUX.1-schnell."""
    data   = await req.json()
    prompt = data.get("prompt", "")
    steps  = data.get("steps", 4)
    width  = data.get("width", 1024)
    height = data.get("height", 1024)

    if not prompt:
        return JSONResponse({"error": "prompt required"}, 400)

    comfy_up = _comfyui_ready()
    if not comfy_up:
        return JSONResponse({"error": "Image generator not ready. Start it first."}, 503)

    try:
        image_b64 = _flux_generate(prompt, steps, width, height)
        return JSONResponse({"image": image_b64, "prompt": prompt})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/imagine/status")
async def imagine_status():
    ready = _comfyui_ready()
    return {"ready": ready, "url": COMFYUI}


@app.get("/health")
async def health():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
    except:
        models = []
    return {
        "status":        "ok",
        "ollama_models": models,
        "claude_ready":  bool(ANTHROPIC_KEY),
        "image_ready":   _comfyui_ready(),
    }


def _comfyui_ready() -> bool:
    try:
        r = requests.get(f"{COMFYUI}/system_stats", timeout=2)
        return r.status_code == 200
    except:
        return False


def _flux_generate(prompt: str, steps: int = 4, width: int = 1024, height: int = 1024) -> str:
    """Run FLUX.1-schnell via ComfyUI API, return base64 image."""
    client_id = str(uuid.uuid4())
    workflow = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-schnell-Q8_0.gguf"}},
        "2": {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": int(time.time()), "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "fraqtoos"}},
    }

    # Queue prompt
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=10)
    prompt_id = r.json()["prompt_id"]

    # Wait for completion (max 120s)
    for _ in range(120):
        time.sleep(1)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=5).json()
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            outputs = hist[prompt_id]["outputs"]
            for node_id, node_out in outputs.items():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_r = requests.get(
                        f"{COMFYUI}/view",
                        params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]},
                        timeout=10
                    )
                    return base64.b64encode(img_r.content).decode()
    raise TimeoutError("Image generation timed out")


def ollama_stream(model, messages, system=""):
    # Use /api/chat for proper multi-turn support
    chat_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
    payload = {
        "model": model, "messages": chat_msgs, "stream": True,
        "options": {"temperature": 0.7, "num_predict": 2000}
    }
    if system:
        payload["system"] = system
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=300)
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                token = d.get("message", {}).get("content", "")
                if token:
                    yield json.dumps({"token": token}) + "\n"
                if d.get("done"):
                    break
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


def claude_stream(model, messages, system=""):
    if not ANTHROPIC_KEY:
        yield json.dumps({"error": "Add ANTHROPIC_API_KEY to /home/work/fraqtoos-chat/.env"}) + "\n"
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        kwargs = dict(model=model, max_tokens=4096, messages=messages)
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield json.dumps({"token": text}) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


if __name__ == "__main__":
    print(f"FraqtoOS Chat → http://100.67.1.60:8080")
    print(f"Claude: {'✓ loaded' if ANTHROPIC_KEY else '✗ no key'}")
    print(f"Images: ComfyUI on {COMFYUI}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
