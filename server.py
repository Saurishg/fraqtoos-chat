#!/usr/bin/env python3
"""
FraqtoOS Chat — Tailscale chatbot.
Access: http://192.168.2.108:8080
Supports: Ollama models + Claude API + FLUX.1-schnell image generation
"""
import json, os, requests, base64, uuid, time, sys, io
from collections import defaultdict, deque
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

sys.path.insert(0, "/home/work/fraqtoos")
try:
    from core.web_search import search as _web_search, is_up as _searx_up
except Exception:
    _web_search = lambda *a, **k: []
    _searx_up   = lambda: False

load_dotenv("/home/work/fraqtoos-chat/.env")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA        = "http://localhost:11434"
COMFYUI       = "http://localhost:8188"
STATIC        = "/home/work/fraqtoos-chat/static"
CONV_DIR      = "/home/work/fraqtoos-chat/conversations"
MEMORY_FILE   = "/home/work/fraqtoos-chat/memory.json"
CLAUDE_MODELS = {"claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5-20251001"}

os.makedirs(CONV_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_RATE_BUCKETS: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
_RATE_LIMITS = {"chat": (20, 60), "imagine": (5, 60), "search": (30, 60),
                "upload": (10, 60), "conv": (60, 60)}


def _rate_ok(ip: str, bucket: str) -> bool:
    limit, window = _RATE_LIMITS[bucket]
    now = time.time()
    dq = _RATE_BUCKETS[bucket][ip]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


@app.get("/")
async def index():
    return FileResponse(f"{STATIC}/index.html")


VISION_MODELS = {"llava:7b", "llava:13b", "llava", "llama3.2-vision",
                 "qwen2-vl", "bakllava", "moondream"}


def _has_vision_model() -> str:
    """Return first available vision model name, or empty string."""
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        installed = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        installed.update(m["name"] for m in r.json().get("models", []))
        for v in ["llava:7b", "llava:13b", "llava", "llama3.2-vision",
                  "qwen2-vl", "bakllava", "moondream"]:
            if v in installed or v.split(":")[0] in installed:
                return v
    except Exception:
        pass
    return ""


def _trim_history(messages: list, system: str, keep_first: int = 2, keep_last: int = 10) -> tuple[list, str]:
    """Keep first N + last N messages. Summarize the middle with phi4 if dropped."""
    if len(messages) <= keep_first + keep_last + 2:
        return messages, system
    head = messages[:keep_first]
    tail = messages[-keep_last:]
    middle = messages[keep_first:-keep_last]
    if not middle:
        return messages, system
    middle_text = "\n".join(
        f"{m['role'].upper()}: {(m.get('content') or '')[:600]}"
        for m in middle
    )[:6000]
    summary = ""
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": "phi4", "stream": False,
            "prompt": ("Summarize this conversation segment in 80 words. "
                       "Preserve names, decisions, numbers, file paths. No preamble.\n\n" + middle_text),
            "options": {"temperature": 0.2, "num_predict": 200}
        }, timeout=20)
        summary = (r.json().get("response", "") or "").strip()
    except Exception:
        summary = f"({len(middle)} earlier messages omitted)"
    aug_system = (system + "\n\n" if system else "") + \
                 f"[Earlier conversation summary: {summary}]"
    return head + tail, aug_system


@app.post("/chat")
async def chat(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "chat"):
        return JSONResponse({"error": "rate limit: 20 req/min"}, 429)
    data     = await req.json()
    model    = data.get("model", "phi4")
    messages = data.get("messages", [])
    system   = data.get("system", "")
    images   = data.get("images") or []

    messages, system = _trim_history(messages, system)

    # Always prepend persistent user memory to system context
    mem_block = _memory_as_system_block()
    if mem_block:
        system = mem_block + ("\n\n" + system if system else "")

    if images:
        vision = _has_vision_model()
        if not vision:
            return StreamingResponse(
                iter([json.dumps({"error": "No vision model installed. Run: ollama pull llava:7b"})+"\n"]),
                media_type="text/plain")
        return StreamingResponse(
            ollama_stream(vision, messages, system, images=images),
            media_type="text/plain")
    if model in CLAUDE_MODELS:
        return StreamingResponse(claude_stream(model, messages, system), media_type="text/plain")
    return StreamingResponse(ollama_stream(model, messages, system), media_type="text/plain")


# ─── Smart auto-routing ──────────────────────────────────────────────
ROUTING_TARGETS = {
    "code":      "deepseek-r1:14b",
    "reasoning": "deepseek-r1:14b",
    "finance":   "qwen3:14b",
    "copy":      "gemma4:latest",
    "long":      "llama4:latest",
    "general":   "phi4",
    "quick":     "phi4",
}

CLASSIFY_PROMPT = """Classify the user's request into ONE category. Reply with only the category word.

Categories:
- code: programming, debugging, code review, regex, scripts
- reasoning: math, logic, multi-step problem solving, hard analysis
- finance: stocks, crypto, trading, accounting, BSR/Amazon analytics
- copy: marketing copy, listings, descriptions, emails, polish
- long: needs >500 word output (reports, full essays, deep research)
- general: general Q&A, simple chat, summaries, quick lookups

Request: {q}

Category:"""


@app.post("/classify")
async def classify(req: Request):
    """Pick the best local model for a prompt using phi4."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data = await req.json()
    q = (data.get("text") or "").strip()
    if not q:
        return {"category": "general", "model": ROUTING_TARGETS["general"]}
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": "phi4", "stream": False,
            "prompt": CLASSIFY_PROMPT.format(q=q[:1500]),
            "options": {"temperature": 0.0, "num_predict": 8}
        }, timeout=15)
        raw = (r.json().get("response", "") or "").strip().lower()
        cat = "general"
        for k in ROUTING_TARGETS:
            if k in raw:
                cat = k
                break
        # Verify target model is installed; fall back to phi4
        try:
            tags = requests.get(f"{OLLAMA}/api/tags", timeout=3).json()
            installed = {m["name"] for m in tags.get("models", [])}
            target = ROUTING_TARGETS[cat]
            if target not in installed:
                target = "phi4:latest" if "phi4:latest" in installed else "phi4"
        except Exception:
            target = ROUTING_TARGETS[cat]
        return {"category": cat, "model": target, "raw": raw}
    except Exception as e:
        return JSONResponse({"category": "general", "model": "phi4",
                             "error": str(e)}, 200)


# ─── PWA ─────────────────────────────────────────────────────────────
@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name":             "FraqtoOS Chat",
        "short_name":       "Fraqtoos",
        "description":      "Local AI chat with vision, search, image gen, and bot bridge",
        "start_url":        "/",
        "display":          "standalone",
        "background_color": "#1a1917",
        "theme_color":      "#cc7722",
        "orientation":      "portrait",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"},
        ],
    })


@app.get("/service-worker.js")
async def service_worker():
    sw = """const CACHE = 'fraqtoos-v1';
const ASSETS = ['/', '/static/icon-192.png', '/static/icon-512.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Never cache API calls — always go to network
  if (url.pathname.startsWith('/chat') || url.pathname.startsWith('/imagine') ||
      url.pathname.startsWith('/search') || url.pathname.startsWith('/upload') ||
      url.pathname.startsWith('/conversations') || url.pathname.startsWith('/bridge') ||
      url.pathname.startsWith('/classify') || url.pathname.startsWith('/health')) {
    return; // default network fetch
  }
  if (e.request.method !== 'GET') return;
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request).then(resp => {
      if (resp.ok && url.origin === location.origin) {
        const clone = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return resp;
    }).catch(() => caches.match('/')))
  );
});
"""
    return Response(content=sw, media_type="application/javascript")


@app.post("/imagine")
async def imagine(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    data        = await req.json()
    prompt      = data.get("prompt", "")
    steps       = data.get("steps", None)
    width       = data.get("width", 1024)
    height      = data.get("height", 1024)
    image_model = data.get("image_model", "flux-schnell")
    negative    = data.get("negative", "")

    if not prompt:
        return JSONResponse({"error": "prompt required"}, 400)
    if not _comfyui_ready():
        return JSONResponse({"error": "Image generator not ready."}, 503)

    try:
        image_b64 = _generate(prompt, image_model, steps, width, height, negative)
        return JSONResponse({"image": image_b64, "prompt": prompt, "model": image_model})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.get("/imagine/models")
async def imagine_models():
    """Return which image models are available (file exists on disk)."""
    base = "/home/work/ComfyUI/models"
    available = []
    checks = {
        "flux-schnell": f"{base}/unet/flux1-schnell-Q8_0.gguf",
        "flux-dev":     f"{base}/unet/flux1-dev-Q4_0.gguf",
        "sdxl":         f"{base}/checkpoints/sd_xl_base_1.0.safetensors",
        "sd15":         f"{base}/checkpoints/v1-5-pruned-emaonly.safetensors",
        "juggernaut":   f"{base}/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    }
    for name, path in checks.items():
        if os.path.exists(path) and os.path.getsize(path) > 1024*1024:
            available.append(name)
    return {"models": available}


@app.get("/imagine/status")
async def imagine_status():
    ready = _comfyui_ready()
    return {"ready": ready, "url": COMFYUI}


@app.post("/suggest")
async def suggest(req: Request):
    """Given recent chat, return 3 short follow-up prompts the user might want to ask."""
    data = await req.json()
    msgs = data.get("messages", [])[-6:]
    if not msgs:
        return {"suggestions": []}
    convo = "\n".join(f"{m['role']}: {(m.get('content') or '')[:400]}" for m in msgs)
    prompt = (
        "Based on this chat, suggest 3 short follow-up questions the user might ask next. "
        "Each must be ≤8 words, phrased as a user would type, no numbering, one per line. "
        "Be specific to the topic.\n\n" + convo + "\n\n3 follow-ups:"
    )
    try:
        r = requests.post(
            f"{OLLAMA}/api/generate",
            json={"model": "phi4", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.5, "num_predict": 80}},
            timeout=15,
        )
        text = r.json().get("response", "").strip()
        lines = [l.strip("•-1234567890. ").strip() for l in text.split("\n") if l.strip()]
        lines = [l for l in lines if 3 <= len(l) <= 70][:3]
        return {"suggestions": lines}
    except Exception as e:
        return {"suggestions": [], "error": str(e)}


@app.post("/face-swap")
async def face_swap(req: Request, source: UploadFile = File(...), target: UploadFile = File(...)):
    """Swap face from `source` onto `target`. Returns base64 PNG."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    try:
        from face_swap import swap as _swap
        src_b = await source.read()
        tgt_b = await target.read()
        if max(len(src_b), len(tgt_b)) > 12 * 1024 * 1024:
            return JSONResponse({"error": "image too large (max 12 MB)"}, 413)
        out = _swap(src_b, tgt_b)
        return JSONResponse({"image": base64.b64encode(out).decode(), "model": "inswapper_128"})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


def _build_avatar_workflow(face_image_name: str, prompt: str, steps: int, width: int, height: int, weight: float = 1.0) -> dict:
    """PuLID-FLUX workflow: face image + prompt → image of that person in scene."""
    return {
        "1":  {"class_type": "UnetLoaderGGUF",  "inputs": {"unet_name": "flux1-dev-Q4_0.gguf"}},
        "2":  {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
        "3":  {"class_type": "VAELoader",       "inputs": {"vae_name": "ae.safetensors"}},
        "4":  {"class_type": "LoadImage",       "inputs": {"image": face_image_name}},
        "5":  {"class_type": "PulidFluxModelLoader",      "inputs": {"pulid_file": "pulid_flux_v0.9.1.safetensors"}},
        "6":  {"class_type": "PulidFluxInsightFaceLoader","inputs": {"provider": "CUDA"}},
        "7":  {"class_type": "PulidFluxEvaClipLoader",    "inputs": {}},
        "8":  {"class_type": "ApplyPulidFlux",  "inputs": {"model": ["1", 0], "pulid_flux": ["5", 0], "eva_clip": ["7", 0], "face_analysis": ["6", 0], "image": ["4", 0], "weight": weight, "start_at": 0.0, "end_at": 1.0}},
        "9":  {"class_type": "CLIPTextEncode",  "inputs": {"text": prompt, "clip": ["2", 0]}},
        "10": {"class_type": "CLIPTextEncode",  "inputs": {"text": "", "clip": ["2", 0]}},
        "11": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "12": {"class_type": "KSampler",        "inputs": {"model": ["8", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["11", 0], "seed": int(time.time()), "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "13": {"class_type": "VAEDecode",       "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage",       "inputs": {"images": ["13", 0], "filename_prefix": "avatar"}},
    }


def _avatar_image(face_bytes: bytes, face_filename: str, prompt: str, steps: int = 25, width: int = 1024, height: int = 1024, weight: float = 1.0) -> str:
    files = {"image": (face_filename, face_bytes, "application/octet-stream")}
    up = requests.post(f"{COMFYUI}/upload/image", files=files, data={"overwrite": "true"}, timeout=30)
    up.raise_for_status()
    uploaded_name = up.json().get("name") or face_filename

    wf = _build_avatar_workflow(uploaded_name, prompt, steps, width, height, weight)
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=10)
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", str(resp["error"])))
    prompt_id = resp["prompt_id"]
    for _ in range(360):
        time.sleep(1)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=5).json()
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            for node_out in hist[prompt_id]["outputs"].values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_r = requests.get(f"{COMFYUI}/view",
                        params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]},
                        timeout=15)
                    return base64.b64encode(img_r.content).decode()
    raise TimeoutError("Avatar generation timed out")


@app.post("/avatar")
async def avatar(req: Request, face: UploadFile = File(...), prompt: str = "", steps: int = 25, width: int = 1024, height: int = 1024, weight: float = 1.0):
    """Generate an image of `face` in a scene described by `prompt` via PuLID-FLUX."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not prompt:
        form = await req.form()
        prompt = (form.get("prompt") or "").strip()
        try:    steps = int(form.get("steps") or steps)
        except: pass
        try:    weight = float(form.get("weight") or weight)
        except: pass
    if not prompt:
        return JSONResponse({"error": "prompt required"}, 400)
    if not _comfyui_ready():
        return JSONResponse({"error": "Image generator not ready."}, 503)
    if not os.path.exists("/home/work/ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors"):
        return JSONResponse({"error": "PuLID-FLUX model missing."}, 503)
    try:
        face_b = await face.read()
        if len(face_b) > 12 * 1024 * 1024:
            return JSONResponse({"error": "image too large (max 12 MB)"}, 413)
        fname  = face.filename or f"face_{int(time.time())}.png"
        edited = _avatar_image(face_b, fname, prompt, max(8, min(int(steps), 40)),
                                int(width), int(height), float(weight))
        return JSONResponse({"image": edited, "prompt": prompt, "model": "pulid-flux"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/edit-image")
async def edit_image(req: Request, image: UploadFile = File(...), prompt: str = "", steps: int = 20):
    """Edit `image` per text `prompt` using FLUX.1 Kontext. Multipart form."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not prompt:
        form = await req.form()
        prompt = (form.get("prompt") or "").strip()
        try:    steps = int(form.get("steps") or steps)
        except: pass
    if not prompt:
        return JSONResponse({"error": "prompt required"}, 400)
    if not _comfyui_ready():
        return JSONResponse({"error": "Image generator not ready."}, 503)
    if not os.path.exists("/home/work/ComfyUI/models/unet/flux1-kontext-dev-Q4_0.gguf"):
        return JSONResponse({"error": "Kontext model not yet downloaded."}, 503)
    try:
        img_bytes = await image.read()
        if len(img_bytes) > 12 * 1024 * 1024:
            return JSONResponse({"error": "image too large (max 12 MB)"}, 413)
        fname = image.filename or f"upload_{int(time.time())}.png"
        edited = _edit_image(img_bytes, fname, prompt, max(4, min(steps, 40)))
        return JSONResponse({"image": edited, "prompt": prompt, "model": "flux-kontext"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/search")
async def search(req: Request):
    """Web search via local SearXNG. Returns top results as a text block."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "search"):
        return JSONResponse({"error": "rate limit: 30 req/min"}, 429)
    data = await req.json()
    query = (data.get("query") or "").strip()
    n     = max(1, min(int(data.get("n", 5)), 10))
    if not query:
        return JSONResponse({"error": "query required"}, 400)
    if not _searx_up():
        return JSONResponse({"error": "SearXNG is down"}, 503)
    hits = _web_search(query, n=n)
    return {"query": query, "results": hits, "count": len(hits)}


IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}


@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    """Accept .txt/.md/.pdf or image. Text→extracted text. Image→base64."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "upload"):
        return JSONResponse({"error": "rate limit: 10 req/min"}, 429)
    name = file.filename or "file"
    raw  = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        return JSONResponse({"error": "file too large (10MB max)"}, 413)
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext in IMAGE_EXTS:
        b64 = base64.b64encode(raw).decode()
        return {"name": name, "ext": ext, "kind": "image",
                "image_b64": b64, "bytes": len(raw)}
    text = ""
    try:
        if ext in ("txt", "md", "log", "json", "csv", "py", "js", "html", "xml", "yml", "yaml"):
            text = raw.decode("utf-8", errors="replace")
        elif ext == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            return JSONResponse({"error": f"unsupported file type: .{ext}"}, 400)
    except Exception as e:
        return JSONResponse({"error": f"extract failed: {e}"}, 500)
    truncated = len(text) > 50000
    if truncated:
        text = text[:50000]
    return {"name": name, "ext": ext, "kind": "text", "text": text,
            "chars": len(text), "truncated": truncated}


# ─── Bot bridge ───────────────────────────────────────────────────────
def _bridge_competitor() -> str:
    import glob
    snaps = sorted(glob.glob("/home/work/amazon-bot/logs/competitor_history/*.json"))
    if not snaps:
        return "No competitor snapshots yet."
    latest = json.load(open(snaps[-1]))
    rows = latest.get("rows", [])
    out = [f"## Competitor Watch — {latest.get('date','?')} ('{latest.get('keyword','?')}')",
           f"Top {len(rows)} organic ASINs:\n"]
    for i, r in enumerate(rows, 1):
        out.append(f"{i}. **{r.get('asin')}** — {(r.get('brand') or '?')[:30]} · "
                   f"₹{r.get('price') or '?'} · {r.get('rating') or '?'}★ · "
                   f"{int(r.get('reviews') or 0)} reviews · BSR #{r.get('bsr') or '?'}")
    return "\n".join(out)


def _bridge_watchdog() -> str:
    p = "/home/work/fraqtoos/logs/watchdog_latest.json"
    if not os.path.exists(p):
        return "No watchdog report."
    d = json.load(open(p))
    snap = d.get("snapshot", {})
    out = [f"## Watchdog — {snap.get('timestamp','?')}",
           f"**System**: disk={snap.get('system',{}).get('disk','?')[:60]}",
           f"**SearXNG**: {'✓ up' if snap.get('searxng_up') else '✗ DOWN'}",
           "\n**Bots:**"]
    for b in snap.get("bots", []):
        icon = "🟢" if b.get("running") else ("🔴" if b.get("critical") else "🟡")
        out.append(f"- {icon} {b.get('name')}" +
                   (f" — {b['errors'][-1][:80]}" if b.get('errors') else ""))
    out.append(f"\n**AI diagnosis:**\n{(d.get('analysis','') or '')[:1200]}")
    return "\n".join(out)


def _bridge_digest() -> str:
    p = "/home/work/fraqtoos/logs/ai_context.json"
    if not os.path.exists(p):
        return "No ai_context yet."
    d = json.load(open(p))
    if not d:
        return "ai_context is empty."
    today = sorted(d.keys())[-1]
    bots = d[today]
    out = [f"## Daily Digest — {today}\n"]
    for name, summary in bots.items():
        out.append(f"### {name}\n{summary}\n")
    return "\n".join(out)


def _bridge_bots() -> str:
    p = "/home/work/fraqtoos/logs/state.json"
    state = {}
    if os.path.exists(p):
        try: state = json.load(open(p))
        except Exception: pass
    out = ["## Bot State"]
    if not state:
        out.append("(no state file)")
    for k, v in state.items():
        out.append(f"- **{k}**: {v}")
    return "\n".join(out)


def _bridge_help() -> str:
    return ("## Bot bridge commands\n"
            "- `/competitor` — latest top-10 ASIN snapshot\n"
            "- `/watchdog` — bot health + AI diagnosis\n"
            "- `/digest` — today's per-bot summaries\n"
            "- `/bots` — orchestrator state\n"
            "- `/help` — this message")


_BRIDGE = {
    "competitor": _bridge_competitor,
    "watchdog":   _bridge_watchdog,
    "digest":     _bridge_digest,
    "bots":       _bridge_bots,
    "help":       _bridge_help,
}


@app.get("/bridge/{cmd}")
async def bridge(cmd: str, req: Request):
    """Read live FraqtoOS data for chat slash commands."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    cmd = cmd.lower().strip()
    fn = _BRIDGE.get(cmd)
    if not fn:
        return JSONResponse({"error": f"unknown command: /{cmd}",
                             "available": list(_BRIDGE.keys())}, 404)
    try:
        return {"cmd": cmd, "text": fn()}
    except Exception as e:
        return JSONResponse({"error": f"{cmd} failed: {e}"}, 500)


# ─── Conversation search ──────────────────────────────────────────────
@app.get("/conversations/search/q")
async def conv_search(req: Request, q: str = ""):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    q = (q or "").strip().lower()
    if not q:
        return {"matches": []}
    matches = []
    for fn in os.listdir(CONV_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fn)) as f:
                c = json.load(f)
        except Exception:
            continue
        title  = (c.get("title") or "").lower()
        joined = "\n".join(m.get("content", "") for m in c.get("history", [])).lower()
        if q in title or q in joined:
            idx = joined.find(q)
            snippet = ""
            if idx >= 0:
                start = max(0, idx - 40)
                snippet = joined[start:idx+len(q)+80].replace("\n", " ")
            matches.append({
                "id":      c.get("id"),
                "title":   c.get("title", "Untitled"),
                "updated": c.get("updated", 0),
                "snippet": snippet,
                "in_title": q in title,
            })
    matches.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return {"matches": matches[:30]}


# ─── Auto-title generation ────────────────────────────────────────────
@app.post("/conversations/{conv_id}/autotitle")
async def conv_autotitle(conv_id: str, req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        path = _conv_path(conv_id)
    except ValueError:
        return JSONResponse({"error": "invalid id"}, 400)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, 404)
    with open(path) as f:
        c = json.load(f)
    hist = c.get("history", [])
    if len(hist) < 2:
        return {"title": c.get("title", "Untitled"), "skipped": "not enough turns"}
    snippet = "\n".join(
        f"{m['role'].upper()}: {m.get('content','')[:400]}"
        for m in hist[:4]
    )
    prompt = ("Write a concise 3-6 word title for this conversation. "
              "Plain text only, no quotes, no punctuation at end.\n\n" + snippet)
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": "phi4", "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 30}
        }, timeout=30)
        title = (r.json().get("response", "") or "").strip().strip('"').strip("'")
        title = title.split("\n")[0][:80] or c.get("title", "Untitled")
    except Exception as e:
        return JSONResponse({"error": f"phi4 failed: {e}"}, 500)
    c["title"] = title
    c["updated"] = int(time.time())
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, indent=2)
    os.replace(tmp, path)
    return {"id": conv_id, "title": title}


def _conv_path(conv_id: str) -> str:
    safe = "".join(c for c in conv_id if c.isalnum() or c in "_-")[:64]
    if not safe:
        raise ValueError("invalid id")
    return os.path.join(CONV_DIR, f"{safe}.json")


@app.get("/conversations")
async def conv_list(req: Request):
    """List all saved conversations (metadata only — no full history)."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    out = []
    for fn in sorted(os.listdir(CONV_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fn)) as f:
                c = json.load(f)
            out.append({
                "id":        c.get("id"),
                "title":     c.get("title", "Untitled"),
                "model":     c.get("model", ""),
                "updated":   c.get("updated", 0),
                "msg_count": len(c.get("history", [])),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return {"conversations": out}


@app.get("/conversations/{conv_id}")
async def conv_get(conv_id: str, req: Request):
    """Load full conversation by id."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        path = _conv_path(conv_id)
    except ValueError:
        return JSONResponse({"error": "invalid id"}, 400)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, 404)
    with open(path) as f:
        return json.load(f)


@app.post("/conversations")
async def conv_save(req: Request):
    """Create or update a conversation. Body: {id?, title, history, model}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data = await req.json()
    history = data.get("history", [])
    if not isinstance(history, list):
        return JSONResponse({"error": "history must be list"}, 400)
    conv_id = data.get("id") or f"c_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    try:
        path = _conv_path(conv_id)
    except ValueError:
        return JSONResponse({"error": "invalid id"}, 400)
    now = int(time.time())
    record = {
        "id":      conv_id,
        "title":   (data.get("title") or "Untitled")[:200],
        "model":   data.get("model", ""),
        "history": history,
        "created": data.get("created", now),
        "updated": now,
    }
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
            record["created"] = old.get("created", now)
        except Exception:
            pass
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)
    return {"id": conv_id, "updated": now, "msg_count": len(history)}


# ─── User memory (cross-conversation) ────────────────────────────────
def _load_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        return json.load(open(MEMORY_FILE))
    except Exception:
        return []


def _save_memory(items: list):
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, MEMORY_FILE)


def _memory_as_system_block() -> str:
    items = _load_memory()
    if not items:
        return ""
    lines = ["# What you know about the user (persistent memory):"]
    for it in items:
        lines.append(f"- {it.get('fact','')}")
    return "\n".join(lines)


@app.get("/memory")
async def memory_list(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    return {"memory": _load_memory()}


@app.post("/memory")
async def memory_add(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data = await req.json()
    fact = (data.get("fact") or "").strip()
    if not fact:
        return JSONResponse({"error": "fact required"}, 400)
    if len(fact) > 500:
        fact = fact[:500]
    items = _load_memory()
    new_id = f"m_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
    items.append({"id": new_id, "fact": fact, "ts": int(time.time())})
    _save_memory(items)
    return {"id": new_id, "fact": fact, "count": len(items)}


@app.delete("/memory/{mem_id}")
async def memory_delete(mem_id: str, req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    items = _load_memory()
    new = [m for m in items if m.get("id") != mem_id]
    if len(new) == len(items):
        return JSONResponse({"error": "not found"}, 404)
    _save_memory(new)
    return {"ok": True, "count": len(new)}


@app.post("/memory/extract")
async def memory_extract(req: Request):
    """Run a user message through phi4 to pull out memorable facts."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data = await req.json()
    text = (data.get("text") or "").strip()
    if not text:
        return {"facts": []}
    prompt = (
        "Extract 0-3 enduring user facts from this message that would be useful to remember "
        "across future conversations. Examples of GOOD facts: name, role, business, ongoing "
        "projects, preferences, tools they use, recurring goals. NOT good: one-off requests, "
        "questions, code snippets, transient state.\n\n"
        "Reply ONLY with a JSON array of short fact strings (max 12 words each). "
        "Empty array if nothing memorable.\n\n"
        f"Message: {text[:1500]}\n\nFacts JSON:"
    )
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": "phi4", "stream": False, "prompt": prompt,
            "options": {"temperature": 0.1, "num_predict": 200}
        }, timeout=30)
        raw = (r.json().get("response", "") or "").strip()
        # Try to find a JSON array in the response
        import re
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return {"facts": [], "raw": raw[:200]}
        try:
            facts = json.loads(m.group(0))
            facts = [str(f).strip() for f in facts if isinstance(f, str) and f.strip()][:3]
            return {"facts": facts}
        except Exception:
            return {"facts": [], "raw": raw[:200]}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.delete("/conversations/{conv_id}")
async def conv_delete(conv_id: str, req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        path = _conv_path(conv_id)
    except ValueError:
        return JSONResponse({"error": "invalid id"}, 400)
    if os.path.exists(path):
        os.remove(path)
        return {"ok": True}
    return JSONResponse({"error": "not found"}, 404)


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
        "search_ready":  _searx_up(),
    }


def _comfyui_ready() -> bool:
    try:
        r = requests.get(f"{COMFYUI}/system_stats", timeout=2)
        return r.status_code == 200
    except:
        return False


def _build_flux_workflow(unet_file: str, prompt: str, steps: int, width: int, height: int) -> dict:
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_file}},
        "2": {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": int(time.time()), "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "fraqtoos"}},
    }


def _build_sdxl_workflow(ckpt_file: str, prompt: str, negative: str, steps: int, width: int, height: int) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_file}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "ugly, blurry, watermark, text, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": int(time.time()), "steps": steps, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "fraqtoos"}},
    }


def _build_sd15_workflow(ckpt_file: str, prompt: str, negative: str, steps: int, width: int, height: int) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_file}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "ugly, blurry, watermark, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": min(width, 768), "height": min(height, 768), "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": int(time.time()), "steps": steps, "cfg": 7.5, "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "fraqtoos"}},
    }


def _generate(prompt: str, model: str, steps, width: int, height: int, negative: str = "") -> str:
    """Route to correct workflow based on model name, return base64 PNG."""
    if model == "flux-schnell":
        wf = _build_flux_workflow("flux1-schnell-Q8_0.gguf", prompt, steps or 4, width, height)
    elif model == "flux-dev":
        wf = _build_flux_workflow("flux1-dev-Q4_0.gguf", prompt, steps or 20, width, height)
    elif model == "sdxl":
        wf = _build_sdxl_workflow("sd_xl_base_1.0.safetensors", prompt, negative, steps or 25, width, height)
    elif model == "juggernaut":
        wf = _build_sdxl_workflow("Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", prompt, negative, steps or 25, width, height)
    elif model == "sd15":
        wf = _build_sd15_workflow("v1-5-pruned-emaonly.safetensors", prompt, negative, steps or 20, width, height)
    else:
        raise ValueError(f"Unknown image model: {model}")

    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=10)
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", str(resp["error"])))
    prompt_id = resp["prompt_id"]

    for _ in range(180):
        time.sleep(1)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=5).json()
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            for node_out in hist[prompt_id]["outputs"].values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_r = requests.get(f"{COMFYUI}/view",
                        params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]},
                        timeout=10)
                    return base64.b64encode(img_r.content).decode()
    raise TimeoutError("Image generation timed out")


def _build_kontext_workflow(unet_file: str, image_name: str, prompt: str, steps: int) -> dict:
    """Flux Kontext: edit `image_name` according to `prompt`. Image must already be
    uploaded to ComfyUI's input dir via /upload/image."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_file}},
        "2": {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "9": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["9", 0], "negative": ["8", 0], "latent_image": ["6", 0], "seed": int(time.time()), "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "kontext"}},
    }


def _edit_image(image_bytes: bytes, image_filename: str, prompt: str, steps: int = 20) -> str:
    """Upload image to ComfyUI, run Kontext edit, return base64 PNG."""
    files = {"image": (image_filename, image_bytes, "application/octet-stream")}
    data  = {"overwrite": "true"}
    up = requests.post(f"{COMFYUI}/upload/image", files=files, data=data, timeout=30)
    up.raise_for_status()
    uploaded_name = up.json().get("name") or image_filename

    wf = _build_kontext_workflow("flux1-kontext-dev-Q4_0.gguf", uploaded_name, prompt, steps)
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=10)
    resp = r.json()
    if "error" in resp:
        raise RuntimeError(resp["error"].get("message", str(resp["error"])))
    prompt_id = resp["prompt_id"]

    for _ in range(240):
        time.sleep(1)
        hist = requests.get(f"{COMFYUI}/history/{prompt_id}", timeout=5).json()
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            for node_out in hist[prompt_id]["outputs"].values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_r = requests.get(f"{COMFYUI}/view",
                        params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]},
                        timeout=15)
                    return base64.b64encode(img_r.content).decode()
    raise TimeoutError("Image edit timed out")


def ollama_stream(model, messages, system="", images=None):
    chat_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
    if images and chat_msgs:
        for m in reversed(chat_msgs):
            if m["role"] == "user":
                m["images"] = images
                break
    if system:
        chat_msgs = [{"role": "system", "content": system}] + chat_msgs
    payload = {
        "model": model, "messages": chat_msgs, "stream": True,
        "options": {"temperature": 0.7, "num_predict": 2000}
    }
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
    print(f"FraqtoOS Chat → http://192.168.2.108:8080")
    print(f"Claude: {'✓ loaded' if ANTHROPIC_KEY else '✗ no key'}")
    print(f"Images: ComfyUI on {COMFYUI}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
