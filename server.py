#!/usr/bin/env python3
"""
FraqtoOS Chat — Tailscale chatbot.
Access: http://192.168.2.108:8080
Supports: Ollama models + FLUX.1-schnell image generation
"""
import asyncio
import json
import os
import requests
import base64
import uuid
import time
import sys
import io
import subprocess
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
    def _web_search(*a, **k):
        return []
    def _searx_up():
        return False

load_dotenv("/home/work/fraqtoos-chat/.env")
OLLAMA        = "http://localhost:11434"
# Image generation runs on the ROCm instance (6800 XT, comfyui-rocm.service)
# so FLUX jobs never grab the 3080 Ti — that card is the Chia harvester's only
# CUDA decompression GPU and an OOM there means missed block rewards.
COMFYUI       = "http://localhost:8189"
# PuLID avatar workflow stays on the CUDA instance: ApplyPulidFlux and its
# deps are only installed in the 8188 venv. Rare, user-triggered use only.
COMFYUI_CUDA  = "http://localhost:8188"
STATIC        = "/home/work/fraqtoos-chat/static"
CONV_DIR      = "/home/work/fraqtoos-chat/conversations"
MEMORY_FILE   = "/home/work/fraqtoos-chat/memory.json"

# ── Odysseus Deep Research integration ──────────────────────────────
# fraqtoos-chat drives Odysseus's Deep Research API (login session cookie)
# and streams the report back into chat. See /deep-research below.
ODYSSEUS_URL   = os.getenv("ODYSSEUS_URL", "http://localhost:7000").rstrip("/")
ODYSSEUS_USER  = os.getenv("ODYSSEUS_USER", "admin")
ODYSSEUS_PASS  = os.getenv("ODYSSEUS_PASS", "")
# Default research model. keep_alive=0 on the host means big models cold-load
# and blow the research probe's short timeout — phi4 cold-loads in ~8s and is
# reliable. Override via ODYSSEUS_RESEARCH_MODEL.
ODYSSEUS_RESEARCH_MODEL = os.getenv("ODYSSEUS_RESEARCH_MODEL", "phi4:latest")

os.makedirs(CONV_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_RATE_BUCKETS: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
_RATE_LIMITS = {"chat": (20, 60), "imagine": (5, 60), "search": (30, 60),
                "upload": (10, 60), "conv": (60, 60)}

# Locks for shared file writes (prevents race condition data loss)
_memory_lock     = asyncio.Lock()
_embed_cache_lock = asyncio.Lock()
_conv_lock       = asyncio.Lock()


def _pgrep_safe(pattern: str, timeout: float = 3) -> "subprocess.CompletedProcess|None":
    """Run pgrep with a hard timeout and swallow FileNotFoundError/timeouts.
    Returns None if pgrep is missing or hangs — callers should treat as 'no match'."""
    try:
        return subprocess.run(["pgrep", "-af", pattern],
                              capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


async def _safe_json(req: Request) -> "tuple[dict|None, JSONResponse|None]":
    """Parse request body as JSON. Returns (data, None) on success or (None, error_response) on failure."""
    try:
        data = await req.json()
    except Exception:
        return None, JSONResponse({"error": "invalid JSON body"}, 400)
    if not isinstance(data, dict):
        return None, JSONResponse({"error": "JSON body must be an object"}, 400)
    return data, None


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


# ── Odysseus Deep Research client ───────────────────────────────────
_odysseus = {"session": None}  # cached requests.Session with auth cookie


def _odysseus_login():
    """Return an authenticated requests.Session against Odysseus, logging in
    (and caching) on first use. Raises RuntimeError on failure."""
    s = _odysseus["session"]
    if s is not None:
        return s
    if not ODYSSEUS_PASS:
        raise RuntimeError("ODYSSEUS_PASS not set in fraqtoos-chat/.env")
    s = requests.Session()
    r = s.post(f"{ODYSSEUS_URL}/api/auth/login",
               json={"username": ODYSSEUS_USER, "password": ODYSSEUS_PASS, "remember": True},
               timeout=15)
    if r.status_code != 200 or not r.json().get("ok"):
        raise RuntimeError(f"Odysseus login failed (HTTP {r.status_code})")
    _odysseus["session"] = s
    return s


def _odysseus_request(method, path, **kw):
    """Authenticated Odysseus call that re-logs in once on 401 (expired cookie)."""
    s = _odysseus_login()
    url = f"{ODYSSEUS_URL}{path}"
    r = s.request(method, url, timeout=kw.pop("timeout", 30), **kw)
    if r.status_code == 401:
        _odysseus["session"] = None
        s = _odysseus_login()
        r = s.request(method, url, timeout=30, **kw)
    return r


def _odysseus_pick_endpoint(model):
    """Find an enabled Odysseus model endpoint that serves `model` (falls back
    to the first enabled endpoint). Returns (endpoint_id, model) or (None, model)."""
    r = _odysseus_request("GET", "/api/model-endpoints")
    if r.status_code != 200:
        return None, model
    eps = r.json() or []
    enabled = [e for e in eps if e.get("is_enabled", True)]
    for e in enabled:
        if model in (e.get("models") or []):
            return e.get("id"), model
    if enabled:  # model not found anywhere — use first endpoint's first model
        e = enabled[0]
        ms = e.get("models") or []
        return e.get("id"), (model if model in ms else (ms[0] if ms else model))
    return None, model


def _deep_research_stream(query, model, max_rounds, max_time):
    """Generator yielding NDJSON lines: {progress}, then {report,sources}, or {error}."""
    def emit(obj):
        return json.dumps(obj) + "\n"
    try:
        ep_id, model = _odysseus_pick_endpoint(model)
        body = {"query": query, "max_rounds": max_rounds, "max_time": max_time, "model": model}
        if ep_id:
            body["endpoint_id"] = ep_id
        r = _odysseus_request("POST", "/api/research/start", json=body)
        if r.status_code != 200:
            detail = ""
            try: detail = r.json().get("detail", "")
            except Exception: detail = r.text[:160]
            yield emit({"error": f"Could not start research: {detail or ('HTTP '+str(r.status_code))}"})
            return
        sid = r.json().get("session_id")
        if not sid:
            yield emit({"error": "Odysseus did not return a research session id"})
            return
        yield emit({"progress": "🧭 Planning research…"})

        # Poll status until terminal (research can run several minutes).
        last = None
        deadline = time.time() + max_time + 120
        while time.time() < deadline:
            sr = _odysseus_request("GET", f"/api/research/status/{sid}")
            if sr.status_code != 200:
                time.sleep(2); continue
            st = sr.json() or {}
            status = st.get("status", "")
            prog = st.get("progress") or {}
            phase = prog.get("phase", "")
            rnd = prog.get("round", "")
            label = {
                "planning": "🧭 Planning research…",
                "searching": f"🔎 Searching the web (round {rnd})…",
                "reading":   "📖 Reading sources…",
                "analyzing": f"🧠 Analyzing findings (round {rnd})…",
                "writing":   "✍️ Writing the report…",
            }.get(phase, f"⏳ {phase or 'working'}…")
            if label != last:
                last = label
                yield emit({"progress": label})
            if status and status.lower() in ("done", "complete", "completed", "error"):
                if status.lower() == "error":
                    yield emit({"error": "Research failed on the Odysseus side. Try again or a different model."})
                    return
                break
            time.sleep(3)
        else:
            yield emit({"error": "Research timed out."})
            return

        # Fetch the final report.
        pr = _odysseus_request("POST", f"/api/research/result-peek/{sid}")
        if pr.status_code != 200:
            yield emit({"error": "Research finished but the report could not be retrieved."})
            return
        d = pr.json() or {}
        report = (d.get("result") or "").strip()
        sources = d.get("sources") or []
        if not report:
            yield emit({"error": "Research produced an empty report."})
            return
        yield emit({"report": report, "sources": len(sources),
                    "report_url": f"{ODYSSEUS_URL}/api/research/report/{sid}"})
    except Exception as e:
        yield emit({"error": f"Deep research error: {e}"})


@app.post("/deep-research")
async def deep_research(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "search"):
        return JSONResponse({"error": "rate limit"}, 429)
    data, err = await _safe_json(req)
    if err:
        return err
    query = (data.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, 400)
    model = data.get("model") or ODYSSEUS_RESEARCH_MODEL
    try:
        max_rounds = max(0, min(20, int(data.get("max_rounds", 3))))
    except (TypeError, ValueError):
        max_rounds = 3
    try:
        max_time = max(60, min(1800, int(data.get("max_time", 600))))
    except (TypeError, ValueError):
        max_time = 600
    return StreamingResponse(
        _deep_research_stream(query, model, max_rounds, max_time),
        media_type="text/plain",
    )


# ── Odysseus memory + documents (shared via login session) ──────────
def _odysseus_memory_add(text, category="fact"):
    """Mirror a fact into Odysseus's vector memory. Best-effort; returns bool."""
    try:
        r = _odysseus_request("POST", "/api/memory/add",
                              json={"text": text[:500], "category": category, "source": "fraqtoos-chat"})
        return r.status_code == 200
    except Exception:
        return False


def _odysseus_memory_search(query):
    """Semantic search Odysseus memory. Returns list of {text/category} dicts."""
    r = _odysseus_request("POST", "/api/memory/search", data={"query": query})
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("memories", []) or []


def _odysseus_doc_create(title, content, language="markdown"):
    """Create an Odysseus library document. Returns (doc_id, error)."""
    r = _odysseus_request("POST", "/api/document",
                          json={"title": title[:200], "content": content, "language": language})
    if r.status_code != 200:
        detail = ""
        try: detail = r.json().get("detail", "")
        except Exception: detail = r.text[:160]
        return None, (detail or f"HTTP {r.status_code}")
    return (r.json() or {}).get("id"), None


@app.post("/odysseus-memory/search")
async def odysseus_memory_search(req: Request):
    data, err = await _safe_json(req)
    if err:
        return err
    query = (data.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, 400)
    loop = asyncio.get_running_loop()
    try:
        mems = await loop.run_in_executor(None, _odysseus_memory_search, query)
    except Exception as e:
        return JSONResponse({"error": f"Odysseus memory error: {e}"}, 502)
    out = []
    for m in mems[:20]:
        out.append({"text": m.get("text") or m.get("content") or "",
                    "category": (m.get("categories") or [m.get("category", "")])[0] if isinstance(m.get("categories"), list) else m.get("category", "")})
    return {"memories": out, "total": len(out)}


@app.post("/save-document")
async def save_document(req: Request):
    data, err = await _safe_json(req)
    if err:
        return err
    content = (data.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, 400)
    title = (data.get("title") or "").strip() or ("Note " + time.strftime("%Y-%m-%d %H:%M"))
    language = data.get("language") or "markdown"
    loop = asyncio.get_running_loop()
    try:
        doc_id, derr = await loop.run_in_executor(None, _odysseus_doc_create, title, content, language)
    except Exception as e:
        return JSONResponse({"error": f"Odysseus document error: {e}"}, 502)
    if not doc_id:
        return JSONResponse({"error": derr or "could not create document"}, 502)
    return {"id": doc_id, "title": title, "url": f"{ODYSSEUS_URL}/?doc={doc_id}"}


@app.get("/")
async def index():
    return FileResponse(f"{STATIC}/index.html")


# Preferred order when several vision models are installed.
_VISION_PREFERENCE = ("llava", "llama3.2-vision", "qwen2.5-vl", "qwen2-vl",
                      "gemma4", "minicpm-v", "bakllava", "moondream")
# capabilities per model name, so image uploads don't re-hit /api/show
_caps_cache: dict[str, list] = {}


def _model_caps(name: str) -> list:
    """Ollama capabilities for a model (cached): completion/tools/vision/thinking."""
    if name not in _caps_cache:
        try:
            r = requests.post(f"{OLLAMA}/api/show", json={"model": name}, timeout=5)
            _caps_cache[name] = r.json().get("capabilities", []) if r.ok else []
        except Exception:
            return []  # transient — don't cache the failure
    return _caps_cache[name]


def _has_vision_model() -> str:
    """Return the name of an installed vision-capable model, or empty string.
    Uses Ollama's real capability flags instead of name guessing, so models
    like gemma4 (vision-capable, no 'vl' in the name) are picked up."""
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        installed = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return ""
    vision = [n for n in installed if "vision" in _model_caps(n)]
    if not vision:
        return ""
    for pref in _VISION_PREFERENCE:
        for n in vision:
            if pref in n.lower():
                return n
    return vision[0]


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
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, 400)
    model    = data.get("model", "phi4")
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return JSONResponse({"error": "messages must be a list"}, 400)
    system   = data.get("system", "")
    images   = data.get("images") or []
    if not isinstance(images, list):
        images = []
    try:
        temperature = max(0.0, min(2.0, float(data.get("temperature", 0.7))))
    except (TypeError, ValueError):
        temperature = 0.7

    loop = asyncio.get_running_loop()
    messages, system = await loop.run_in_executor(None, _trim_history, messages, system)

    # Always prepend persistent user memory to system context
    mem_block = _memory_as_system_block()
    if mem_block:
        system = mem_block + ("\n\n" + system if system else "")

    if images:
        vision = _has_vision_model()
        if not vision:
            return JSONResponse(
                {"error": "No vision model installed. Run: ollama pull llava:7b"}, 503)
        return StreamingResponse(
            ollama_stream(vision, messages, system, images=images, temperature=temperature),
            media_type="text/plain")
    return StreamingResponse(ollama_stream(model, messages, system, images=images, temperature=temperature), media_type="text/plain")


# ─── Smart auto-routing ──────────────────────────────────────────────
ROUTING_TARGETS = {
    "code":      "deepseek-r1:14b",  # reasoning model, thinking ON (streamed to UI)
    "reasoning": "deepseek-r1:14b",  # reasoning model, thinking ON (streamed to UI)
    "finance":   "qwen3:30b-a3b",    # MoE depth, thinking ON (worth the wait here)
    "copy":      "gemma4:latest",    # back on gemma4 since the Ollama 0.30.7 fix
    "long":      "qwen3:30b-a3b",    # replaces llama4 (removed)
    "general":   "gemma4:latest",    # fast + clean (qwen3 thinks ~90s even for hello)
    "quick":     "phi4:latest",      # phi4 still fastest for 1-liners
}

CLASSIFY_PROMPT = """Classify the user's request into ONE category. Reply with only the category word.

Categories:
- code: programming, debugging, code review, regex, scripts
- reasoning: math, logic, multi-step problem solving, hard analysis
- finance: stocks, crypto, trading, accounting, market analytics
- copy: marketing copy, descriptions, emails, polish
- long: needs >500 word output (reports, full essays, deep research)
- quick: one-liners — greetings, yes/no, single facts, conversions
- general: general Q&A, simple chat, summaries, quick lookups

Request: {q}

Category:"""


@app.post("/classify")
async def classify(req: Request):
    """Pick the best local model for a prompt using phi4."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data, err = await _safe_json(req)
    if err:
        return err
    q = (data.get("text") or "").strip()
    if not q:
        return {"category": "general", "model": ROUTING_TARGETS["general"]}
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": "phi4", "stream": False,
            "prompt": CLASSIFY_PROMPT.format(q=q[:1500]),
            "options": {"temperature": 0.0, "num_predict": 8}
        }, timeout=45)
        raw = (r.json().get("response", "") or "").strip().lower()
        cat = "general"
        for k in ROUTING_TARGETS:
            if k in raw:
                cat = k
                break
        # Verify target model is installed; fall back to qwen3 then phi4
        try:
            tags = requests.get(f"{OLLAMA}/api/tags", timeout=3).json()
            installed = {m["name"] for m in tags.get("models", [])}
            target = ROUTING_TARGETS[cat]
            if target not in installed:
                for fb in ("qwen3:30b-a3b", "gemma4:latest", "phi4:latest"):
                    if fb in installed:
                        target = fb; break
        except Exception:
            target = ROUTING_TARGETS[cat]
        return {"category": cat, "model": target, "raw": raw}
    except Exception as e:
        return JSONResponse({"category": "general", "model": "qwen3:30b-a3b",
                             "error": str(e)}, 500)


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
    sw = """const CACHE = 'fraqtoos-v35';
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
  const API_PREFIXES = ['/chat','/imagine','/search','/upload','/conversations',
    '/bridge','/classify','/health','/models','/gpu','/memory','/suggest','/exec',
    '/status','/logs/','/ask-vault','/feedback',
    '/edit-image','/face-swap','/avatar','/mimic-motion','/animate-anyone','/champ','/champ-status',
    '/wan-video','/wan-i2v','/wan-animate','/vace','/comfy-interrupt','/manifest.json'];
  if (API_PREFIXES.some(p => url.pathname.startsWith(p))) return;
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


async def _stream_image_job(fn):
    """Run blocking fn() in a thread executor, yield NDJSON progress pings every 5s,
    then yield the final result dict or an error dict as the last line."""
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, fn)
    tick = 0
    while not fut.done():
        tick += 1
        yield json.dumps({"progress": tick * 5}) + "\n"
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=5.0)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break
        else:
            break
    try:
        yield json.dumps(fut.result()) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"
    finally:
        # Always retrieve the future's outcome (e.g. on client disconnect) so asyncio
        # doesn't log "Future exception was never retrieved".
        if fut.done() and not fut.cancelled():
            try: fut.exception()
            except Exception: pass


@app.post("/imagine")
async def imagine(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    data, err = await _safe_json(req)
    if err: return err
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

    def fn():
        return {"image": _generate(prompt, image_model, steps, width, height, negative),
                      "prompt": prompt, "model": image_model}
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


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
        "juggernaut-xi": f"{base}/checkpoints/Juggernaut-XI-v11.safetensors",
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
    data, err = await _safe_json(req)
    if err: return err
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
    up = requests.post(f"{COMFYUI_CUDA}/upload/image", files=files, data={"overwrite": "true"}, timeout=30)
    up.raise_for_status()
    uploaded_name = up.json().get("name") or face_filename

    wf = _build_avatar_workflow(uploaded_name, prompt, steps, width, height, weight)
    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI_CUDA}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=10)
    resp = r.json()
    if "error" in resp:
        err = resp["error"]; raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    prompt_id = resp["prompt_id"]
    for _ in range(360):
        time.sleep(1)
        hist = requests.get(f"{COMFYUI_CUDA}/history/{prompt_id}", timeout=5).json()
        if prompt_id in hist and hist[prompt_id].get("outputs"):
            for node_out in hist[prompt_id]["outputs"].values():
                if "images" in node_out:
                    img = node_out["images"][0]
                    img_r = requests.get(f"{COMFYUI_CUDA}/view",
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
        except (ValueError, TypeError): pass
        try:    weight = float(form.get("weight") or weight)
        except (ValueError, TypeError): pass
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
        def fn():
            return {"image": _avatar_image(face_b, fname, prompt, max(8, min(int(steps), 40)),
                                                      int(width), int(height), float(weight)),
                              "prompt": prompt, "model": "pulid-flux"}
        return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


MIMIC_VENV_PYTHON = "/home/work/MimicMotion/venv/bin/python"
MIMIC_SCRIPT      = "/home/work/MimicMotion/run_api.py"
AA_VENV_PYTHON    = "/home/work/AnimateAnyone/venv/bin/python"
AA_SCRIPT         = "/home/work/AnimateAnyone/run_api.py"
CHAMP_VENV_PYTHON = "/home/work/champ/venv/bin/python"
CHAMP_SCRIPT      = "/home/work/champ/run_api.py"
WAN_VENV_PYTHON   = "/home/work/Wan2.1/venv/bin/python"
WAN_SCRIPT        = "/home/work/Wan2.1/run_wan.py"
WAN_I2V_SCRIPT    = "/home/work/Wan2.1/run_wan_i2v.py"
# Wan2.2-Animate: ROCm venv + runner (preprocess + ComfyUI GGUF on 6800 XT / port 8189)
WANANIM_VENV_PYTHON = "/home/work/ComfyUI/venv-rocm/bin/python"
WANANIM_SCRIPT      = "/home/work/fraqtoos-chat/scripts/wan_animate_run.py"
# Wan2.1-VACE-14B: motion/structure control (control video + optional reference image)
VACE_SCRIPT         = "/home/work/fraqtoos-chat/scripts/vace_run.py"


def _run_mimic_motion(avatar_bytes: bytes, avatar_name: str,
                       driving_bytes: bytes, driving_name: str,
                       num_frames: int = 16, resolution: int = 576,
                       fps: int = 15, steps: int = 25,
                       guidance: float = 2.0, stride: int = 4) -> str:
    """Run MimicMotion in its own venv; return base64-encoded mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="mimic_")
    try:
        avatar_path  = os.path.join(tmp, avatar_name)
        driving_path = os.path.join(tmp, driving_name)
        output_path  = os.path.join(tmp, "output.mp4")
        with open(avatar_path,  "wb") as f: f.write(avatar_bytes)
        with open(driving_path, "wb") as f: f.write(driving_bytes)
        cmd = [
            MIMIC_VENV_PYTHON, MIMIC_SCRIPT,
            "--image",      avatar_path,
            "--video",      driving_path,
            "--output",     output_path,
            "--num_frames", str(num_frames),
            "--resolution", str(resolution),
            "--fps",        str(fps),
            "--steps",      str(steps),
            "--guidance",   str(guidance),
            "--stride",     str(stride),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "MimicMotion failed")[-1000:])
        if not os.path.exists(output_path):
            raise RuntimeError("MimicMotion produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/mimic-motion")
async def mimic_motion_endpoint(req: Request, avatar: UploadFile = File(...), driving: UploadFile = File(...)):
    """Animate `avatar` image with `driving` video via MimicMotion. Returns NDJSON → {video: base64_mp4}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(MIMIC_VENV_PYTHON):
        return JSONResponse({"error": "MimicMotion venv not found — check /home/work/MimicMotion/venv/"}, 503)

    form = await req.form()
    try:    num_frames = max(8, min(int(form.get("num_frames") or 16), 72))
    except (ValueError, TypeError): num_frames = 16
    try:    resolution = int(form.get("resolution") or 576)
    except (ValueError, TypeError): resolution = 576
    try:    fps = max(8, min(int(form.get("fps") or 15), 30))
    except (ValueError, TypeError): fps = 15
    try:    steps = max(8, min(int(form.get("steps") or 25), 50))
    except (ValueError, TypeError): steps = 25
    try:    guidance = float(form.get("guidance") or 2.0)
    except (ValueError, TypeError): guidance = 2.0
    try:    stride = max(1, min(int(form.get("stride") or 4), 8))
    except (ValueError, TypeError): stride = 4

    avatar_bytes  = await avatar.read()
    driving_bytes = await driving.read()

    if len(avatar_bytes) > 12 * 1024 * 1024:
        return JSONResponse({"error": "avatar image too large (max 12 MB)"}, 413)
    if len(driving_bytes) > 200 * 1024 * 1024:
        return JSONResponse({"error": "driving video too large (max 200 MB)"}, 413)

    avatar_name  = avatar.filename  or f"avatar_{int(time.time())}.jpg"
    driving_name = driving.filename or f"driving_{int(time.time())}.mp4"

    def fn():
        return {
            "video": _run_mimic_motion(avatar_bytes, avatar_name, driving_bytes, driving_name,
                                        num_frames=num_frames, resolution=resolution, fps=fps,
                                        steps=steps, guidance=guidance, stride=stride),
            "model": "mimic-motion"
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_animate_anyone(avatar_bytes: bytes, avatar_name: str,
                         driving_bytes: bytes, driving_name: str,
                         width: int = 512, height: int = 784,
                         chunk: int = 16, steps: int = 20,
                         cfg: float = 3.5, fps: int = 30) -> str:
    """Run AnimateAnyone in its own venv; return base64-encoded mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="aa_")
    try:
        avatar_path  = os.path.join(tmp, avatar_name)
        driving_path = os.path.join(tmp, driving_name)
        output_path  = os.path.join(tmp, "output.mp4")
        with open(avatar_path,  "wb") as f: f.write(avatar_bytes)
        with open(driving_path, "wb") as f: f.write(driving_bytes)
        cmd = [
            AA_VENV_PYTHON, AA_SCRIPT,
            "--image",  avatar_path,
            "--video",  driving_path,
            "--output", output_path,
            "--width",  str(width),
            "--height", str(height),
            "--chunk",  str(chunk),
            "--steps",  str(steps),
            "--cfg",    str(cfg),
            "--fps",    str(fps),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "AnimateAnyone failed")[-1000:])
        if not os.path.exists(output_path):
            raise RuntimeError("AnimateAnyone produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/animate-anyone")
async def animate_anyone_endpoint(req: Request, avatar: UploadFile = File(...), driving: UploadFile = File(...)):
    """Animate `avatar` image with `driving` video via AnimateAnyone. Returns NDJSON → {video: base64_mp4}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(AA_VENV_PYTHON):
        return JSONResponse({"error": "AnimateAnyone venv not found — check /home/work/AnimateAnyone/venv/"}, 503)

    form = await req.form()
    try:    width  = int(form.get("width")  or 512)
    except (ValueError, TypeError): width  = 512
    try:    height = int(form.get("height") or 784)
    except (ValueError, TypeError): height = 784
    try:    chunk  = max(8, min(int(form.get("chunk") or 16), 32))
    except (ValueError, TypeError): chunk  = 16
    try:    steps  = max(10, min(int(form.get("steps") or 20), 40))
    except (ValueError, TypeError): steps  = 20
    try:    cfg    = float(form.get("cfg") or 3.5)
    except (ValueError, TypeError): cfg    = 3.5
    try:    fps    = max(8, min(int(form.get("fps") or 30), 60))
    except (ValueError, TypeError): fps    = 30

    avatar_bytes  = await avatar.read()
    driving_bytes = await driving.read()

    if len(avatar_bytes) > 12 * 1024 * 1024:
        return JSONResponse({"error": "avatar image too large (max 12 MB)"}, 413)
    if len(driving_bytes) > 200 * 1024 * 1024:
        return JSONResponse({"error": "driving video too large (max 200 MB)"}, 413)

    avatar_name  = avatar.filename  or f"avatar_{int(time.time())}.jpg"
    driving_name = driving.filename or f"driving_{int(time.time())}.mp4"

    def fn():
        return {
            "video": _run_animate_anyone(avatar_bytes, avatar_name, driving_bytes, driving_name,
                                          width=width, height=height, chunk=chunk,
                                          steps=steps, cfg=cfg, fps=fps),
            "model": "animate-anyone"
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_wan_animate(avatar_bytes: bytes, avatar_name: str,
                     driving_bytes: bytes, driving_name: str,
                     prompt: str = "a person performing the motion, high quality",
                     frames: int = 49, steps: int = 15,
                     width: int = 832, height: int = 480) -> str:
    """Wan2.2-Animate-14B (Q8 GGUF) on the 6800 XT: image + driving video -> base64 mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="wananim_")
    try:
        avatar_path  = os.path.join(tmp, avatar_name)
        driving_path = os.path.join(tmp, driving_name)
        output_path  = os.path.join(tmp, "output.mp4")
        with open(avatar_path,  "wb") as f: f.write(avatar_bytes)
        with open(driving_path, "wb") as f: f.write(driving_bytes)
        cmd = [WANANIM_VENV_PYTHON, WANANIM_SCRIPT,
               "--image", avatar_path, "--video", driving_path, "--output", output_path,
               "--prompt", prompt, "--frames", str(frames), "--steps", str(steps),
               "--width", str(width), "--height", str(height)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "Wan2.2-Animate failed")[-1000:])
        if not os.path.exists(output_path):
            raise RuntimeError("Wan2.2-Animate produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/wan-animate")
async def wan_animate_endpoint(req: Request, avatar: UploadFile = File(...), driving: UploadFile = File(...)):
    """Wan2.2-Animate-14B: animate `avatar` image with `driving` video. NDJSON -> {video: base64_mp4}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(WANANIM_SCRIPT):
        return JSONResponse({"error": "Wan2.2-Animate runner not found"}, 503)
    # ComfyUI-ROCm (6800 XT) must be up on :8189
    try:
        requests.get("http://127.0.0.1:8189/system_stats", timeout=3).raise_for_status()
    except Exception:
        return JSONResponse({"error": "ComfyUI-ROCm (port 8189) is not running — start comfyui-rocm.service"}, 503)

    form = await req.form()
    prompt = (form.get("prompt") or "a person performing the motion, high quality").strip()
    try:    frames = max(5, min(int(form.get("frames") or 33), 121))
    except (ValueError, TypeError): frames = 49
    try:    steps  = max(6, min(int(form.get("steps") or 15), 30))
    except (ValueError, TypeError): steps  = 15

    avatar_bytes  = await avatar.read()
    driving_bytes = await driving.read()
    if len(avatar_bytes) > 12 * 1024 * 1024:
        return JSONResponse({"error": "character image too large (max 12 MB)"}, 413)
    if len(driving_bytes) > 200 * 1024 * 1024:
        return JSONResponse({"error": "driving video too large (max 200 MB)"}, 413)

    avatar_name  = avatar.filename  or f"char_{int(time.time())}.jpg"
    driving_name = driving.filename or f"drive_{int(time.time())}.mp4"

    def fn():
        return {
            "video": _run_wan_animate(avatar_bytes, avatar_name, driving_bytes, driving_name,
                                      prompt=prompt, frames=frames, steps=steps),
            "model": "wan2.2-animate"
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_vace(driving_bytes: bytes, driving_name: str,
              ref_bytes: bytes = b"", ref_name: str = "",
              prompt: str = "high quality, detailed, smooth motion",
              frames: int = 49, steps: int = 20, strength: float = 1.0) -> str:
    """Wan2.1-VACE-14B (Q8 GGUF) on the 6800 XT: control video (+ optional ref image) -> base64 mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="vace_")
    try:
        driving_path = os.path.join(tmp, driving_name)
        output_path  = os.path.join(tmp, "output.mp4")
        with open(driving_path, "wb") as f: f.write(driving_bytes)
        cmd = [WANANIM_VENV_PYTHON, VACE_SCRIPT,
               "--video", driving_path, "--output", output_path, "--prompt", prompt,
               "--frames", str(frames), "--steps", str(steps), "--strength", str(strength)]
        if ref_bytes:
            ref_path = os.path.join(tmp, ref_name or "ref.png")
            with open(ref_path, "wb") as f: f.write(ref_bytes)
            cmd += ["--image", ref_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "VACE failed")[-1000:])
        if not os.path.exists(output_path):
            raise RuntimeError("VACE produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/vace")
async def vace_endpoint(req: Request, driving: UploadFile = File(...), avatar: UploadFile = File(None)):
    """Wan2.1-VACE-14B motion/structure control: control video (+ optional reference image). NDJSON -> {video}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(VACE_SCRIPT):
        return JSONResponse({"error": "VACE runner not found"}, 503)
    try:
        requests.get("http://127.0.0.1:8189/system_stats", timeout=3).raise_for_status()
    except Exception:
        return JSONResponse({"error": "ComfyUI-ROCm (port 8189) is not running — start comfyui-rocm.service"}, 503)

    form = await req.form()
    prompt = (form.get("prompt") or "high quality, detailed, smooth motion").strip()
    try:    frames   = max(5, min(int(form.get("frames") or 33), 121))
    except (ValueError, TypeError): frames = 49
    try:    steps    = max(6, min(int(form.get("steps") or 20), 40))
    except (ValueError, TypeError): steps = 20
    try:    strength = max(0.0, min(float(form.get("strength") or 1.0), 2.0))
    except (ValueError, TypeError): strength = 1.0

    driving_bytes = await driving.read()
    if len(driving_bytes) > 200 * 1024 * 1024:
        return JSONResponse({"error": "control video too large (max 200 MB)"}, 413)
    driving_name = driving.filename or f"ctrl_{int(time.time())}.mp4"

    ref_bytes, ref_name = b"", ""
    if avatar is not None:
        ref_bytes = await avatar.read()
        if len(ref_bytes) > 12 * 1024 * 1024:
            return JSONResponse({"error": "reference image too large (max 12 MB)"}, 413)
        ref_name = avatar.filename or f"ref_{int(time.time())}.jpg"

    def fn():
        return {
            "video": _run_vace(driving_bytes, driving_name, ref_bytes, ref_name,
                               prompt=prompt, frames=frames, steps=steps, strength=strength),
            "model": "wan-vace"
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_champ(avatar_bytes: bytes, avatar_name: str,
               driving_bytes: bytes, driving_name: str,
               width: int = 512, height: int = 512, frames: int = 9999,
               steps: int = 20, guidance: float = 3.5, fps: int = 30) -> str:
    """Run CHAMP across both GPUs; return base64-encoded mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="champ_")
    try:
        avatar_path  = os.path.join(tmp, avatar_name)
        driving_path = os.path.join(tmp, driving_name)
        output_path  = os.path.join(tmp, "output.mp4")
        with open(avatar_path,  "wb") as f: f.write(avatar_bytes)
        with open(driving_path, "wb") as f: f.write(driving_bytes)
        cmd = [
            CHAMP_VENV_PYTHON, CHAMP_SCRIPT,
            "--image",    avatar_path,
            "--video",    driving_path,
            "--output",   output_path,
            "--width",    str(width),
            "--height",   str(height),
            "--frames",   str(frames),
            "--steps",    str(steps),
            "--guidance", str(guidance),
            "--fps",      str(fps),
            "--gpu",      "-1",    # auto dual-GPU split
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "CHAMP failed")[-1000:])
        if not os.path.exists(output_path):
            raise RuntimeError("CHAMP produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/champ-status")
async def champ_status():
    """Return CHAMP guidance mode: full (4-guidance) or pose (DWPose-only)."""
    smpl_pkl    = "/home/work/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl"
    hmr2_ckpt   = "/home/work/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
    d2_model    = "/home/work/champ/pretrained_models/detectron2/model_final_f05665.pkl"
    full_ready  = all(os.path.exists(p) for p in [smpl_pkl, hmr2_ckpt, d2_model])
    return JSONResponse({
        "mode":        "full" if full_ready else "pose",
        "description": "depth+normal+semantic+dwpose" if full_ready else "DWPose-only",
        "smpl_ready":  os.path.exists(smpl_pkl),
        "hmr2_ready":  os.path.exists(hmr2_ckpt),
        "d2_ready":    os.path.exists(d2_model),
        "smpl_path":   smpl_pkl if not full_ready else None,
    })


@app.post("/champ")
async def champ_endpoint(req: Request, avatar: UploadFile = File(...), driving: UploadFile = File(...)):
    """Animate `avatar` image with `driving` video via CHAMP (dual-GPU). Returns NDJSON → {video: base64_mp4}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(CHAMP_VENV_PYTHON):
        return JSONResponse({"error": "CHAMP venv not found — check /home/work/champ/venv/"}, 503)

    form = await req.form()
    try:    width   = int(form.get("width")   or 512)
    except (ValueError, TypeError): width   = 512
    try:    height  = int(form.get("height")  or 512)
    except (ValueError, TypeError): height  = 512
    try:    frames  = max(16, min(int(form.get("frames") or 9999), 9999))
    except (ValueError, TypeError): frames  = 9999
    try:    steps   = max(10, min(int(form.get("steps")  or 20), 40))
    except (ValueError, TypeError): steps   = 20
    try:    guidance = float(form.get("guidance") or 3.5)
    except (ValueError, TypeError): guidance = 3.5
    try:    fps     = max(8, min(int(form.get("fps")    or 30), 60))
    except (ValueError, TypeError): fps     = 30

    avatar_bytes  = await avatar.read()
    driving_bytes = await driving.read()

    if len(avatar_bytes) > 12 * 1024 * 1024:
        return JSONResponse({"error": "avatar image too large (max 12 MB)"}, 413)
    if len(driving_bytes) > 200 * 1024 * 1024:
        return JSONResponse({"error": "driving video too large (max 200 MB)"}, 413)

    avatar_name  = avatar.filename  or f"avatar_{int(time.time())}.jpg"
    driving_name = driving.filename or f"driving_{int(time.time())}.mp4"

    def fn():
        return {
            "video": _run_champ(avatar_bytes, avatar_name, driving_bytes, driving_name,
                                 width=width, height=height, frames=frames,
                                 steps=steps, guidance=guidance, fps=fps),
            "model": "champ"
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_wan(prompt: str, negative: str = "", frames: int = 81,
             width: int = 832, height: int = 480,
             steps: int = 50, guidance: float = 5.0,
             fps: int = 16, seed: int = 42) -> str:
    """Run Wan2.1 T2V in its own venv; return base64-encoded mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="wan_")
    try:
        output_path = os.path.join(tmp, "output.mp4")
        cmd = [
            WAN_VENV_PYTHON, WAN_SCRIPT,
            "--prompt",   prompt,
            "--output",   output_path,
            "--frames",   str(frames),
            "--width",    str(width),
            "--height",   str(height),
            "--steps",    str(steps),
            "--guidance", str(guidance),
            "--fps",      str(fps),
            "--seed",     str(seed),
        ]
        if negative:
            cmd += ["--negative", negative]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "Wan2.1 failed")[-1200:])
        if not os.path.exists(output_path):
            raise RuntimeError("Wan2.1 produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/wan-video")
async def wan_video_endpoint(req: Request):
    """Generate a short video from a text prompt using Wan2.1 T2V-1.3B."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(WAN_VENV_PYTHON):
        return JSONResponse({"error": "Wan2.1 venv not found — check /home/work/Wan2.1/venv/"}, 503)

    data, err = await _safe_json(req)
    if err: return err
    prompt   = (data.get("prompt") or "").strip()
    negative = (data.get("negative") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, 400)

    try:    frames   = max(17, min(int(data.get("frames",  81)),  161))
    except (ValueError, TypeError): frames   = 81
    try:    width    = int(data.get("width",    832))
    except (ValueError, TypeError): width    = 832
    try:    height   = int(data.get("height",   480))
    except (ValueError, TypeError): height   = 480
    try:    steps    = max(10, min(int(data.get("steps",   50)),  80))
    except (ValueError, TypeError): steps    = 50
    try:    guidance = float(data.get("guidance", 5.0))
    except (ValueError, TypeError): guidance = 5.0
    try:    fps      = max(8,  min(int(data.get("fps",     16)),  30))
    except (ValueError, TypeError): fps      = 16
    try:    seed     = int(data.get("seed", 42))
    except (ValueError, TypeError): seed     = 42

    def fn():
        return {
            "video":  _run_wan(prompt, negative, frames, width, height, steps, guidance, fps, seed),
            "prompt": prompt,
            "model":  "wan2.1-t2v-1.3b",
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


def _run_wan_i2v(image_bytes: bytes, image_name: str,
                 prompt: str = "", negative: str = "",
                 frames: int = 81, width: int = 832, height: int = 480,
                 steps: int = 50, guidance: float = 5.0,
                 fps: int = 16, seed: int = 42) -> str:
    """Run Wan2.1 I2V in its own venv; return base64-encoded mp4."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="wan_i2v_")
    try:
        image_path  = os.path.join(tmp, image_name)
        output_path = os.path.join(tmp, "output.mp4")
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        cmd = [
            WAN_VENV_PYTHON, WAN_I2V_SCRIPT,
            "--image",    image_path,
            "--output",   output_path,
            "--frames",   str(frames),
            "--width",    str(width),
            "--height",   str(height),
            "--steps",    str(steps),
            "--guidance", str(guidance),
            "--fps",      str(fps),
            "--seed",     str(seed),
        ]
        if prompt:   cmd += ["--prompt",   prompt]
        if negative: cmd += ["--negative", negative]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "Wan2.1 I2V failed")[-1200:])
        if not os.path.exists(output_path):
            raise RuntimeError("Wan2.1 I2V produced no output file")
        with open(output_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/wan-i2v")
async def wan_i2v_endpoint(req: Request, image: UploadFile = File(...),
                            prompt: str = "", steps: int = 50,
                            frames: int = 81, fps: int = 16):
    """Animate an image using Wan2.1 I2V-14B. Returns NDJSON → {video: base64_mp4}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit: 5 req/min"}, 429)
    if not os.path.exists(WAN_VENV_PYTHON):
        return JSONResponse({"error": "Wan2.1 venv not found — check /home/work/Wan2.1/venv/"}, 503)

    form = await req.form()
    prompt   = (form.get("prompt")   or "").strip()
    negative = (form.get("negative") or "").strip()
    try:    steps   = max(10, min(int(form.get("steps",  50)),  80))
    except (ValueError, TypeError): steps   = 50
    try:    frames  = max(17, min(int(form.get("frames", 81)),  161))
    except (ValueError, TypeError): frames  = 81
    try:    fps     = max(8,  min(int(form.get("fps",    16)),  30))
    except (ValueError, TypeError): fps     = 16
    try:    seed    = int(form.get("seed", 42))
    except (ValueError, TypeError): seed    = 42

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 20 MB)"}, 413)
    image_name = image.filename or f"input_{int(time.time())}.jpg"

    def fn():
        return {
            "video":  _run_wan_i2v(image_bytes, image_name, prompt, negative,
                                   frames, 832, 480, steps, 5.0, fps, seed),
            "prompt": prompt,
            "model":  "wan2.1-i2v-14b",
        }
    return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")


@app.get("/wan-video/status")
async def wan_video_status():
    """Check whether Wan2.1 venv is installed."""
    ready = os.path.exists(WAN_VENV_PYTHON)
    return {"ready": ready, "model": "Wan2.1-T2V-1.3B"}


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
        except (ValueError, TypeError): pass
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
        def fn():
            return {"image": _edit_image(img_bytes, fname, prompt, max(4, min(steps, 40))),
                              "prompt": prompt, "model": "flux-kontext"}
        return StreamingResponse(_stream_image_job(fn), media_type="application/x-ndjson")
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/search")
async def search(req: Request):
    """Web search via local SearXNG. Returns top results as a text block."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "search"):
        return JSONResponse({"error": "rate limit: 30 req/min"}, 429)
    data, err = await _safe_json(req)
    if err: return err
    query = (data.get("query") or "").strip()
    try:
        n = max(1, min(int(data.get("n", 5)), 10))
    except (ValueError, TypeError):
        n = 5
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
def _bridge_watchdog() -> str:
    p = "/home/work/fraqtoos/logs/watchdog_latest.json"
    if not os.path.exists(p):
        return "No watchdog report."
    with open(p) as _f: d = json.load(_f)
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
    with open(p) as _f: d = json.load(_f)
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
        try:
            with open(p) as _f: state = json.load(_f)
        except Exception: pass
    out = ["## Bot State"]
    if not state:
        out.append("(no state file)")
    for k, v in state.items():
        out.append(f"- **{k}**: {v}")
    return "\n".join(out)


def _bridge_portfolio() -> str:
    ctx = "/home/work/fraqtoos/logs/ai_context.json"
    if not os.path.exists(ctx):
        return "No portfolio data yet — run portfolio_bot.py first."
    try:
        with open(ctx) as _f: d = json.load(_f)
        today = sorted(d.keys())[-1]
        summary = d[today].get("Portfolio Bot", "")
        if not summary:
            return f"No portfolio entry for {today}."
        return f"## Portfolio Summary — {today}\n\n{summary}"
    except Exception as e:
        return f"Portfolio read error: {e}"


def _bridge_help() -> str:
    return ("## Bot bridge commands\n"
            "- `/watchdog` — bot health + AI diagnosis\n"
            "- `/digest` — today's per-bot summaries\n"
            "- `/bots` — orchestrator state\n"
            "- `/portfolio` — portfolio P&L from 5Paisa + Kite\n"
            "- `/help` — this message")


_BRIDGE = {
    "watchdog":   _bridge_watchdog,
    "digest":     _bridge_digest,
    "bots":       _bridge_bots,
    "portfolio":  _bridge_portfolio,
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


# ─── Semantic search (nomic-embed-text) ──────────────────────────────
EMBED_CACHE_FILE = "/home/work/fraqtoos-chat/conv_embeddings.json"
EMBED_MODEL      = "nomic-embed-text"

def _embed(text: str) -> list:
    """Get embedding vector from nomic-embed-text. Returns [] on failure."""
    try:
        r = requests.post(f"{OLLAMA}/api/embeddings",
                          json={"model": EMBED_MODEL, "prompt": text[:2000]},
                          timeout=10)
        return r.json().get("embedding", [])
    except Exception:
        return []

def _cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def _conv_text(c: dict) -> str:
    """Extract representative text from a conversation for embedding."""
    parts = [c.get("title", "")]
    for m in c.get("history", [])[:8]:
        role = m.get("role", "")
        txt  = (m.get("content") or "")[:400]
        if txt:
            parts.append(f"{role}: {txt}")
    return " ".join(parts)[:2000]

def _load_embed_cache() -> dict:
    try:
        with open(EMBED_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_embed_cache(cache: dict):
    tmp = EMBED_CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, EMBED_CACHE_FILE)

def _nomic_available() -> bool:
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=3).json()
        return any(EMBED_MODEL in m["name"] for m in tags.get("models", []))
    except Exception:
        return False


# ─── Obsidian vault RAG ───────────────────────────────────────────────
# Answers grounded in the user's canonical Obsidian vault (see CLAUDE.md /
# memory). Builds a small cached embedding index over the .md notes and
# retrieves the most relevant chunks to ground a local-model answer.
VAULT_DIR        = "/home/work/obsidian-vault"
VAULT_INDEX_FILE = "/home/work/fraqtoos-chat/vault_index.json"
_vault_lock      = asyncio.Lock()


def _vault_files() -> list:
    out = []
    for root, dirs, files in os.walk(VAULT_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", ".obsidian", "Templates")]
        for fn in files:
            if fn.endswith(".md"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _chunk_md(text: str, size: int = 1000, overlap: int = 150) -> list:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


def _build_vault_index(force: bool = False) -> dict:
    """Return {'sig','chunks':[{path,text,emb}], 'built','files'}; rebuild only
    when the set of files or their mtimes changed (or force=True)."""
    files = _vault_files()
    sig = json.dumps([[os.path.relpath(p, VAULT_DIR), int(os.path.getmtime(p))] for p in files])
    if not force:
        try:
            with open(VAULT_INDEX_FILE) as f:
                cached = json.load(f)
            if cached.get("sig") == sig:
                return cached
        except Exception:
            pass
    chunks = []
    for p in files:
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            continue
        rel = os.path.relpath(p, VAULT_DIR)
        for ch in _chunk_md(txt):
            emb = _embed(ch)
            if emb:
                chunks.append({"path": rel, "text": ch, "emb": emb})
    idx = {"sig": sig, "chunks": chunks, "built": time.time(), "files": len(files)}
    try:
        tmp = VAULT_INDEX_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(idx, f)
        os.replace(tmp, VAULT_INDEX_FILE)
    except Exception:
        pass
    return idx


def _vault_retrieve(query: str, k: int = 6) -> list:
    idx = _build_vault_index()
    qe = _embed(query)
    if not qe:
        return []
    scored = [(_cosine(qe, c["emb"]), c) for c in idx.get("chunks", [])]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"path": c["path"], "text": c["text"], "score": round(s, 3)}
            for s, c in scored[:k] if s > 0.2]


@app.get("/ask-vault/status")
async def ask_vault_status(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        with open(VAULT_INDEX_FILE) as f:
            idx = json.load(f)
        built, chunks = idx.get("built", 0), len(idx.get("chunks", []))
    except Exception:
        built, chunks = 0, 0
    return {"nomic_ready": _nomic_available(), "vault_files": len(_vault_files()),
            "indexed_chunks": chunks, "built": built}


@app.post("/ask-vault/reindex")
async def ask_vault_reindex(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "imagine"):
        return JSONResponse({"error": "rate limit"}, 429)
    if not _nomic_available():
        return JSONResponse({"error": f"{EMBED_MODEL} not installed"}, 503)
    loop = asyncio.get_running_loop()
    async with _vault_lock:
        idx = await loop.run_in_executor(None, _build_vault_index, True)
    return {"ok": True, "files": idx.get("files", 0), "chunks": len(idx.get("chunks", []))}


@app.post("/ask-vault")
async def ask_vault(req: Request):
    """RAG over the Obsidian vault: retrieve top notes, answer with a local model."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "chat"):
        return JSONResponse({"error": "rate limit: 20 req/min"}, 429)
    data, err = await _safe_json(req)
    if err:
        return err
    q = (data.get("text") or data.get("query") or "").strip()
    if not q:
        return JSONResponse({"error": "query required"}, 400)
    if not _nomic_available():
        return JSONResponse({"error": f"{EMBED_MODEL} not installed (ollama pull {EMBED_MODEL})"}, 503)
    model = data.get("model") or ROUTING_TARGETS["general"]
    loop = asyncio.get_running_loop()
    async with _vault_lock:
        hits = await loop.run_in_executor(None, _vault_retrieve, q, 6)
    if not hits:
        return JSONResponse({"error": "no relevant vault notes found"}, 404)
    context = "\n\n".join(f"[{h['path']}]\n{h['text']}" for h in hits)
    system = ("You are answering from the user's personal Obsidian vault. "
              "Use ONLY the notes below. Cite the [path] of every note you draw on. "
              "If the notes don't contain the answer, say so plainly.\n\n"
              "=== VAULT NOTES ===\n" + context)
    messages = [{"role": "user", "content": q}]
    return StreamingResponse(ollama_stream(model, messages, system),
                             media_type="text/plain")


# ─── Conversation file cache (avoids re-reading unchanged files on every search) ──
_conv_cache: dict[str, object] = {}        # filename → parsed JSON content
_conv_cache_mtime: dict[str, float] = {}   # filename → mtime at last read


def _load_conv_cached(filepath: str, fn: str) -> "object | None":
    """Return parsed conversation JSON, re-reading only if mtime changed."""
    try:
        mtime = os.path.getmtime(filepath)
    except OSError:
        return None
    if _conv_cache_mtime.get(fn) == mtime:
        return _conv_cache.get(fn)
    try:
        with open(filepath) as f:
            data = json.load(f)
        _conv_cache[fn] = data
        _conv_cache_mtime[fn] = mtime
        return data
    except Exception:
        return None


# ─── Conversation search ──────────────────────────────────────────────
@app.get("/conversations/search/q")
async def conv_search(req: Request, q: str = ""):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    q = (q or "").strip()
    if not q:
        return {"matches": []}

    # Load conversations using mtime cache — only re-read files that changed
    convs = []
    for fn in os.listdir(CONV_DIR):
        if not fn.endswith(".json"):
            continue
        data = _load_conv_cached(os.path.join(CONV_DIR, fn), fn)
        if data is not None:
            convs.append(data)

    # ── Semantic search ──────────────────────────────────────────────
    if _nomic_available():
        q_emb = _embed(q)
        if q_emb:
            cache = _load_embed_cache()
            cache_dirty = False
            scored = []
            for c in convs:
                cid     = c.get("id", "")
                updated = c.get("updated", 0)
                cached  = cache.get(cid, {})
                # Recompute if missing or conversation was updated since last embed
                if not cached.get("emb") or cached.get("updated") != updated:
                    emb = _embed(_conv_text(c))
                    if emb:
                        cache[cid] = {"emb": emb, "updated": updated}
                        cache_dirty = True
                    else:
                        emb = cached.get("emb", [])
                else:
                    emb = cached["emb"]
                score = _cosine(q_emb, emb)
                scored.append((score, c))
            if cache_dirty:
                async with _embed_cache_lock:
                    current = _load_embed_cache()
                    current.update(cache)
                    _save_embed_cache(current)
            scored.sort(key=lambda x: x[0], reverse=True)
            matches = []
            for score, c in scored[:20]:
                if score < 0.25:
                    continue
                # Find a text snippet for context
                joined = " ".join(m.get("content", "")[:200] for m in c.get("history", [])[:4])
                matches.append({
                    "id":      c.get("id"),
                    "title":   c.get("title", "Untitled"),
                    "updated": c.get("updated", 0),
                    "snippet": joined[:120],
                    "score":   round(score, 3),
                    "semantic": True,
                })
            return {"matches": matches, "mode": "semantic"}

    # ── Fallback: text search ─────────────────────────────────────────
    ql = q.lower()
    matches = []
    for c in convs:
        title  = (c.get("title") or "").lower()
        joined = "\n".join(m.get("content", "") for m in c.get("history", [])).lower()
        if ql in title or ql in joined:
            idx     = joined.find(ql)
            start   = max(0, idx - 40)
            snippet = joined[start:idx+len(ql)+80].replace("\n", " ") if idx >= 0 else ""
            matches.append({
                "id":       c.get("id"),
                "title":    c.get("title", "Untitled"),
                "updated":  c.get("updated", 0),
                "snippet":  snippet,
                "in_title": ql in title,
                "semantic": False,
            })
    matches.sort(key=lambda x: x.get("updated", 0), reverse=True)
    return {"matches": matches[:30], "mode": "text"}


@app.post("/conversations/reindex")
async def conv_reindex(req: Request):
    """Rebuild the semantic embedding cache for all conversations."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    if not _nomic_available():
        return JSONResponse({"error": "nomic-embed-text not installed"}, 503)
    cache = {}
    count = 0
    for fn in os.listdir(CONV_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fn)) as f:
                c = json.load(f)
            emb = _embed(_conv_text(c))
            if emb:
                cache[c.get("id", fn)] = {"emb": emb, "updated": c.get("updated", 0)}
                count += 1
        except Exception:
            continue
    async with _embed_cache_lock:
        _save_embed_cache(cache)
    return {"indexed": count, "dims": 768}


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
    try:
        with open(path) as f:
            c = json.load(f)
    except Exception:
        return JSONResponse({"error": "conversation corrupted"}, 500)
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
    # Re-read inside the lock so we don't clobber a concurrent /conversations save
    async with _conv_lock:
        try:
            with open(path) as f:
                latest = json.load(f)
        except Exception:
            latest = c
        latest["title"] = title
        latest["updated"] = int(time.time())
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(latest, f, indent=2)
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
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return JSONResponse({"error": "conversation corrupted"}, 500)


@app.get("/conversations/{conv_id}/export")
async def conv_export(conv_id: str, req: Request):
    """Download a conversation as a Markdown file."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        path = _conv_path(conv_id)
    except ValueError:
        return JSONResponse({"error": "invalid id"}, 400)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, 404)
    try:
        with open(path) as f:
            c = json.load(f)
    except Exception:
        return JSONResponse({"error": "conversation corrupted"}, 500)

    title = c.get("title", "Untitled")
    lines = [f"# {title}", "",
             f"- **Model:** {c.get('model', '')}",
             f"- **Messages:** {len(c.get('history', []))}",
             f"- **Exported:** {time.strftime('%Y-%m-%d %H:%M:%S')}", "", "---", ""]
    for m in c.get("history", []):
        role = (m.get("role") or "?").capitalize()
        who = {"User": "🧑 You", "Assistant": "🤖 Assistant"}.get(role, role)
        lines.append(f"### {who}")
        lines.append("")
        lines.append((m.get("content") or "").rstrip())
        lines.append("")
    md = "\n".join(lines)
    safe = "".join(ch for ch in (title or "conversation") if ch.isalnum() or ch in " _-").strip()[:48] or "conversation"
    return Response(
        content=md, media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe}.md"'})


@app.post("/conversations")
async def conv_save(req: Request):
    """Create or update a conversation. Body: {id?, title, history, model}."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data, err = await _safe_json(req)
    if err: return err
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
    async with _conv_lock:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    old = json.load(f)
                record["created"] = old.get("created", now)
                # Preserve title set by autotitle if caller didn't supply one
                if not data.get("title") and old.get("title"):
                    record["title"] = old["title"]
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
        with open(MEMORY_FILE) as _f: return json.load(_f)
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
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, 400)
    fact = (data.get("fact") or "").strip()
    if not fact:
        return JSONResponse({"error": "fact required"}, 400)
    if len(fact) > 500:
        fact = fact[:500]
    async with _memory_lock:
        items = _load_memory()
        new_id = f"m_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
        items.append({"id": new_id, "fact": fact, "ts": int(time.time())})
        _save_memory(items)
    # Shared memory: best-effort mirror into Odysseus's vector store so both
    # apps recall the same facts. Non-blocking — never fails the local save.
    if ODYSSEUS_PASS:
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _odysseus_memory_add, fact, "fact")
        except Exception:
            pass
    return {"id": new_id, "fact": fact, "count": len(items)}


@app.delete("/memory/{mem_id}")
async def memory_delete(mem_id: str, req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    async with _memory_lock:
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
    try:
        data = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, 400)
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
            facts = [str(f).strip().replace('\n', ' ')[:100] for f in facts if isinstance(f, str) and f.strip()][:3]
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
        async with _embed_cache_lock:
            cache = _load_embed_cache()
            if conv_id in cache:
                del cache[conv_id]
                _save_embed_cache(cache)
        return {"ok": True}
    return JSONResponse({"error": "not found"}, 404)


# ─── Message feedback (👍/👎) — cross-conversation model quality log ─
FEEDBACK_FILE = "/home/work/fraqtoos-chat/feedback.json"


def _load_feedback() -> list:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _norm_model(name: str) -> str:
    return (name or "unknown").removesuffix(":latest")


@app.post("/feedback")
async def feedback_add(req: Request):
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    data, err = await _safe_json(req)
    if err:
        return err
    score = data.get("score")
    if score not in (1, -1):
        return JSONResponse({"error": "score must be 1 or -1"}, 400)
    items = _load_feedback()
    items.append({
        "model":   _norm_model(data.get("model", "")),
        "score":   score,
        "snippet": str(data.get("snippet", ""))[:200],
        "conv_id": str(data.get("conv_id") or "")[:64],
        "ts":      int(time.time()),
    })
    items = items[-2000:]
    tmp = FEEDBACK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
    os.replace(tmp, FEEDBACK_FILE)
    return {"ok": True, "total": len(items)}


@app.get("/feedback/stats")
async def feedback_stats():
    """Aggregate 👍/👎 counts per model across all conversations."""
    stats: dict = {}
    for it in _load_feedback():
        m = it.get("model", "unknown")
        s = stats.setdefault(m, {"up": 0, "down": 0})
        if it.get("score", 0) > 0:
            s["up"] += 1
        else:
            s["down"] += 1
    return {"stats": stats}


@app.get("/chia-harvester")
async def chia_harvester_status():
    """Return Chia harvester running state."""
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(None, lambda: _pgrep_safe("chia_harvester"))
    running = bool(r and r.returncode == 0)
    return {"running": running}

@app.post("/chia-harvester")
async def chia_harvester_toggle(req: Request):
    """Start or stop the Chia harvester. Body: {"action": "start"|"stop"}"""
    body, err = await _safe_json(req)
    if err: return err
    action = body.get("action")
    if action not in ("start", "stop"):
        return JSONResponse({"error": "action must be start or stop"}, 400)
    cmd = ["chia", action, "harvester"]
    loop = asyncio.get_running_loop()
    def _do():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            return f"chia {action} harvester timed out — command still running in background"
        except Exception as e:
            return f"error: {e}"
    output = await loop.run_in_executor(None, _do)
    r = await loop.run_in_executor(None, lambda: _pgrep_safe("chia_harvester"))
    running = bool(r and r.returncode == 0)
    return {"running": running, "output": output}


@app.get("/gpu")
async def gpu_stats():
    try:
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'], timeout=3
        ).decode().strip())
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({"vram_used": int(parts[0]), "vram_total": int(parts[1]),
                             "gpu_util": int(parts[2]), "temp": int(parts[3])})
        if not gpus:
            return JSONResponse({"error": "no gpu data"}, 500)
        total_used  = sum(g["vram_used"]  for g in gpus)
        total_vram  = sum(g["vram_total"] for g in gpus)
        avg_util    = round(sum(g["gpu_util"] for g in gpus) / len(gpus))
        max_temp    = max(g["temp"] for g in gpus)
        return {"vram_used": total_used, "vram_total": total_vram,
                "gpu_util": avg_util, "temp": max_temp,
                "gpu_count": len(gpus), "gpus": gpus}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _amd_card_path():
    """Find the amdgpu card's sysfs device dir (the 6800 XT Ollama runs on)."""
    import glob
    for c in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            drv = os.path.basename(os.path.realpath(os.path.join(c, "driver")))
        except Exception:
            drv = ""
        if drv == "amdgpu" and os.path.exists(os.path.join(c, "mem_info_vram_total")):
            return c
    return None


@app.get("/gpu-amd")
async def gpu_amd():
    """Live AMD GPU (RX 6800 XT) VRAM/util/temp straight from sysfs — fast, no sudo."""
    import glob
    c = _amd_card_path()
    if not c:
        return JSONResponse({"error": "no amdgpu card found"}, 404)
    total = _read_int(f"{c}/mem_info_vram_total")
    used  = _read_int(f"{c}/mem_info_vram_used")
    if total is None or used is None or total == 0:
        return JSONResponse({"error": "vram info unavailable"}, 500)
    busy = _read_int(f"{c}/gpu_busy_percent")
    temp = None
    hw = sorted(glob.glob(f"{c}/hwmon/hwmon*"))
    if hw:
        t = _read_int(f"{hw[0]}/temp1_input")
        temp = round(t / 1000) if t is not None else None
    return {
        "name": "RX 6800 XT",
        "vram_used_gb":  round(used / 1073741824, 1),
        "vram_total_gb": round(total / 1073741824, 1),
        "vram_pct":      round(used / total * 100),
        "gpu_util":      busy,
        "temp":          temp,
    }


@app.post("/comfy-interrupt")
async def comfy_interrupt(req: Request):
    """Cancel the running image/video generation by interrupting both ComfyUI
    instances (8189 ROCm image+video / 8188 CUDA avatar) — frees the GPU immediately."""
    loop = asyncio.get_running_loop()
    def _hit(url):
        try:
            requests.post(f"{url}/interrupt", timeout=4)
            return "interrupted"
        except Exception:
            return "unreachable"
    results = {}
    for name, url in (("image_8188", "http://127.0.0.1:8188"),
                      ("rocm_8189",  "http://127.0.0.1:8189")):
        results[name] = await loop.run_in_executor(None, _hit, url)
    return {"ok": True, "interrupted": results}


@app.get("/health")
async def health():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        models = []
    return {
        "status":        "ok",
        "ollama_models": models,
        "image_ready":   _comfyui_ready(),
        "search_ready":  _searx_up(),
    }


@app.get("/models")
async def model_inventory(req: Request):
    """Detailed local model inventory for the UI model diagnostics panel."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=4)
        r.raise_for_status()
        raw_models = r.json().get("models", [])
    except Exception as e:
        return JSONResponse({"error": str(e), "models": []}, 503)

    models = []
    for m in raw_models:
        details = m.get("details") or {}
        size = int(m.get("size") or 0)
        models.append({
            "name": m.get("name") or m.get("model") or "",
            "model": m.get("model") or m.get("name") or "",
            "size": size,
            "size_gb": round(size / (1024 ** 3), 2) if size else 0,
            "modified_at": m.get("modified_at") or "",
            "family": details.get("family") or "",
            "parameter_size": details.get("parameter_size") or "",
            "quantization": details.get("quantization_level") or "",
        })

    models.sort(key=lambda item: item["size"], reverse=True)
    return {
        "models": models,
        "count": len(models),
        "ollama_url": OLLAMA,
        "image_ready": _comfyui_ready(),
        "search_ready": _searx_up(),
    }


def _comfyui_ready() -> bool:
    try:
        r = requests.get(f"{COMFYUI}/system_stats", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─── Agent exec ───────────────────────────────────────────────────────
@app.post("/exec")
async def exec_agent(req: Request):
    """Run the local gemma-agent with a task; stream NDJSON progress + final result.

    Request:  {"task": "restart chia harvester", "model": "auto"|"phi4"|...}
    Response: NDJSON lines — {"progress": <seconds>} ... {"result": "...", "model": "...", "tool_calls": N}
    """
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "chat"):
        return JSONResponse({"error": "rate limit: 20 req/min"}, 429)
    data, err = await _safe_json(req)
    if err: return err
    task  = (data.get("task") or "").strip()
    model = (data.get("model") or "auto").strip()
    if not task:
        return JSONResponse({"error": "task required"}, 400)

    def _run():
        # Lazy import — keeps module-level side effects in agent.py out of server startup
        _agent_dir = "/home/work/gemma-agent"
        if _agent_dir not in sys.path:
            sys.path.insert(0, _agent_dir)
        from agent import run_chain, run_agent  # noqa: PLC0415

        # Capture tool-call count via a simple counting wrapper
        tool_calls = [0]
        _orig_exec = None
        try:
            import agent as _ag
            _orig_exec = _ag.EXECUTORS.copy()
            def _counting_exec(name):
                orig = _orig_exec[name]
                def _wrap(args):
                    tool_calls[0] += 1
                    return orig(args)
                return _wrap
            _ag.EXECUTORS = {k: _counting_exec(k) for k in _orig_exec}
        except Exception:
            pass

        try:
            if model == "auto":
                result, used = run_chain(task, verbose=False)
                return {
                    "result":     result or "Agent returned no output.",
                    "model":      used or "unknown",
                    "tool_calls": tool_calls[0],
                }
            else:
                result = run_agent(task, model=model, verbose=False)
                return {
                    "result":     result or "Agent returned no output.",
                    "model":      model,
                    "tool_calls": tool_calls[0],
                }
        except Exception as e:
            return {"error": str(e)}
        finally:
            # Restore original executors
            try:
                import agent as _ag
                if _orig_exec:
                    _ag.EXECUTORS = _orig_exec
            except Exception:
                pass

    return StreamingResponse(_stream_image_job(_run), media_type="application/x-ndjson")


# ─── Quick status (no AI) ─────────────────────────────────────────────
@app.get("/status")
async def quick_status(req: Request):
    """Fast bot status: reads watchdog JSON + live pgrep for critical procs. No AI."""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    wp = "/home/work/fraqtoos/logs/watchdog_latest.json"
    report = {}
    if os.path.exists(wp):
        try:
            with open(wp) as f:
                report = json.load(f)
        except Exception:
            pass
    snap = report.get("snapshot", {})
    loop = asyncio.get_running_loop()
    live = {}
    for name, proc in [("Orchestrator", "orchestrator.py"), ("WhatsApp", "wa-service")]:
        r = await loop.run_in_executor(None, lambda p=proc: _pgrep_safe(p))
        if r is None:
            live[name] = False
            continue
        live[name] = any(proc in l and "grep" not in l and "watchdog" not in l
                         for l in r.stdout.splitlines())
    try:
        disk_raw = await loop.run_in_executor(
            None, lambda: subprocess.check_output(["df", "-h", "/home/work"], timeout=3).decode().strip())
        disk_out = disk_raw.splitlines()
        disk = disk_out[-1] if disk_out else "?"
        disk_pct = int(disk.split()[4].rstrip("%")) if len(disk.split()) > 4 else 0
    except Exception:
        disk = "?"; disk_pct = 0
    try:
        ram_raw = await loop.run_in_executor(
            None, lambda: subprocess.check_output(["free", "-h"], timeout=3).decode().strip())
        ram_out = ram_raw.splitlines()
        ram = ram_out[1] if len(ram_out) > 1 else "?"
    except Exception:
        ram = "?"
    bots = snap.get("bots", [])
    for b in bots:
        if b["name"] in live:
            b["running"] = live[b["name"]]
    return {
        "timestamp":  snap.get("timestamp", "no report yet"),
        "bots":       bots,
        "live":       live,
        "disk":       disk,
        "disk_pct":   disk_pct,
        "ram":        ram,
        "searxng_up": snap.get("searxng_up"),
        "analysis":   (report.get("analysis", "") or "")[:400],
    }


@app.get("/logs/{service}")
async def tail_service_log(service: str, req: Request, lines: int = 60):
    lines = max(1, min(lines, 200))
    """Tail log lines for a named service. service = orchestrator|chia|portfolio|watcher|fixes|watchdog"""
    ip = req.client.host if req.client else "unknown"
    if not _rate_ok(ip, "conv"):
        return JSONResponse({"error": "rate limit"}, 429)
    service = service.lower().strip()
    if service in ("crypto", "whatsapp", "wa"):
        unit = "whatsapp-crypto-bot" if service == "crypto" else "wa-service"
        try:
            out = subprocess.check_output(
                ["journalctl", "-u", unit, "-n", str(min(lines, 150)), "--no-pager"],
                timeout=5).decode(errors="replace")
            return {"service": service, "lines": out, "source": f"journalctl -u {unit}"}
        except Exception as e:
            return JSONResponse({"error": str(e)}, 500)
    paths = {
        "orchestrator": "/home/work/fraqtoos/logs/fraqtoos.log",
        "fraqtoos":     "/home/work/fraqtoos/logs/fraqtoos.log",
        "portfolio":    "/home/work/portfolio_bot/logs/portfolio.log",
        "chia":         "/home/work/.chia/mainnet/log/debug.log",
        "watcher":      "/home/work/fraqtoos/logs/chia_ai_latest.json",
        "fixes":        "/home/work/fraqtoos/logs/chia_ai_fixes.log",
        "watchdog":     "/home/work/fraqtoos/logs/watchdog_latest.json",
    }
    path = paths.get(service)
    if not path:
        return JSONResponse({"error": f"unknown service '{service}'. Known: {', '.join(paths)}"}, 404)
    if not os.path.exists(path):
        return JSONResponse({"error": f"log not found: {path}"}, 404)
    try:
        out = subprocess.check_output(["tail", f"-{min(lines, 200)}", path], timeout=5).decode(errors="replace")
        return {"service": service, "lines": out, "source": path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


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
    elif model == "juggernaut-xi":
        wf = _build_sdxl_workflow("Juggernaut-XI-v11.safetensors", prompt, negative, steps or 30, width, height)
    elif model == "sd15":
        wf = _build_sd15_workflow("v1-5-pruned-emaonly.safetensors", prompt, negative, steps or 20, width, height)
    else:
        raise ValueError(f"Unknown image model: {model}")

    client_id = str(uuid.uuid4())
    r = requests.post(f"{COMFYUI}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=10)
    resp = r.json()
    if "error" in resp:
        err = resp["error"]; raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
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
        err = resp["error"]; raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
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


# Models whose thinking we suppress for snappy chat (they support think=false).
# qwen3 is NOT here: with think=false it leaks reasoning prose into content —
# with thinking ON its content stays clean and the thinking tokens stream to
# the UI as {"thinking": ...} lines (same as deepseek-r1 / gpt-oss).
_NO_THINK_MODELS = ("gemma4",)


def ollama_stream(model, messages, system="", images=None, temperature=0.7):
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
        "options": {"temperature": temperature, "num_predict": 2000},
    }
    # "think" is a top-level chat param, NOT an option — inside options it is
    # silently ignored (the old bug: qwen3 was never actually skipping thinking).
    if any(model.startswith(m) for m in _NO_THINK_MODELS):
        payload["think"] = False
    try:
        r = requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=300)
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                # Ollama signals failures (bad image, OOM, missing model, …) as an
                # {"error": ...} line. Without this the loop only looked for
                # "message"/"done", so errors were silently dropped and the UI got a
                # blank reply. Surface it instead. error may be a str or nested dict.
                err = d.get("error")
                if err:
                    # Normalize: error can be a plain string, a JSON-encoded string,
                    # or a nested {"error":{"message":...}} dict — pull the message out.
                    if isinstance(err, str):
                        try: err = json.loads(err)
                        except Exception: pass
                    while isinstance(err, dict):
                        nxt = err.get("message") or err.get("error")
                        err = nxt if nxt is not None else json.dumps(err)
                    yield json.dumps({"error": str(err)}) + "\n"
                    break
                msg = d.get("message", {})
                thinking = msg.get("thinking", "")
                if thinking:
                    yield json.dumps({"thinking": thinking}) + "\n"
                token = msg.get("content", "")
                if token:
                    yield json.dumps({"token": token}) + "\n"
                if d.get("done"):
                    # Forward real generation stats so the UI can show true tok/s
                    eval_count = d.get("eval_count") or 0
                    eval_ns    = d.get("eval_duration") or 0
                    tok_s = round(eval_count / (eval_ns / 1e9), 1) if eval_count and eval_ns else 0
                    yield json.dumps({"stats": {
                        "model": model,
                        "prompt_tokens": d.get("prompt_eval_count") or 0,
                        "completion_tokens": eval_count,
                        "tok_s": tok_s,
                    }}) + "\n"
                    break
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


if __name__ == "__main__":
    print("FraqtoOS Chat → http://192.168.2.108:8080")
    print(f"Images: ComfyUI on {COMFYUI}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
