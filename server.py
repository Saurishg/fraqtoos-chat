#!/usr/bin/env python3
"""
FraqtoOS Chat — Tailscale chatbot.
Access: http://100.67.1.60:8080
"""
import json, os, requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

load_dotenv("/home/work/fraqtoos-chat/.env")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA        = "http://localhost:11434"
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

    if model in CLAUDE_MODELS:
        return StreamingResponse(claude_stream(model, messages), media_type="text/plain")
    else:
        return StreamingResponse(ollama_stream(model, messages), media_type="text/plain")


def ollama_stream(model, messages):
    prompt = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        prompt += f"{role}: {m['content']}\n"
    prompt += "Assistant:"
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={
            "model": model, "prompt": prompt, "stream": True,
            "options": {"temperature": 0.7, "num_predict": 1200}
        }, stream=True, timeout=300)
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                token = d.get("response", "")
                if token:
                    yield json.dumps({"token": token}) + "\n"
                if d.get("done"):
                    break
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


def claude_stream(model, messages):
    if not ANTHROPIC_KEY:
        yield json.dumps({"error": "Add ANTHROPIC_API_KEY to /home/work/fraqtoos-chat/.env then restart service"}) + "\n"
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        with client.messages.stream(model=model, max_tokens=2048, messages=messages) as stream:
            for text in stream.text_stream:
                yield json.dumps({"token": text}) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"


@app.get("/health")
async def health():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
    except:
        models = []
    return {"status": "ok", "ollama_models": models, "claude_ready": bool(ANTHROPIC_KEY)}


if __name__ == "__main__":
    print(f"FraqtoOS Chat → http://100.67.1.60:8080")
    print(f"Claude: {'✓ API key loaded' if ANTHROPIC_KEY else '✗ No key — edit .env'}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
