# FraqtoOS Chat — Architecture

## Purpose

Single-host local AI chat for the work server. A FastAPI app (`server.py`,
~2025 lines) serves a vanilla-JS SPA (`static/index.html`, ~3794 lines) and
proxies to **Ollama** (chat / classify / embed), **ComfyUI** (image + video),
**SearXNG** (web search), an optional **Anthropic API**, and the local
**gemma-agent** (invoked via `/exec`). It also persists per-user **memory**,
named **conversations**, **prompt templates**, and a **semantic-search
embedding cache**, and exposes server-status bridge endpoints (watchdog,
digest, BTC, GPU, disk, journal logs) so the chat UI doubles as a server
console.

## Topology

```
              ┌─────────────────────────────────────────┐
   browser ── │  Cloudflare Tunnel  /  Tailscale (LAN)  │
              └────────────┬────────────────────────────┘
                           │
                           ▼  http://192.168.2.108:8080
              ┌─────────────────────────────────────────┐
              │  FastAPI  (server.py, uvicorn)          │
              │  routes: /chat /imagine /wan-* /memory  │
              │          /conversations /exec /status   │
              └────────────┬────────────────────────────┘
                           │
       ┌──────────┬────────┼─────────┬──────────────┬─────────────┐
       ▼          ▼        ▼         ▼              ▼             ▼
   Ollama    ComfyUI    SearXNG  Anthropic API  gemma-agent   filesystem
  :11434      :8188    (core.    (claude-*)     (Python module)  conversations/
  qwen3,    FLUX/SDXL/  web_      via          /home/work/      memory.json
  phi4,     WAN/Kontext search)   anthropic     gemma-agent     conv_embeddings.json
  deepseek, /Champ/...            SDK
  nomic-
  embed-text
```

`start_tunnel.sh` launches two `cloudflared` quick-tunnels (chat 8080 +
ComfyUI 8188), captures their `*.trycloudflare.com` URLs, watches a third
service unit (`cloudflared-grafana`) for Grafana, then runs `notify_url.py`
to push the URLs out (WhatsApp). Tailscale is the always-on LAN path.

## Server modules (server.py)

| Range          | Group                                                                |
|----------------|----------------------------------------------------------------------|
| 1–78           | Imports, env, FastAPI app, rate-limit buckets, locks, helpers        |
| 80–101         | `/`, vision-model probe                                              |
| 104–172        | `_trim_history` + `/chat` (streaming Ollama or Claude)               |
| 175–243        | Smart auto-routing (`ROUTING_TARGETS`, `/classify`)                  |
| 245–299        | PWA: `/manifest.json`, `/service-worker.js`                          |
| 302–401        | Image gen: `_stream_image_job`, `/imagine`, models, status, `/suggest` |
| 403–512        | Avatar (PuLID-FLUX): `/avatar`                                       |
| 513–771        | Video: mimic-motion, animate-anyone, Champ                           |
| 770–924        | WAN text→video and image→video (`/wan-video`, `/wan-i2v`)            |
| 925–953        | `/edit-image` (FLUX Kontext)                                         |
| 954–972        | `/search` (SearXNG bridge)                                           |
| 975–1010       | `/upload` (text / pdf / image)                                       |
| 1012–1150      | Bot bridge: `/bridge/{cmd}` → watchdog, digest, bots, btc, portfolio |
| 1152–1202      | Semantic-search helpers (nomic-embed-text)                           |
| 1206–1318      | `/conversations/search/q`, `/conversations/reindex`                  |
| 1320–1368      | `/conversations/{id}/autotitle` (phi4)                               |
| 1371–1463      | `/conversations` CRUD (`_conv_path`, list, get, save)                |
| 1466–1577      | Memory: load/save, `/memory` GET/POST/DELETE, `/memory/extract`      |
| 1580–1597      | `/conversations/{id}` DELETE (also evicts embed cache entry)         |
| 1600–1672      | `/chia-harvester`, `/gpu`, `/health`                                 |
| 1675–1750      | `/exec` — gemma-agent runner (NDJSON progress + result)              |
| 1754–1846      | `/status`, `/logs/{service}`                                         |
| 1848–1971      | ComfyUI workflow builders + `_generate` / `_edit_image`              |
| 1973–2018      | `ollama_stream`, `claude_stream` (NDJSON token producers)            |
| 2021–end       | `__main__` — uvicorn boot                                            |

## State / data files

All persistent state lives in the project dir (not a dotdir):

- `conversations/<id>.json` — `{id, title, model, history, created, updated}`,
  written via `tmp + os.replace` under `_conv_lock` (server.py:1448).
- `memory.json` — list of `{id, fact, ts}`; written under `_memory_lock`
  (server.py:1515,1528).
- `conv_embeddings.json` — `{conv_id: {emb:[768f], updated}}`, nomic-embed-text
  cache; written under `_embed_cache_lock` (server.py:1250,1315,1591).
- Prompt library lives client-side in `localStorage`; `/suggest` gets prompt
  ideas from phi4 on demand.
- Locks (server.py:42–44) are `asyncio.Lock`, NOT cross-process — single
  uvicorn worker is required.

## External dependencies

- **Ollama** — `http://localhost:11434`, models: `qwen3:14b`, `phi4`,
  `deepseek-r1:14b`, `gpt-oss:20b`, `gemma4`, `llava:*` (vision auto-detected),
  `nomic-embed-text` (embeddings).
- **ComfyUI** — `http://localhost:8188`, workflows constructed inline
  (`_build_flux_workflow`, `_build_sdxl_workflow`, `_build_kontext_workflow`,
  WAN / Champ / mimic-motion / animate-anyone JSON graphs).
- **SearXNG** — imported from `/home/work/fraqtoos` (`core.web_search`); soft
  fallback to no-op lambdas if import fails (server.py:17–21).
- **Anthropic API** — optional, `ANTHROPIC_API_KEY` in `.env`; gates
  `claude_stream` (server.py:2004).
- **gemma-agent** — `/home/work/gemma-agent/agent.py`, lazy-imported inside
  `/exec`'s thread runner; `EXECUTORS` is wrapped to count tool calls
  (server.py:1706–1748).

## UI structure (static/index.html)

| Range     | Section                                                       |
|-----------|---------------------------------------------------------------|
| 1–12      | `<head>` start, highlight.js CDN                              |
| 13–663    | `<style>` — theme vars, layout, modals, accessibility         |
| 665–750   | Sidebar: logo, model picker, conv list, action buttons        |
| 752–752   | Sidebar collapse toggle                                       |
| 754–902   | `#main`: topbar, sysprompt bar, chat-search bar, `#chat`,     |
|           | `#inputwrap` (textarea, attach row, neg-prompt, tool buttons, |
|           | image controls, send button), exec shortcuts, quick actions   |
| 904–906   | Lightbox                                                      |
| 908–918   | Prompt library modal                                          |
| 920–926   | Starred messages modal                                        |
| 927–929   | Drop overlay                                                  |
| 931–1073  | Features-guide modal                                          |
| 1075–1092 | Memory modal                                                  |
| 1093–1100 | Command palette                                               |
| 1101–1103 | Toast container                                               |
| 1104–3792 | `<script>` — state, markdown render, send/stream loops,       |
|           | conv list grouping (line ~2449), slash-command menu, voice,   |
|           | image / video send paths, modal logic                         |

Chat content uses `.chat-inner { max-width: min(820px, 72ch); }`
(index.html:246) — bumped to `880px / 74ch` ≥1600px viewport. The
"1080-wide" claim refers to the whole `#main` column, not `.chat-inner`.

## Streaming protocol

Server returns `application/x-ndjson` (or `text/plain` for the chat route);
each line is one JSON object:

```
{"token":"hello"}      ← chat tokens (ollama_stream / claude_stream)
{"progress": 5}        ← every 5s while a long job runs (_stream_image_job)
{"image":"data:..."}   ← terminal payload for /imagine
{"result":"…","model":"…","tool_calls":N}   ← terminal for /exec
{"error":"…"}          ← any failure, terminal
```

Client (index.html:1832, 1897, 3315): `reader.read()` →
`buf += decoder.decode(value, {stream:true})` → split on `\n`, keep last
fragment in `buf`, `JSON.parse` each line. After loop: flush
`buf + decoder.decode()` to catch a trailing line with no newline.

## Beta / recent additions

- **Semantic search** (server.py:1152–1290) via nomic-embed-text with on-write
  cache + `/conversations/reindex` rebuild; falls back to substring search
  when the model is missing.
- **`/exec` gemma-agent route** (server.py:1684) — auto model-chain or pinned
  model, NDJSON progress, tool-call counting wrapper.
- **Date-grouped conversation list** — Today / Yesterday / Last 7 days /
  Last 30 days / Older (index.html:~2449).
- **Accessibility** — `role="log"` + `aria-live="polite"` on `#chat`,
  `aria-live="assertive"` on input errors, `aria-pressed` on toggles,
  `:focus-visible` rings (index.html:591), `prefers-reduced-motion` honored
  (index.html:598).
- **Wider main column** — `#main` widened (≈1080px feel) while chat text
  stays at ~72ch for readability.
- **Auto-routing** — phi4 classifies prompt, server picks qwen3 / deepseek /
  phi4 (server.py:175).

## Operational

- Service: `fraqtoos-chat.service` — `python3 /home/work/fraqtoos-chat/server.py`,
  `User=work`, `Restart=always`, ordered after `ollama.service`.
- `systemctl status fraqtoos-chat` · `sudo systemctl restart fraqtoos-chat`
  (sudo password 0000).
- `journalctl -u fraqtoos-chat -f` for live logs.
- LAN: `http://192.168.2.108:8080`. Public: Cloudflare quick-tunnel via
  `start_tunnel.sh` (logs in `tunnel_chat.log`, `tunnel_comfyui.log`).
- ComfyUI must be up on :8188 for `/imagine`, `/edit-image`, `/wan-*`,
  `/avatar`, `/champ`, mimic / animate routes.

## Known gotchas

- **JSON race on conv save** — fixed by `_conv_lock` + tmp-file +
  `os.replace`; autotitle re-reads inside the same lock so it can't clobber a
  concurrent `POST /conversations` (server.py:1356).
- **`asyncio.Lock` is in-process** — running multiple uvicorn workers would
  reintroduce the race. Keep it single-worker.
- **`npx claude-flow` cache vs Node v20** — claude-flow's npm cache can
  desync against the system Node v20 install; clear `~/.npm/_npx/` if the
  agent CLI errors on import.
- **Cloudflare tunnel notify** — `notify_url.py` only fires once per
  `start_tunnel.sh` run; if a tunnel auto-reconnects with a new URL the
  WhatsApp ping is not re-sent. Re-run the script.
- **`sys.path.insert` at import time** — `core.web_search` import is from
  `/home/work/fraqtoos`; if that path moves, search silently no-ops
  (server.py:17–21 swallows the ImportError).
- **Vision model auto-pick** — `_has_vision_model` only looks for a fixed
  list (server.py:85). New vision tags (`qwen2.5-vl`, etc.) won't be
  detected until added.

## New issue spotted

`_has_vision_model` (server.py:89) calls `r.json()` twice on the same
`Response` and rebuilds `installed` from both calls; the second call works
fine but the first comprehension only stores the bare model name (split on
`:`), which means a tag like `llava:13b` matches only when the loop's
fallback `v.split(":")[0] in installed` branch hits — minor, not a bug, but
the double `r.json()` call is wasteful. Cap: 1 nit, no fix applied per
instructions.
