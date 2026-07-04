#!/usr/bin/env python3
# Sends the boot "FraqtoOS is live" WhatsApp. Args are the raw tunnel URLs
# (passed by start_tunnel.sh) but only their PRESENCE is used — the message
# always shows the permanent /go/ links, which never change across restarts.
import sys
sys.path.insert(0, "/home/work/fraqtoos")
from core.notifier import send

BASE = "https://saurishg.github.io/links-page"

args = sys.argv[1:]
chat_up      = len(args) > 0 and args[0]
comfyui_up   = len(args) > 1 and args[1]
grafana_up   = len(args) > 2 and args[2]
dashboard_up = len(args) > 3 and args[3]

lines = ["*FraqtoOS is live!*\n"]
if chat_up:
    lines.append(f"💬 *Chat* (phi4/gemma4/Claude)\n{BASE}/go/chat/")
if comfyui_up:
    lines.append(f"🎨 *ComfyUI* (FLUX images)\n{BASE}/go/comfyui/")
if dashboard_up:
    lines.append(f"📊 *Creator Dashboard*\n{BASE}/go/dashboard/")
if grafana_up:
    lines.append(f"🖥 *Grafana* (login: admin)\n{BASE}/go/grafana/")

lines.append(f"🔗 *All services:* {BASE}/")
lines.append("\n_These links are permanent — safe to bookmark_")
msg = "\n\n".join(lines)

ok = send(msg)
print(f"WhatsApp {'sent' if ok else 'FAILED'}")
