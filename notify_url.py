#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/work/fraqtoos")
from core.notifier import send

args = sys.argv[1:]
chat_url    = args[0] if len(args) > 0 else ""
comfyui_url = args[1] if len(args) > 1 else ""
obs_url     = args[2] if len(args) > 2 else ""

lines = ["FraqtoOS is live!\n"]
if chat_url:
    lines.append(f"Chat (phi4/gemma4/Claude):\n{chat_url}")
if comfyui_url:
    lines.append(f"ComfyUI (FLUX images):\n{comfyui_url}")
if obs_url:
    lines.append(f"Obsidian (notes):\n{obs_url}")

msg = "\n\n".join(lines)

NUMBERS = [None, "+919821777908"]
for phone in NUMBERS:
    ok = send(msg, phone=phone) if phone else send(msg)
    print(f"WhatsApp {'sent' if ok else 'FAILED'}: {phone or 'default'}")
