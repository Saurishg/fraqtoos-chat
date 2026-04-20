#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/work/fraqtoos")
from core.notifier import send

url = sys.argv[1] if len(sys.argv) > 1 else "unknown"
msg = f"FraqtoOS Chat is live!\n\n{url}\n\nModels: phi4 / gemma4 / qwen3 / llama4 / Claude"

NUMBERS = [None, "+919821777908"]  # None = default number from notifier
for phone in NUMBERS:
    ok = send(msg, phone=phone) if phone else send(msg)
    print(f"WhatsApp {'sent' if ok else 'FAILED'}: {phone or 'default'}")
