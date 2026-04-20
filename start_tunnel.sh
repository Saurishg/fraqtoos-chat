#!/bin/bash
# Start cloudflared tunnel. A separate watcher sends the URL to WhatsApp.

LOGFILE="/home/work/fraqtoos-chat/tunnel.log"
> "$LOGFILE"

exec /usr/local/bin/cloudflared tunnel --url http://localhost:8080 --no-autoupdate 2>&1 | \
  tee -a "$LOGFILE" | \
  grep --line-buffered "trycloudflare.com" | \
  while IFS= read -r line; do
    URL=$(echo "$line" | grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com')
    [ -n "$URL" ] && /usr/bin/python3 /home/work/fraqtoos-chat/notify_url.py "$URL" &
  done
