#!/bin/bash
# Start 2 Cloudflare quick tunnels: Chat, ComfyUI
# Waits for both URLs then sends one WhatsApp message

LOGDIR="/home/work/fraqtoos-chat"
CF="/usr/local/bin/cloudflared"

> "$LOGDIR/tunnel_chat.log"
> "$LOGDIR/tunnel_comfyui.log"
rm -f /tmp/cf_url_chat /tmp/cf_url_comfyui

# Start a tunnel and write its URL to a file when detected
start_tunnel() {
  local port="$1" urlfile="$2" logfile="$3"
  $CF tunnel --url "http://localhost:$port" --no-autoupdate 2>&1 | \
    tee -a "$logfile" | \
    grep --line-buffered "trycloudflare.com" | \
    while IFS= read -r line; do
      URL=$(echo "$line" | grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com')
      [ -n "$URL" ] && echo "$URL" > "$urlfile"
    done &
}

start_tunnel 8080 /tmp/cf_url_chat    "$LOGDIR/tunnel_chat.log"
start_tunnel 8188 /tmp/cf_url_comfyui "$LOGDIR/tunnel_comfyui.log"

echo "Tunnels started. Waiting for URLs (up to 90s)..."

for i in $(seq 1 90); do
  sleep 1
  CHAT=$(cat /tmp/cf_url_chat 2>/dev/null)
  COMFY=$(cat /tmp/cf_url_comfyui 2>/dev/null)
  [ -n "$CHAT" ] && [ -n "$COMFY" ] && break
done

echo "Chat:    ${CHAT:-not found}"
echo "ComfyUI: ${COMFY:-not found}"

/usr/bin/python3 /home/work/fraqtoos-chat/notify_url.py "${CHAT:-}" "${COMFY:-}" &

wait
