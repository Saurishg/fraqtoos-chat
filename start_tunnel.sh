#!/bin/bash
# Start 3 Cloudflare quick tunnels: Chat, ComfyUI, Obsidian
# Waits for all 3 URLs then sends one WhatsApp message

LOGDIR="/home/work/fraqtoos-chat"
CF="/usr/local/bin/cloudflared"

> "$LOGDIR/tunnel_chat.log"
> "$LOGDIR/tunnel_comfyui.log"
> "$LOGDIR/tunnel_obsidian.log"
rm -f /tmp/cf_url_chat /tmp/cf_url_comfyui /tmp/cf_url_obsidian

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

start_tunnel 8080 /tmp/cf_url_chat     "$LOGDIR/tunnel_chat.log"
start_tunnel 8188 /tmp/cf_url_comfyui  "$LOGDIR/tunnel_comfyui.log"
start_tunnel 6080 /tmp/cf_url_obsidian "$LOGDIR/tunnel_obsidian.log"

echo "Tunnels started. Waiting for URLs (up to 90s)..."

# Wait up to 90s for all 3 URLs
for i in $(seq 1 90); do
  sleep 1
  CHAT=$(cat /tmp/cf_url_chat 2>/dev/null)
  COMFY=$(cat /tmp/cf_url_comfyui 2>/dev/null)
  OBS=$(cat /tmp/cf_url_obsidian 2>/dev/null)
  [ -n "$CHAT" ] && [ -n "$COMFY" ] && [ -n "$OBS" ] && break
done

echo "Chat:     ${CHAT:-not found}"
echo "ComfyUI:  ${COMFY:-not found}"
echo "Obsidian: ${OBS:-not found}"

# Send one WhatsApp with all URLs
/usr/bin/python3 /home/work/fraqtoos-chat/notify_url.py \
  "${CHAT:-}" "${COMFY:-}" "${OBS:-}" &

# Keep alive — tunnels are background subshells of this process
wait
