#!/bin/bash
# Start Cloudflare quick tunnels: Chat, ComfyUI, Creator Dashboard, AtInUs
# Also captures Grafana + Obsidian tunnel URLs from their systemd services
# Waits for all URLs then sends one WhatsApp message

LOGDIR="/home/work/fraqtoos-chat"
CF="/usr/local/bin/cloudflared"

> "$LOGDIR/tunnel_chat.log"
> "$LOGDIR/tunnel_comfyui.log"
> "$LOGDIR/tunnel_dashboard.log"
> "$LOGDIR/tunnel_atinus.log"
> "$LOGDIR/tunnel_comfyui_rocm.log"
rm -f /tmp/cf_url_chat /tmp/cf_url_comfyui /tmp/cf_url_grafana /tmp/cf_url_dashboard /tmp/cf_url_obsidian /tmp/cf_url_atinus /tmp/cf_url_comfyui_rocm

# ── Start the AtInUs static server (port 8765) ─────────────────────────────
ATINUS_DIR="/home/work/atinus-website"
ATINUS_PORT=8765
ATINUS_LOG="$LOGDIR/atinus_server.log"

# Kill any existing process on the port before starting fresh
fuser -k -n tcp $ATINUS_PORT 2>/dev/null
sleep 1
if [ -d "$ATINUS_DIR" ]; then
  ( cd "$ATINUS_DIR" && /usr/bin/python3 -m http.server $ATINUS_PORT >> "$ATINUS_LOG" 2>&1 ) &
  echo "AtInUs static server started on port $ATINUS_PORT (PID $!)"
fi

# Start a tunnel and write its URL to a file when detected
start_tunnel() {
  local port="$1" urlfile="$2" logfile="$3"
  $CF tunnel --url "http://localhost:$port" --no-autoupdate 2>&1 | \
    tee -a "$logfile" | \
    grep --line-buffered "trycloudflare.com" | \
    while IFS= read -r line; do
      URL=$(echo "$line" | grep -oP 'https://(?!api\.)[a-z0-9\-]+\.trycloudflare\.com')
      [ -n "$URL" ] && echo "$URL" > "$urlfile"
    done &
}

# Start a tunnel to a remote URL (not localhost)
start_tunnel_url() {
  local target="$1" urlfile="$2" logfile="$3"
  $CF tunnel --url "$target" --no-autoupdate 2>&1 | \
    tee -a "$logfile" | \
    grep --line-buffered "trycloudflare.com" | \
    while IFS= read -r line; do
      URL=$(echo "$line" | grep -oP 'https://(?!api\.)[a-z0-9\-]+\.trycloudflare\.com')
      [ -n "$URL" ] && echo "$URL" > "$urlfile"
    done &
}

# Watch journalctl for a systemd-managed tunnel URL
watch_service_tunnel() {
  local service="$1" urlfile="$2"
  journalctl -u "$service" -f --no-pager 2>/dev/null | \
    grep --line-buffered "trycloudflare.com" | \
    while IFS= read -r line; do
      URL=$(echo "$line" | grep -oP 'https://(?!api\.)[a-z0-9\-]+\.trycloudflare\.com')
      [ -n "$URL" ] && echo "$URL" > "$urlfile"
    done &
}

start_tunnel 8080 /tmp/cf_url_chat      "$LOGDIR/tunnel_chat.log"
start_tunnel 8188 /tmp/cf_url_comfyui   "$LOGDIR/tunnel_comfyui.log"
start_tunnel 3000 /tmp/cf_url_dashboard "$LOGDIR/tunnel_dashboard.log"
start_tunnel $ATINUS_PORT /tmp/cf_url_atinus "$LOGDIR/tunnel_atinus.log"
start_tunnel 8189 /tmp/cf_url_comfyui_rocm "$LOGDIR/tunnel_comfyui_rocm.log"
start_tunnel_url "http://192.168.2.103" /tmp/cf_url_ipmi "$LOGDIR/tunnel_ipmi.log"
watch_service_tunnel cloudflared-grafana  /tmp/cf_url_grafana
watch_service_tunnel cloudflared-obsidian /tmp/cf_url_obsidian

echo "Tunnels started. Waiting for URLs (up to 90s)..."

for i in $(seq 1 90); do
  sleep 1
  CHAT=$(cat /tmp/cf_url_chat 2>/dev/null)
  COMFY=$(cat /tmp/cf_url_comfyui 2>/dev/null)
  DASHBOARD=$(cat /tmp/cf_url_dashboard 2>/dev/null)
  GRAFANA=$(cat /tmp/cf_url_grafana 2>/dev/null)
  OBSIDIAN=$(cat /tmp/cf_url_obsidian 2>/dev/null)
  IPMI=$(cat /tmp/cf_url_ipmi 2>/dev/null)
  ATINUS=$(cat /tmp/cf_url_atinus 2>/dev/null)
  [ -n "$CHAT" ] && [ -n "$COMFY" ] && [ -n "$DASHBOARD" ] && [ -n "$ATINUS" ] && break
done

echo "Chat:      ${CHAT:-not found}"
echo "ComfyUI:   ${COMFY:-not found}"
echo "Dashboard: ${DASHBOARD:-not found}"
echo "Grafana:   ${GRAFANA:-not found}"
echo "Obsidian:  ${OBSIDIAN:-not found}"
echo "IPMI:      ${IPMI:-not found}"
echo "AtInUs:    ${ATINUS:-not found}"

/usr/bin/python3 /home/work/fraqtoos-chat/notify_url.py "${CHAT:-}" "${COMFY:-}" "${GRAFANA:-}" "${DASHBOARD:-}" &

# Auto-update saurishg.github.io/links-page with new tunnel URLs
/home/work/links-page/update_urls.sh >> /home/work/links-page/update.log 2>&1 &

wait
