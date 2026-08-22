#!/bin/bash
# Start Cloudflare quick tunnels: Chat, ComfyUI, Creator Dashboard
# Also captures Grafana + Obsidian tunnel URLs from their systemd services
# Waits for all URLs then sends one WhatsApp message

# 2026-07-30 SECURITY: the tunnels below were disabled. Each published an
# unauthenticated service to the public internet:
#   8188/8189 ComfyUI  -> no auth, custom nodes can execute code (RCE)
#   192.168.0.103 IPMI -> out-of-band server control
#   8501 Streamlit     -> no auth
# Re-enable only behind Cloudflare Access or another auth layer.
LOGDIR="/home/work/fraqtoos-chat"
CF="/usr/local/bin/cloudflared"

> "$LOGDIR/tunnel_chat.log"
> "$LOGDIR/tunnel_comfyui.log"
> "$LOGDIR/tunnel_dashboard.log"
> "$LOGDIR/tunnel_comfyui_rocm.log"
> "$LOGDIR/tunnel_ipmi.log"
rm -f /tmp/cf_url_chat /tmp/cf_url_comfyui /tmp/cf_url_grafana /tmp/cf_url_dashboard /tmp/cf_url_obsidian /tmp/cf_url_comfyui_rocm /tmp/cf_url_ipmi

# AtInUs (port 8765) removed entirely 2026-08-09 — project deleted.

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
  # Seed from journal history first — the running service registered its URL long ago,
  # and `journalctl -f` alone only replays the last 10 lines
  local seed
  seed=$(journalctl -u "$service" --no-pager -n 5000 2>/dev/null | \
    grep -oP 'https://(?!api\.)[a-z0-9\-]+\.trycloudflare\.com' | tail -1)
  [ -n "$seed" ] && echo "$seed" > "$urlfile"
  journalctl -u "$service" -f --no-pager 2>/dev/null | \
    grep --line-buffered "trycloudflare.com" | \
    while IFS= read -r line; do
      URL=$(echo "$line" | grep -oP 'https://(?!api\.)[a-z0-9\-]+\.trycloudflare\.com')
      [ -n "$URL" ] && echo "$URL" > "$urlfile"
    done &
}

# 2026-08-22: chat and dashboard are now real systemd units
# (cloudflared-chat.service / cloudflared-dashboard.service) instead of
# background children of this script. Starting them here too would run a
# SECOND tunnel per service — two URLs for one origin, one of which the
# links page would never publish. Watch the units instead.
# start_tunnel 8080 /tmp/cf_url_chat      "$LOGDIR/tunnel_chat.log"
# start_tunnel 8188 /tmp/cf_url_comfyui   "$LOGDIR/tunnel_comfyui.log"   # disabled 2026-07-30 (unauth RCE)
# start_tunnel 3000 /tmp/cf_url_dashboard "$LOGDIR/tunnel_dashboard.log"
watch_service_tunnel cloudflared-chat      /tmp/cf_url_chat
watch_service_tunnel cloudflared-dashboard /tmp/cf_url_dashboard
# start_tunnel 8189 /tmp/cf_url_comfyui_rocm "$LOGDIR/tunnel_comfyui_rocm.log"   # disabled 2026-07-30 (unauth RCE)
# start_tunnel_url "http://192.168.0.103" /tmp/cf_url_ipmi "$LOGDIR/tunnel_ipmi.log"   # disabled 2026-07-30 (IPMI = full server control)
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
  [ -n "$CHAT" ] && [ -n "$DASHBOARD" ] && break   # COMFY tunnel disabled 2026-07-30
done

echo "Chat:      ${CHAT:-not found}"
echo "ComfyUI:   ${COMFY:-not found}"
echo "Dashboard: ${DASHBOARD:-not found}"
echo "Grafana:   ${GRAFANA:-not found}"
echo "Obsidian:  ${OBSIDIAN:-not found}"
echo "IPMI:      ${IPMI:-not found}"

/usr/bin/python3 /home/work/fraqtoos-chat/notify_url.py "${CHAT:-}" "${COMFY:-}" "${GRAFANA:-}" "${DASHBOARD:-}" &

# Auto-update saurishg.github.io/links-page with new tunnel URLs
/home/work/links-page/update_urls.sh >> /home/work/links-page/update.log 2>&1 &

wait
