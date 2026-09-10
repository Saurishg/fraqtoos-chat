"""Regression check for the server-rail /tunnels panel.

The panel read 0/4 forever after the cloudflared quick tunnels were replaced by
nginx vhosts on 2026-09-05. It must report the real public sites, and the
service worker must never cache it (cache-first froze it on installed PWAs).

Run:  python3.12 scripts/test_tunnels.py   (against the live service on :8080)
"""
import json, re, sys, urllib.request

BASE = "http://127.0.0.1:8080"
EXPECTED = {"chat", "dashboard", "grafana", "obsidian", "ntfy"}
fails = []

t = json.load(urllib.request.urlopen(BASE + "/tunnels", timeout=30))
names = {x["name"] for x in t["tunnels"]}
if names != EXPECTED:
    fails.append(f"sites {sorted(names)} != {sorted(EXPECTED)}")
down = [f'{x["name"]}: {x["protocol"]}' for x in t["tunnels"] if not x["active"]]
if down or t["up"] != t["total"]:
    fails.append(f"up {t['up']}/{t['total']}, down: {down}")
for x in t["tunnels"]:
    if not x["url"].startswith("https://") or "trycloudflare" in x["url"]:
        fails.append(f'{x["name"]} url {x["url"]!r}')

sw = urllib.request.urlopen(BASE + "/service-worker.js", timeout=10).read().decode()
prefixes = re.search(r"API_PREFIXES = \[(.*?)\];", sw, re.S).group(1)
for p in ("/tunnels", "/voice/status"):
    if f"'{p}'" not in prefixes:
        fails.append(f"service worker would cache {p}")

cc = urllib.request.urlopen(BASE + "/", timeout=10).headers.get("Cache-Control", "")
if "no-cache" not in cc:
    fails.append(f"/ sent without no-cache ({cc!r}) - browsers keep a stale UI for days")

print("FAIL:\n  " + "\n  ".join(fails) if fails else f"PASS: {t['up']}/{t['total']} sites up")
sys.exit(1 if fails else 0)
