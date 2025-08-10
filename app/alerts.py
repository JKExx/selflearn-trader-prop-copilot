# app/alerts.py
from __future__ import annotations
import json, requests

def send_slack(webhook_url: str, text: str) -> bool:
    if not webhook_url: return False
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code in (200,204)
    except Exception:
        return False

def send_discord(webhook_url: str, content: str) -> bool:
    if not webhook_url: return False
    try:
        r = requests.post(webhook_url, data={"content": content}, timeout=10)
        return 200 <= r.status_code < 300
    except Exception:
        return False
