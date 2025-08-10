from __future__ import annotations

import requests

def send_slack(webhook_url: str, text: str) -> bool:
    if not webhook_url:
        return False
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False

def send_discord(webhook_url: str, content: str) -> bool:
    if not webhook_url:
        return False
    try:
        r = requests.post(webhook_url, data={"content": content}, timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False
