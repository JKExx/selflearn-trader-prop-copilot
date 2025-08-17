import json
from pathlib import Path

STATE_DIR = Path(".state")
STATE_FILE = STATE_DIR / "live.json"

def load_state() -> dict:
    STATE_DIR.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_state(d: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(d, indent=2, default=str))
