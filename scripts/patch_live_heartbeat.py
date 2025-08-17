from __future__ import annotations

import pathlib
import re

P = pathlib.Path("app/ui/st_app.py")
s = P.read_text()

# (A) Sidebar version chip (once), right after "import streamlit as st"
if "st.sidebar.caption(" not in s:
    s = re.sub(
        r"(^\s*import\s+streamlit\s+as\s+st\s*$)",
        r"\1\nimport os\nst.sidebar.caption(f\"Prop Copilot {os.getenv('APP_VERSION','dev')}\")",
        s,
        count=1,
        flags=re.M,
    )

# (B) Heartbeat helpers at EOF (once). NOTE: no outer f-string.
HB_MARK = "# --- helpers: heartbeat (auto-added) ---"
if HB_MARK not in s:
    s += """
\n# --- helpers: heartbeat (auto-added) ---
import pandas as pd
from dataclasses import dataclass

@dataclass
class _LiveStatus:
    last_bar_utc: pd.Timestamp | None
    minutes_since_bar: float | None
    fetch_ms: int | None
    ok: bool
    reason: str

def _heartbeat(fetch_fn, symbol: str, interval: str) -> _LiveStatus:
    import time
    t0 = time.perf_counter()
    try:
        df = fetch_fn(symbol=symbol, interval=interval, count=3, complete_only=True)
        ms = int((time.perf_counter() - t0) * 1000)
        last = pd.to_datetime(df.index[-1]).tz_convert("UTC") if len(df) else None
        mins = (pd.Timestamp.utcnow().tz_localize("UTC") - last).total_seconds()/60.0 if last is not None else None
        ok = mins is not None and mins < 120  # 1–2 bars stale tolerance for 1h
        return _LiveStatus(last, mins, ms, ok, "")
    except Exception as e:
        return _LiveStatus(None, None, None, False, f"{type(e).__name__}: {e}")
"""

# (C) Insert heartbeat UI under the Go Live header (once)
anchor = 'st.subheader("Go Live (paper signals for Copiix)")'
if anchor in s and "Heartbeat" not in s:
    insert_at = s.index(anchor) + len(anchor)
    snippet = """
\n        # Heartbeat (data freshness + fetch latency)
        try:
            _fetch = get_ohlcv_oanda if source == "OANDA" else get_ohlcv_yf
            _sym = _to_oanda_instrument(symbol) if source == "OANDA" else symbol
            _hb = _heartbeat(_fetch, _sym, interval)
            _chip = "🟢" if _hb.ok else ("🟠" if not _hb.reason else "🔴")
            st.caption(
                f"Heartbeat {_chip} • last bar (UTC)={_hb.last_bar_utc} • "
                f"+{(_hb.minutes_since_bar or 0):.1f} min • fetch={_hb.fetch_ms or 0} ms"
            )
            if _hb.reason:
                st.warning(f"Data fetch issue: {_hb.reason}")
        except Exception as _e:
            st.warning(f"Heartbeat error: {_e}")
"""
    s = s[:insert_at] + snippet + s[insert_at:]

P.write_text(s)
print("Patched st_app.py ✅")
