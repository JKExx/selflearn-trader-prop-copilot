# app/dataio/oanda.py
from __future__ import annotations

from typing import Optional
import os
import math
import requests
import pandas as pd

# --- helpers -----------------------------------------------------------------

def _env_domain() -> str:
    env = (os.environ.get("OANDA_ENV") or "PRACTICE").upper()
    # PRACTICE = demo, FXTRADE = live
    return "api-fxtrade.oanda.com" if env in {"LIVE", "FXTRADE"} else "api-fxpractice.oanda.com"

def _auth_headers() -> dict:
    token = os.environ.get("OANDA_TOKEN") or ""
    if not token:
        raise RuntimeError("OANDA_TOKEN not set (put it in ~/.selflearntrader/secrets.toml or env).")
    return {"Authorization": f"Bearer {token}"}

def _granularity(interval: str) -> str:
    m = interval.lower()
    if m in {"1h", "h1"}:
        return "H1"
    if m in {"15m", "m15"}:
        return "M15"
    if m in {"30m", "m30"}:
        return "M30"
    if m in {"1d", "d1", "d"}:
        return "D"
    raise ValueError(f"Unsupported interval for OANDA: {interval}")

def _to_iso_utc(s: str) -> str:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    # OANDA likes RFC3339 with Z
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def _only_complete(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only completed candles (no partial current forming bar)
    if "complete" in df.columns:
        df = df[df["complete"] == True]  # noqa: E712
        df = df.drop(columns=["complete"])
    return df

def _map_symbol(sym: str) -> str:
    s = (sym or "").upper().replace("/", "_")
    # handle Yahoo-style metals: XAUUSD=X -> XAU_USD
    if s.endswith("=X"):
        s = s[:-2]
    if "_" in s:
        return s
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s

# Exposed for UI import
def _to_oanda_instrument(symbol: str) -> str:
    return _map_symbol(symbol)

# --- main fetcher ------------------------------------------------------------

def get_ohlcv_oanda(
    symbol: str,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    count: Optional[int] = None,
    price: str = "M",   # M = mid, B = bid, A = ask
) -> pd.DataFrame:
    """
    Fetch OANDA candles as OHLCV DataFrame (UTC index).
    Supports either (start/end) or (count). Filters to complete candles.
    """
    inst = _map_symbol(symbol)
    gran = _granularity(interval)
    host = _env_domain()
    url = f"https://{host}/v3/instruments/{inst}/candles"

    params = {"granularity": gran, "price": price}
    if count is not None:
        # OANDA max is 5000 per request
        params["count"] = int(max(1, min(int(count), 5000)))
    else:
        if start:
            params["from"] = _to_iso_utc(start)
        if end:
            params["to"] = _to_iso_utc(end)

    r = requests.get(url, headers=_auth_headers(), params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("OANDA auth failed (401). Check OANDA_TOKEN / OANDA_ENV.")
    r.raise_for_status()
    data = r.json()

    cands = data.get("candles", []) or []
    rows = []
    for c in cands:
        # expect structure: {time, mid:{o,h,l,c}, volume, complete}
        mid = c.get("mid") or {}
        rows.append(
            {
                "time": c.get("time"),
                "open": float(mid.get("o", "nan")),
                "high": float(mid.get("h", "nan")),
                "low": float(mid.get("l", "nan")),
                "close": float(mid.get("c", "nan")),
                "volume": int(c.get("volume", 0)),
                "complete": bool(c.get("complete", False)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.set_index("time").sort_index()
    df = _only_complete(df)
    return df
