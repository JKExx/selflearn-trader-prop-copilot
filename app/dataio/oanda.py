# app/dataio/oanda.py
import os
import pandas as pd

try:
    import oandapyV20
    from oandapyV20.endpoints.instruments import InstrumentsCandles
except Exception:
    oandapyV20 = None

from ..utils import ensure_datetime_index

# ---------- helpers ----------
def _to_oanda_instrument(symbol: str) -> str:
    s = symbol.upper().replace("=X", "").replace("-", "").replace("/", "").strip()
    if "_" in s:        # e.g. EUR_USD
        return s
    if len(s) == 6:     # e.g. EURUSD
        return f"{s[:3]}_{s[3:]}"
    return s

def _to_oanda_granularity(interval: str) -> str:
    m = {
        "1m": "M1", "2m": "M2", "5m": "M5", "15m": "M15", "30m": "M30",
        "60m": "H1", "1h": "H1", "90m": "H1", "4h": "H4",
        "1d": "D", "1wk": "W", "1mo": "M"
    }
    return m.get(interval, "H1")

def _gran_step_td(gran: str) -> pd.Timedelta:
    return {
        "M1": pd.Timedelta(minutes=1),
        "M2": pd.Timedelta(minutes=2),
        "M5": pd.Timedelta(minutes=5),
        "M15": pd.Timedelta(minutes=15),
        "M30": pd.Timedelta(minutes=30),
        "H1": pd.Timedelta(hours=1),
        "H4": pd.Timedelta(hours=4),
        "D":  pd.Timedelta(days=1),
        "W":  pd.Timedelta(days=7),
        "M":  pd.Timedelta(days=30),
    }.get(gran, pd.Timedelta(hours=1))

def _default_lookback(gran: str) -> pd.Timedelta:
    # sensible defaults if user leaves "Start" blank
    return {
        "M15": pd.Timedelta(days=120),
        "M30": pd.Timedelta(days=180),
        "H1":  pd.Timedelta(days=365),
        "H4":  pd.Timedelta(days=2*365),
        "D":   pd.Timedelta(days=5*365),
    }.get(gran, pd.Timedelta(days=365))

# ---------- main ----------
def get_ohlcv_oanda(symbol: str, interval: str = "1h",
                    start: str | None = None, end: str | None = None,
                    price: str = "M") -> pd.DataFrame:
    """
    Fetch OHLCV from OANDA v20 using from/to pagination (no 'count').
    symbol: 'EUR_USD', 'EURUSD', 'EURUSD=X', 'XAU_USD', ...
    interval: '15m','30m','1h','1d' (mapped to OANDA granularities)
    """
    if oandapyV20 is None:
        raise ImportError("oandapyV20 not installed. Run: pip install oandapyV20")

    token = os.getenv("OANDA_TOKEN")
    env   = os.getenv("OANDA_ENV", "practice")
    if not token:
        raise RuntimeError("Missing OANDA_TOKEN env var. Set it and restart the app.")

    client = oandapyV20.API(access_token=token, environment=env)
    instrument = _to_oanda_instrument(symbol)
    gran = _to_oanda_granularity(interval)
    step = _gran_step_td(gran)

    now = pd.Timestamp.now(tz="UTC")
    end_ts = pd.to_datetime(end, utc=True) if (end and str(end).strip()) else now
    if start and str(start).strip():
        from_ts = pd.to_datetime(start, utc=True)
    else:
        from_ts = end_ts - _default_lookback(gran)

    rows = []
    # paginate forward using from/to window slices; OANDA returns up to ~5000 per call
    cur_from = from_ts
    while cur_from < end_ts:
        slice_to = min(cur_from + step * 4800, end_ts)  # ~4800 bars per request
        params = {
            "granularity": gran,
            "price": price,
            "from": cur_from.isoformat(),
            "to": slice_to.isoformat(),
        }
        r = InstrumentsCandles(instrument=instrument, params=params)
        data = client.request(r)
        candles = data.get("candles", [])
        if not candles:
            # move forward anyway to avoid infinite loop
            cur_from = slice_to + step
            continue

        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid") or c.get("bid") or c.get("ask")
            rows.append({
                "datetime": pd.to_datetime(c["time"]),
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low":  float(mid["l"]),
                "Close":float(mid["c"]),
                "Volume": int(c.get("volume", 0)),
            })

        # next page
        last_time = pd.to_datetime(candles[-1]["time"])
        cur_from = max(slice_to, last_time) + step

    if not rows:
        raise ValueError(f"No OANDA data for {instrument} {gran} {from_ts}→{end_ts}.")

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df = ensure_datetime_index(df)
    return df[["Open","High","Low","Close","Volume"]]
