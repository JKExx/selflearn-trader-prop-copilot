# app/news.py
from __future__ import annotations
import io, re, requests
import pandas as pd

def fetch_ics(url: str) -> list[dict]:
    """
    Returns a list of events: [{"start": Timestamp(UTC), "end": Timestamp(UTC), "summary": str}, ...]
    Minimal ICS parser covering DTSTART/DTEND/SUMMARY.
    """
    if not url:
        return []
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    lines = r.text.splitlines()

    # unfold folded lines
    unfolded = []
    for ln in lines:
        if ln.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += ln.strip()
        else:
            unfolded.append(ln.strip())

    events, cur = [], {}
    for ln in unfolded:
        if ln == "BEGIN:VEVENT":
            cur = {}
        elif ln == "END:VEVENT":
            if "DTSTART" in cur:
                start = _parse_ics_dt(cur.get("DTSTART"))
                end = _parse_ics_dt(cur.get("DTEND") or cur.get("DTSTART"))
                events.append({"start": start, "end": end, "summary": cur.get("SUMMARY","")})
            cur = {}
        else:
            if ":" in ln:
                k, v = ln.split(":", 1)
                k = k.split(";")[0].strip().upper()
                cur[k] = v.strip()
    # normalize & drop invalid
    out = []
    for e in events:
        try:
            s = _to_utc(pd.to_datetime(e["start"], utc=True, errors="coerce"))
            e_ = _to_utc(pd.to_datetime(e["end"], utc=True, errors="coerce"))
            if pd.isna(s):
                continue
            if pd.isna(e_):
                e_ = s
            out.append({"start": s, "end": e_, "summary": e.get("summary","")})
        except Exception:
            pass
    return out

def _parse_ics_dt(x: str) -> str:
    # Support Zulu and floating; pass to pandas
    return x.replace("Z","Z")

def _to_utc(ts):
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

def blackout_now(ts_utc, events: list[dict], keywords: list[str], window_min: int) -> bool:
    """
    True if any event is within ±window_min minutes of ts_utc and matches keywords.
    """
    ts_utc = _to_utc(pd.to_datetime(ts_utc, utc=True))
    lo = ts_utc - pd.Timedelta(minutes=int(window_min))
    hi = ts_utc + pd.Timedelta(minutes=int(window_min))
    kws = [k.strip().lower() for k in (keywords or []) if k.strip()]
    for e in events or []:
        s = _to_utc(e["start"]); en = _to_utc(e["end"])
        if (s <= hi) and (en >= lo):
            text = (e.get("summary","") or "").lower()
            if not kws or any(k in text for k in kws):
                return True
    return False
