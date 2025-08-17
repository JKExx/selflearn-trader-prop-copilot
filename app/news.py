# app/news.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytz
import requests
from icalendar import Calendar

UTC = pytz.UTC


def _to_utc(dt_like) -> pd.Timestamp:
    ts = pd.to_datetime(dt_like, errors="coerce")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def fetch_ics(url: str) -> list[dict[str, Any]]:
    """Fetch an .ics calendar and return list of events: {start, end, summary} in UTC."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    cal = Calendar.from_ical(r.content)
    out: list[dict[str, Any]] = []
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue
        start = comp.get("dtstart").dt
        end = comp.get("dtend").dt
        summary = str(comp.get("summary", "") or "")
        out.append({"start": _to_utc(start), "end": _to_utc(end), "summary": summary})
    return out


def filter_events(
    events: Iterable[dict[str, Any]],
    window_minutes: int,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    keywords: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter events to a time window [lo, hi] and keyword list (case-insensitive)."""
    kws = [k.strip().lower() for k in (keywords or []) if k.strip()]
    rows = []
    for e in events or []:
        s = _to_utc(e["start"])
        en = _to_utc(e["end"])
        if (s <= hi) and (en >= lo):
            text = (e.get("summary", "") or "").lower()
            if not kws or any(k in text for k in kws):
                rows.append({"time_utc": s, "end_utc": en, "summary": e.get("summary", "")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time_utc").reset_index(drop=True)
    return df
