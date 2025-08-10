# app/utils.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- datetime helpers ----------

def ensure_datetime_index(obj, tz: str = "UTC"):
    """
    Ensure a tz-aware DatetimeIndex (default UTC).

    - If `obj` is a DataFrame: converts/sets its index to tz-aware DatetimeIndex
      and returns the SAME DataFrame (mutated).
    - If `obj` is a DatetimeIndex (or anything index-like): returns a tz-aware
      DatetimeIndex.
    """
    if isinstance(obj, pd.DataFrame):
        idx = obj.index
        # if index is not datetime, try to convert
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx, errors="coerce", utc=False)
            except Exception:
                raise TypeError("DataFrame index could not be converted to datetime.")
        # localize/convert to tz
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
        obj.index = idx
        return obj

    # Otherwise treat as index-like
    idx = pd.DatetimeIndex(obj)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    return idx


def _ensure_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """(Internal) Return a tz-aware UTC DatetimeIndex."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


# ---------- session flags (DST-aware) ----------

def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build per-bar session/context flags (DST-aware).
    Returns a DataFrame indexed by UTC with columns:
      - hour          (UTC hour 0..23)
      - dow           (0=Mon..6=Sun)
      - is_london_kz  (1 during 08:00–11:00 *London local*, else 0)
      - is_ny_kz      (1 during 08:00–11:00 *New York local*, else 0)
    """
    utc = _ensure_utc_index(index)

    # Convert to real market local times (handles DST automatically)
    lon = utc.tz_convert("Europe/London")
    ny  = utc.tz_convert("America/New_York")

    # Local killzones (08:00–11:00 local)
    lon_kz = ((lon.hour >= 8) & (lon.hour < 11)).astype(int)
    ny_kz  = ((ny.hour  >= 8) & (ny.hour  < 11)).astype(int)

    # Build output; never use .values on arrays
    out = pd.DataFrame(index=utc)
    out["hour"] = utc.hour
    out["dow"] = utc.weekday
    out["is_london_kz"] = np.asarray(lon_kz, dtype=int)
    out["is_ny_kz"] = np.asarray(ny_kz, dtype=int)
    return out
