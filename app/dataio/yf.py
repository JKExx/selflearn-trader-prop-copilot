# app/dataio/yf.py
import pandas as pd
import yfinance as yf
from ..utils import ensure_datetime_index

INTRADAY = {"1m","2m","5m","15m","30m","60m","90m","1h"}

def _clamp_intraday_start(interval: str, start: str | None) -> str | None:
    if not start:
        return start
    if interval not in INTRADAY:
        return start
    start_ts = pd.to_datetime(start, utc=True)
    earliest = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=729)
    if start_ts < earliest:
        return earliest.strftime("%Y-%m-%d")
    return start

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If Yahoo returns MultiIndex (e.g., ('Open','EURUSD=X')), squash to first level
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        # Prefer the OHLCV level if present
        if {"Open","High","Low","Close","Adj Close","Volume"}.issubset(set(lvl0)) or any(
            v in {"Open","High","Low","Close","Adj Close","Volume"} for v in lvl0
        ):
            df.columns = lvl0
        else:
            # Fallback: join levels with underscore
            df.columns = ["_".join([str(p) for p in tup if p is not None]) for tup in df.columns]
    else:
        # Make sure all column names are strings
        df.columns = [str(c) for c in df.columns]
    return df

def get_ohlcv(symbol: str, interval: str = '1h', start: str = None, end: str = None) -> pd.DataFrame:
    start = _clamp_intraday_start(interval, start)

    df = yf.download(
        tickers=symbol,
        interval=interval,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",   # <- avoid MultiIndex when possible
    )

    if df is None or len(df) == 0:
        raise ValueError(f"No data for {symbol} {interval} {start} {end}")

    df = _flatten_columns(df)

    # Standardize columns
    rename = {
        'Open':'Open','High':'High','Low':'Low','Close':'Close',
        'Adj Close':'Adj Close','Volume':'Volume'
    }
    df = df.rename(columns=rename)

    df = ensure_datetime_index(df)
    return df[['Open','High','Low','Close','Volume']]
