# app/features.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from .utils import session_flags

def make_features(df: pd.DataFrame):
    """
    Build model features from OHLCV DataFrame.
    Expects columns: Open, High, Low, Close, Volume (case/level agnostic).
    Returns (df_with_features, feature_cols).
    """
    df = df.copy()

    # --- Normalize columns (yfinance sometimes returns MultiIndex/tuples) ---
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the OHLCV level (level 0) if present
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = [str(c) for c in df.columns]

    # Map any lowercase/mixed-case to standard OHLCV names
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    for tgt in wanted:
        if tgt not in df.columns:
            lc = tgt.lower()
            if lc in lower_map:
                rename_map[lower_map[lc]] = tgt
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got {list(df.columns)}")

    # --- Returns / momentum / trend ---
    df["ret_1"] = df["Close"].pct_change()
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_20"] = df["Close"].pct_change(20)

    rsi = RSIIndicator(close=df["Close"], window=14).rsi()
    df["rsi_14"] = rsi

    sma20 = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    ema20 = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["sma_20"] = sma20
    df["ema_20"] = ema20
    df["sma_slope"] = df["sma_20"].diff()
    df["ema_slope"] = df["ema_20"].diff()

    # --- Volatility / ATR ---
    atr14 = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()
    df["atr_14"] = atr14
    df["atr_norm"] = df["atr_14"] / df["Close"]

    # --- Price vs MAs ---
    df["px_vs_sma"] = (df["Close"] - df["sma_20"]) / df["sma_20"]
    df["px_vs_ema"] = (df["Close"] - df["ema_20"]) / df["ema_20"]

    # --- Sessions / time context (UTC-safe) ---
    sess = session_flags(df.index)
    df = pd.concat([df, sess], axis=1)

    # --- Label for online learning: next-bar up/down ---
    df["y_next_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop NaNs from indicator warmups
    df = df.dropna().copy()

    feature_cols = [
        "ret_1", "ret_5", "ret_20",
        "rsi_14",
        "sma_slope", "ema_slope",
        "atr_norm",
        "px_vs_sma", "px_vs_ema",
        "hour", "dow", "is_london_kz", "is_ny_kz",
    ]

    return df, feature_cols
