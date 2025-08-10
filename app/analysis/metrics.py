
import numpy as np
import pandas as pd

def equity_from_trades(trades: pd.DataFrame, starting_cash: float = 10000.0) -> pd.Series:
    if "equity_after" in trades.columns and trades["equity_after"].notna().any():
        eq = trades["equity_after"].copy()
        eq = pd.concat([pd.Series([eq.iloc[0] - trades["pnl"].iloc[0]]), eq], ignore_index=True)
        return pd.Series(eq.values)
    eq = starting_cash + trades["pnl"].cumsum()
    return pd.concat([pd.Series([starting_cash]), eq], ignore_index=True)

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0

def sharpe_sortino(trades: pd.DataFrame, periods_per_year: int = 252*24):
    """Compute daily/hourly Sharpe/Sortino from per-trade returns approx."""
    if "equity_after" in trades.columns and trades["equity_after"].notna().any():
        equity_before = trades["equity_after"] - trades["pnl"]
        rets = trades["pnl"] / equity_before.replace(0, np.nan)
    else:
        start = 10000.0
        eq = start + trades["pnl"].cumsum()
        equity_before = pd.Series([start]).append(eq[:-1], ignore_index=True)
        rets = trades["pnl"] / equity_before.replace(0, np.nan)
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 2:
        return 0.0, 0.0
    mean = rets.mean()
    std = rets.std(ddof=1)
    neg = rets[rets < 0]
    dd = neg.std(ddof=1) if len(neg) else np.nan
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std and not np.isnan(std) else 0.0
    sortino = (mean / dd) * np.sqrt(periods_per_year) if dd and not np.isnan(dd) else 0.0
    return float(sharpe), float(sortino)

def profit_factor(trades: pd.DataFrame) -> float:
    wins = trades["pnl"][trades["pnl"] > 0].sum()
    losses = trades["pnl"][trades["pnl"] < 0].sum()
    return float(wins / abs(losses)) if losses != 0 else float("inf")

def streaks(trades: pd.DataFrame):
    s = (trades["pnl"] > 0).astype(int).values
    if len(s) == 0: return 0, 0
    max_w = max_l = cur_w = cur_l = 0
    prev = None
    for v in s:
        if v == 1:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return int(max_w), int(max_l)

def exposure(trades: pd.DataFrame) -> float:
    """Approx exposure as fraction of bars in a position (time-based)."""
    if "time_open" in trades.columns and "time_close" in trades.columns:
        dur = (pd.to_datetime(trades["time_close"], utc=True) - pd.to_datetime(trades["time_open"], utc=True)).dt.total_seconds()
        total = (pd.to_datetime(trades["time_close"].iloc[-1], utc=True) - pd.to_datetime(trades["time_open"].iloc[0], utc=True)).total_seconds()
        return float(dur.sum() / total) if total > 0 else 0.0
    return float("nan")

def compute_metrics(trades: pd.DataFrame) -> dict:
    n = int(len(trades))
    eq = equity_from_trades(trades)
    mdd = max_drawdown(eq)
    pf = profit_factor(trades)
    sharpe, sortino = sharpe_sortino(trades)
    wr = float((trades["pnl"] > 0).mean()) if n else 0.0
    expectancy = float(trades["pnl"].mean()) if n else 0.0
    med = float(trades["pnl"].median()) if n else 0.0
    exp = exposure(trades)
    max_w, max_l = streaks(trades)
    out = {
        "trades": n,
        "win_rate": round(wr, 4),
        "total_pnl": round(float(trades["pnl"].sum()), 2) if "pnl" in trades.columns else 0.0,
        "avg_pnl": round(expectancy, 4),
        "median_pnl": round(med, 4),
        "profit_factor": (float("inf") if pf == float("inf") else round(pf, 3)),
        "max_drawdown_pct": round(mdd * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "exposure_pct": round(exp * 100, 2) if not np.isnan(exp) else None,
        "max_win_streak": max_w,
        "max_lose_streak": max_l,
    }
    return out
