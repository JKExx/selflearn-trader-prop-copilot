from __future__ import annotations

import pandas as pd

def streaks(trades: pd.DataFrame):
    s = (trades["pnl"] > 0).astype(int).values
    if len(s) == 0:
        return 0, 0
    max_w = 0
    max_l = 0
    cur_w = 0
    cur_l = 0
    for v in s:
        if v == 1:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l

def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    df = trades.copy()
    pnl = df["pnl"].astype(float)
    eq = df["equity_after"].astype(float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    pf = (wins / abs(losses)) if float(losses) != 0 else float("inf")
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    hwm = eq.cummax()
    dd = (eq / hwm) - 1.0
    max_win_streak, max_loss_streak = streaks(df)
    return {
        "trades": int(len(df)),
        "net_pnl": float(pnl.sum()),
        "profit_factor": (float("inf") if pf == float("inf") else float(pf)),
        "win_rate": win_rate,
        "max_dd_pct": float(dd.min()) * 100.0,
        "end_equity": float(eq.iloc[-1]),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
    }
