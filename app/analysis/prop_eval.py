# app/analysis/prop_eval.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass
class PropConfig:
    daily_loss_pct: float
    overall_loss_pct: float
    profit_target_pct: float | None
    day_boundary_tz: str = "America/New_York"
    friday_cutoff_hour_local: int = 22
    news_blackout_mins: int = 0
    relevant_currencies: set[str] | None = None

def _to_prop_day(ts_utc: pd.Timestamp, tz: str) -> str:
    local = ts_utc.tz_convert(tz)
    return local.strftime("%Y-%m-%d")

def evaluate_prop(trades: pd.DataFrame, cfg: PropConfig, news_df: pd.DataFrame | None = None):
    """Compute board (summary), per-day table, and violations list."""
    if trades is None or trades.empty:
        board = {"days": 0, "breaches_daily": 0, "breaches_overall": 0, "profit_target_hit": False}
        return board, pd.DataFrame(), []

    df = trades.copy()
    df["time_close"] = pd.to_datetime(df["time_close"], utc=True, errors="coerce")
    df["prop_day"] = df["time_close"].apply(lambda x: _to_prop_day(x, cfg.day_boundary_tz))
    df["equity_after"] = df["equity_after"].astype(float)

    # Daily stats
    grp = df.groupby("prop_day", as_index=False)
    daily = grp.agg(
        day_pnl=("pnl", "sum"),
        day_min_eq=("equity_after", "min"),
        day_max_eq=("equity_after", "max"),
        last_eq=("equity_after", "last"),
    )
    # Daily DD from day high (percentage of day high)
    daily["day_min_dd_pct"] = ((daily["day_min_eq"] / daily["day_max_eq"]) - 1.0).fillna(0.0)

    start_eq = float(df["equity_after"].iloc[0] - df["pnl"].iloc[0]) if len(df) else 0.0
    cur_eq = float(df["equity_after"].iloc[-1])

    # Daily breach: if day PnL <= -daily_loss_pct * start_eq_of_that_day (approx via day_max_eq)
    daily_cap = float(cfg.daily_loss_pct)
    daily["breached_daily"] = (daily["day_min_dd_pct"] <= -(daily_cap / 100.0))

    # Overall breach: from high-water mark
    eq = df["equity_after"].astype(float)
    hwm = eq.cummax()
    dd = (eq / hwm) - 1.0
    overall_breached = float(dd.min()) <= -(cfg.overall_loss_pct / 100.0)

    # Profit target?
    profit_hit = False
    if cfg.profit_target_pct is not None:
        profit_hit = (cur_eq >= start_eq * (1.0 + float(cfg.profit_target_pct)))

    board = {
        "start_eq": start_eq,
        "end_eq": cur_eq,
        "max_dd_pct": float(dd.min()) * 100.0,
        "breaches_daily": int(daily["breached_daily"].sum()),
        "breaches_overall": int(overall_breached),
        "profit_target_hit": bool(profit_hit),
        "days": int(len(daily)),
    }

    violations = []
    if overall_breached:
        violations.append({"type": "overall_dd", "at": df.loc[dd.idxmin(), "time_close"]})

    return board, daily, violations
