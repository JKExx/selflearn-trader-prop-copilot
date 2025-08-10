
import pandas as pd
import numpy as np

class PropConfig:
    def __init__(self,
                 daily_loss_pct=0.05,
                 overall_loss_pct=0.10,
                 profit_target_pct=None,
                 day_boundary_tz="America/New_York",
                 friday_cutoff_hour_local=None,
                 news_blackout_mins=0,
                 relevant_currencies=None):
        self.daily_loss_pct = daily_loss_pct
        self.overall_loss_pct = overall_loss_pct
        self.profit_target_pct = profit_target_pct
        self.day_boundary_tz = day_boundary_tz
        self.friday_cutoff_hour_local = friday_cutoff_hour_local
        self.news_blackout_mins = news_blackout_mins
        self.relevant_currencies = set(relevant_currencies or [])

def evaluate_prop(trades: pd.DataFrame, cfg: PropConfig, news_df: pd.DataFrame | None = None):
    """Return (scoreboard dict, daily table, violations list)."""
    if trades.empty:
        return {"pass": False, "reason": "no-trades"}, pd.DataFrame(), []

    t_open = pd.to_datetime(trades["time_open"], utc=True, errors="coerce")
    t_close = pd.to_datetime(trades["time_close"], utc=True, errors="coerce")
    pnl = trades["pnl"].astype(float)

    # Build equity series
    if "equity_after" in trades.columns and trades["equity_after"].notna().any():
        eq = trades["equity_after"].astype(float)
        start_eq = float((eq.iloc[0] - pnl.iloc[0]))
        end_eq = float(eq.iloc[-1])
    else:
        start_eq = 10000.0
        end_eq = float(start_eq + pnl.sum())

    # Overall DD
    eq_series = pd.Series([start_eq] + list(start_eq + pnl.cumsum()))
    peak = eq_series.cummax()
    overall_dd = float(((eq_series - peak) / peak).min())

    # Day buckets (5pm New York by default)
    ts = t_close.fillna(t_open).copy()
    local = ts.dt.tz_convert(cfg.day_boundary_tz)
    day_key = (local - pd.Timedelta(hours=17)).dt.date
    trades = trades.copy()
    trades["__prop_day"] = day_key

    # Intraday min equity per day
    equity_after = start_eq + pnl.cumsum()
    trades["__eq_after"] = equity_after
    day_stats = []
    for day, grp in trades.groupby("__prop_day"):
        day_start_eq = float((grp["__eq_after"] - grp["pnl"]).iloc[0])
        intraday_min = float(grp["__eq_after"].cummin().min())
        day_min_dd = (intraday_min - day_start_eq) / day_start_eq
        day_stats.append((day, day_start_eq, day_min_dd))
    daily = pd.DataFrame(day_stats, columns=["prop_day","day_start_eq","day_min_dd"])
    daily["day_min_dd_pct"] = daily["day_min_dd"] * 100.0
    daily["breached_daily"] = daily["day_min_dd"] <= -cfg.daily_loss_pct

    # Friday cutoff violations
    violations = []
    if cfg.friday_cutoff_hour_local is not None:
        loc = t_open.dt.tz_convert(cfg.day_boundary_tz)
        fri = loc.dt.weekday == 4
        after_cut = loc.dt.hour >= int(cfg.friday_cutoff_hour_local)
        bad = trades[fri & after_cut]
        for _, r in bad.iterrows():
            violations.append({"type":"friday-cutoff", "time_open": str(r["time_open"]), "side": r.get("side")})

    # News blackout violations (if news_df provided)
    if news_df is not None and len(news_df):
        news = news_df.copy()
        news["time_utc"] = pd.to_datetime(news["time_utc"], utc=True, errors="coerce")
        news = news.dropna(subset=["time_utc"])
        if cfg.relevant_currencies:
            news = news[news["currency"].isin(cfg.relevant_currencies)]
        window = pd.Timedelta(minutes=abs(cfg.news_blackout_mins))
        for _, tr in trades.iterrows():
            to = pd.to_datetime(tr["time_open"], utc=True, errors="coerce")
            if pd.isna(to): continue
            near = news[(news["time_utc"] >= to - window) & (news["time_utc"] <= to + window)]
            if len(near):
                violations.append({"type":"news-blackout", "time_open": str(to), "count_events": int(len(near))})

    # Profit target
    ret_total = (end_eq - start_eq) / start_eq
    target_ok = True
    if cfg.profit_target_pct is not None:
        target_ok = ret_total >= cfg.profit_target_pct

    scoreboard = {
        "start_equity": round(start_eq, 2),
        "end_equity": round(end_eq, 2),
        "return_pct": round(ret_total * 100, 2),
        "overall_max_dd_pct": round(overall_dd * 100, 2),
        "overall_cap_pct": cfg.overall_loss_pct * 100,
        "daily_cap_pct": cfg.daily_loss_pct * 100,
        "phase_target_pct": (cfg.profit_target_pct * 100 if cfg.profit_target_pct is not None else None),
        "days_breached_daily_cap": int(daily["breached_daily"].sum()),
        "violations_count": len(violations),
    }

    pass_rules = (
        (scoreboard["overall_max_dd_pct"] >= -cfg.overall_loss_pct * 100) and
        (scoreboard["days_breached_daily_cap"] == 0) and
        target_ok
    )
    scoreboard["pass"] = bool(pass_rules)
    return scoreboard, daily, violations
