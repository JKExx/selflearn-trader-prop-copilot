# app/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .models.online_classifier import OnlineClassifier
from .features import make_features


def _to_numpy(x):
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x, dtype=float)


def _std_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
    """Coerce incoming data to columns: open, high, low, close, volume."""
    df = df_in.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    o = pick("open")
    h = pick("high")
    l = pick("low")
    c = pick("close", "adj close", "adj_close")
    v = pick("volume")

    out = pd.DataFrame(index=df.index)
    out["open"] = df[o] if o else df.iloc[:, 0]
    out["high"] = df[h] if h else df.iloc[:, 1]
    out["low"] = df[l] if l else df.iloc[:, 2]
    out["close"] = df[c] if c else df.iloc[:, 3]
    out["volume"] = df[v] if v else 0.0
    return out


def _pip_size(symbol: str, default_fx: float = 1e-4) -> float:
    s = (symbol or "").upper().replace("=X", "")
    if "XAU" in s or "GOLD" in s:
        return 0.01
    if "JPY" in s:
        return 0.01
    return default_fx


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: np.ndarray
    stop_reason: str | None = None
    meta: dict | None = None


def run_backtest(
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    interval: str = "",
    # decision
    threshold: float = 0.60,
    epsilon_start: float = 0.02,
    epsilon_final: float = 0.02,
    epsilon_decay_trades: int = 0,  # 0 = constant
    # warm-up
    warmup_bars: int = 300,
    trade_start_date: Optional[str] = None,  # UTC date string -> allow learning but no trades before
    starting_cash: float = 10_000.0,
    # sizing / costs
    risk_per_trade: float = 0.005,
    atr_multiple: float = 1.0,
    spread_pips: float = 0.2,
    slippage_pips: float = 0.1,
    commission_per_trade: float = 0.0,
    # gates & persistence
    gate_killzones: bool = False,
    persist_model: bool = True,
    model_path: str | None = None,  # if None: auto by symbol+interval
    # enforcement (prop)
    enforce_overall_cap: bool = True,
    overall_cap_pct: float = 0.10,
    enforce_daily_cap: bool = True,
    daily_cap_pct: float = 0.05,
    day_boundary_tz: str = "America/New_York",
    friday_cutoff_local: Optional[int] = None,
    stop_at_profit_target_pct: Optional[float] = None,  # e.g., 0.08
    # news filter (precomputed events)
    news_events: Optional[list[dict]] = None,
    news_keywords: Optional[list[str]] = None,
    news_blackout_minutes: int = 0,
    # -------- Guardrails (new) --------
    trailing_hwm_cap_pct: Optional[float] = None,   # stop whole phase if drop from HWM > X%
    max_trades_per_day: Optional[int] = None,       # cap number of trades per prop day
    loss_streak_pause_bars: int = 0,                # pause N bars after loss streak
    loss_streak_trigger: int = 0,                   # trigger after N consecutive losses
    dd_adapt_threshold_pct: Optional[float] = None, # enter adapt mode when eq <= HWM*(1-X)
    dd_adapt_risk: Optional[float] = None,          # risk while in adapt mode
    dd_adapt_threshold_bump: float = 0.0,           # add to threshold while in adapt
    max_intraday_dd_from_high_pct: Optional[float] = None,  # lock rest of day if drop from day-high > X
) -> BacktestResult:
    """One-bar open/close simulator with online learning + prop guardrails."""

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise ValueError("Input OHLCV must have a DatetimeIndex.")

    df = _std_ohlcv(ohlcv).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # --- feature pipeline ---
    feats_out = make_features(df)
    feat_df = feats_out[0].copy() if isinstance(feats_out, (tuple, list)) else feats_out.copy()
    if "y_next_up" not in feat_df.columns:
        raise RuntimeError("make_features() must produce 'y_next_up'.")

    # ATR fallback if not provided by features
    if "atr" not in feat_df.columns:
        hi, lo, cl = df["high"], df["low"], df["close"]
        prev_close = cl.shift(1)
        tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
        feat_df["atr"] = tr.rolling(14, min_periods=1).mean()

    # disable killzones if features missing
    if gate_killzones and not ({"is_london_kz", "is_ny_kz"} <= set(feat_df.columns)):
        gate_killzones = False

    feat_df = feat_df.reindex(df.index).dropna().copy()

    # calendar warm-up (no trades before this UTC date)
    trade_start_ts = pd.Timestamp(trade_start_date, tz="UTC") if trade_start_date else None

    # prop-day boundary (5pm New York)
    local = feat_df.index.tz_convert(day_boundary_tz)
    prop_day = (local - pd.Timedelta(hours=17)).date
    prop_day = pd.Series(prop_day, index=feat_df.index)

    # features/labels
    non_feat = {"y_next_up", "atr", "is_london_kz", "is_ny_kz"}
    X_cols = [c for c in feat_df.columns if c not in non_feat]
    X_full = _to_numpy(feat_df[X_cols])
    y_full = _to_numpy(feat_df["y_next_up"]).ravel().astype(int)
    atr = _to_numpy(feat_df["atr"])
    closes = df["close"].reindex(feat_df.index)

    # model path
    if model_path is None:
        safe = f"{(symbol or 'SYM').upper()}_{(interval or 'TF').lower()}".replace("/", "-")
        model_path = f"models_ckpt/{safe}.joblib"

    model = OnlineClassifier(model_path=model_path)
    if persist_model:
        model.load_if_exists()

    # warm-up by bars to init scaler
    i0 = max(int(warmup_bars), 1)
    if i0 > 0 and i0 < len(X_full):
        model.partial_fit(X_full[:i0], y_full[:i0])
    else:
        i0 = 1

    # price costs
    pip = _pip_size(symbol)
    spread_px = (float(spread_pips) / 2.0) * pip
    slip_px_in = float(slippage_pips) * pip
    slip_px_out = float(slippage_pips) * pip

    # loop state
    trades: List[Dict] = []
    equity = starting_cash
    phase_start_eq = starting_cash
    stop_reason = None

    current_day = None
    day_start_eq = None
    day_locked = False
    day_high_eq = None

    n = len(feat_df)
    rng = np.random.default_rng(42)  # deterministic unless you change seed
    exec_trades = 0
    consec_losses = 0
    pause_bars_left = 0
    hwm = starting_cash
    trades_today = 0

    def in_dd_adapt(eq: float) -> bool:
        if dd_adapt_threshold_pct is None:
            return False
        return eq <= (hwm * (1.0 - float(dd_adapt_threshold_pct)))

    # --- main loop ---
    for i in range(i0, n - 1):
        ts = feat_df.index[i]

        # learning-only warmup before calendar start
        if (trade_start_ts is not None) and (ts < trade_start_ts):
            model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
            continue

        # new prop day
        day_key = prop_day.iloc[i]
        if (current_day is None) or (day_key != current_day):
            current_day = day_key
            day_start_eq = equity
            day_high_eq = equity
            day_locked = False
            trades_today = 0
            consec_losses = 0
            pause_bars_left = 0

        # track day high; lock day if intraday DD from day-high exceeds cap
        if day_high_eq is None or equity > day_high_eq:
            day_high_eq = equity
        if (max_intraday_dd_from_high_pct is not None) and (not day_locked):
            if equity <= day_high_eq * (1.0 - float(max_intraday_dd_from_high_pct)):
                day_locked = True

        # overall cap (from phase start)
        if enforce_overall_cap and equity <= phase_start_eq * (1.0 - float(overall_cap_pct)):
            stop_reason = "overall_cap"
            break

        # trailing HWM cap (peak-to-trough)
        if (trailing_hwm_cap_pct is not None) and (equity <= hwm * (1.0 - float(trailing_hwm_cap_pct))):
            stop_reason = "trailing_hwm_cap"
            break

        # Friday cutoff (local)
        if friday_cutoff_local is not None:
            loc = local[i]
            if loc.weekday() == 4 and loc.hour >= int(friday_cutoff_local):
                model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
                continue

        # daily cap lockout
        if enforce_daily_cap and day_locked:
            model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
            continue

        # max trades per day
        if (max_trades_per_day is not None) and (trades_today >= int(max_trades_per_day)):
            model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
            continue

        # loss-streak cooldown
        if pause_bars_left > 0:
            pause_bars_left -= 1
            model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
            continue

        # news blackout
        if news_events and news_blackout_minutes > 0:
            from .news import blackout_now  # local import to avoid hard dep at import time
            if blackout_now(ts, news_events, news_keywords or [], news_blackout_minutes):
                model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
                continue

        # killzones gating
        if gate_killzones:
            if int(feat_df["is_london_kz"].iloc[i] or feat_df["is_ny_kz"].iloc[i]) != 1:
                model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])
                continue

        # epsilon schedule
        if epsilon_decay_trades and exec_trades < epsilon_decay_trades:
            frac = exec_trades / float(max(epsilon_decay_trades, 1))
            epsilon = float(epsilon_start) + (float(epsilon_final) - float(epsilon_start)) * frac
        else:
            epsilon = float(epsilon_final)

        # adapt mode (below HWM threshold)
        eff_risk = float(risk_per_trade)
        eff_threshold = float(threshold)
        if in_dd_adapt(equity):
            if dd_adapt_risk is not None:
                eff_risk = float(min(eff_risk, dd_adapt_risk))
            eff_threshold = float(eff_threshold + dd_adapt_threshold_bump)

        # model decision
        proba = model.predict_proba(X_full[i:i + 1])[0, 1]
        explored = False
        if rng.random() < epsilon:
            explored = True
            side = rng.choice([-1, 0, 1])
        else:
            side = OnlineClassifier.decide_action(proba, eff_threshold)

        # price & stop math
        entry_mid = float(closes.iloc[i])
        exit_mid = float(closes.iloc[i + 1])
        bar_atr = float(max(atr[i], 1e-12))
        stop_dist = bar_atr * float(atr_multiple)

        # units by risk/stop (float)
        units = 0.0
        if side != 0 and stop_dist > 0:
            units = (equity * eff_risk) / stop_dist

        # execution (apply costs)
        entry_exec, exit_exec = entry_mid, exit_mid
        if side == 1:
            entry_exec += (spread_px + slip_px_in)
            exit_exec -= (spread_px + slip_px_out)
        elif side == -1:
            entry_exec -= (spread_px + slip_px_in)
            exit_exec += (spread_px + slip_px_out)

        pnl = 0.0
        if side == 1:
            pnl = (exit_exec - entry_exec) * units
        elif side == -1:
            pnl = (entry_exec - exit_exec) * units
        pnl -= float(commission_per_trade)

        if side != 0:
            equity += pnl
            exec_trades += 1
            trades_today += 1

            # loss streak tracking
            if pnl < 0:
                consec_losses += 1
                if loss_streak_trigger and consec_losses >= int(loss_streak_trigger):
                    pause_bars_left = int(loss_streak_pause_bars)
                    consec_losses = 0
            else:
                consec_losses = 0

            # daily cap lock
            if enforce_daily_cap and (equity <= day_start_eq * (1.0 - float(daily_cap_pct))):
                day_locked = True

            # update HWM
            if equity > hwm:
                hwm = equity

            # ---- log rich trade so Live can mirror SL/TP exactly ----
            trades.append({
                "time_open": ts,
                "time_close": feat_df.index[i + 1],

                # decision/result
                "side": int(side),
                "p_up": float(proba),
                "explored": int(explored),
                "eff_threshold": float(eff_threshold),
                "eff_risk": float(eff_risk),

                # prices & sizing (mid vs exec after costs)
                "entry_price": round(entry_mid, 6),   # mid used for sizing
                "entry_exec": round(entry_exec, 6),   # executed price after costs
                "exit_exec": round(exit_exec, 6),     # executed exit price after costs
                "units_raw": float(units),
                "units": round(units, 6),

                # risk model (so Live can compute SL/TP identically)
                "atr_used": round(bar_atr, 6),
                "stop_dist_used": round(stop_dist, 6),

                # economics
                "spread_pips": float(spread_pips),
                "slippage_pips": float(slippage_pips),
                "commission": float(commission_per_trade),

                # outcomes
                "pnl": round(pnl, 6),
                "equity_after": round(equity, 6),
            })

        # keep learning each bar
        model.partial_fit(X_full[i:i + 1], y_full[i:i + 1])

        # optional stop at profit target
        if stop_at_profit_target_pct is not None:
            if equity >= phase_start_eq * (1.0 + float(stop_at_profit_target_pct)):
                stop_reason = "profit_target"
                break

    if persist_model:
        try:
            model.save()
        except Exception:
            pass

    trades_df = pd.DataFrame(trades)
    eq_curve = _to_numpy(trades_df["equity_after"]) if not trades_df.empty else np.asarray([starting_cash], dtype=float)
    meta = {
        "hwm": hwm,
        "trailing_hwm_cap_pct": trailing_hwm_cap_pct,
        "max_trades_per_day": max_trades_per_day,
        "loss_streak_trigger": loss_streak_trigger,
        "loss_streak_pause_bars": loss_streak_pause_bars,
        "dd_adapt_threshold_pct": dd_adapt_threshold_pct,
        "dd_adapt_risk": dd_adapt_risk,
        "dd_adapt_threshold_bump": dd_adapt_threshold_bump,
        "max_intraday_dd_from_high_pct": max_intraday_dd_from_high_pct,
    }
    return BacktestResult(trades=trades_df, equity_curve=eq_curve, stop_reason=stop_reason, meta=meta)
