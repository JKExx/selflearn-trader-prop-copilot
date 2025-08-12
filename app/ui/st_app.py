# app/ui/st_app.py
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- internal modules ----------
from ..backtest import run_backtest, _std_ohlcv as _std_ohlcv_bt
from ..analysis.metrics import compute_metrics
from ..analysis.prop_eval import PropConfig, evaluate_prop

# Data fetchers (optionals)
try:
    from ..dataio.yf import get_ohlcv as yf_get
except Exception:  # pragma: no cover
    yf_get = None

HAVE_OANDA = False
try:
    from ..dataio.oanda import get_ohlcv_oanda, _to_oanda_instrument
    HAVE_OANDA = True
except Exception:  # pragma: no cover
    HAVE_OANDA = False

try:
    from ..news import fetch_ics
except Exception:  # pragma: no cover
    fetch_ics = None

try:
    from ..alerts import send_slack, send_discord
except Exception:  # pragma: no cover
    def send_slack(*a, **k) -> bool:  # type: ignore
        return False

    def send_discord(*a, **k) -> bool:  # type: ignore
        return False


# ===================== secrets/bootstrap =====================
SECRETS_HOME = pathlib.Path.home() / ".selflearntrader" / "secrets.toml"


def _bootstrap_secrets():
    """Load OANDA creds from ~/.selflearntrader/secrets.toml or project-local secrets.toml."""
    # home
    with contextlib.suppress(Exception):
        if SECRETS_HOME.exists():
            import toml

            s = toml.load(SECRETS_HOME)
            tok = s.get("OANDA_TOKEN", "")
            env = s.get("OANDA_ENV", "PRACTICE")
            if tok:
                os.environ.setdefault("OANDA_TOKEN", tok)
                st.session_state.setdefault("OANDA_TOKEN", tok)
            os.environ.setdefault("OANDA_ENV", env)
            st.session_state.setdefault("OANDA_ENV", env)

    # project-local optional
    with contextlib.suppress(Exception):
        local = pathlib.Path("secrets.toml")
        if local.exists():
            import toml

            s = toml.load(local)
            tok = s.get("OANDA_TOKEN", "")
            env = s.get("OANDA_ENV", "PRACTICE")
            if "OANDA_TOKEN" not in os.environ and tok:
                os.environ["OANDA_TOKEN"] = tok
                st.session_state.setdefault("OANDA_TOKEN", tok)
            if "OANDA_ENV" not in os.environ:
                os.environ["OANDA_ENV"] = env
                st.session_state.setdefault("OANDA_ENV", env)


def _save_secrets_toml(env: str, token: str) -> str:
    """Persist creds to ~/.selflearntrader/secrets.toml."""
    SECRETS_HOME.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_HOME.write_text(
        f'OANDA_ENV = "{env}"\nOANDA_TOKEN = "{token}"\n',
        encoding="utf-8",
    )
    return str(SECRETS_HOME)


_bootstrap_secrets()


# ===================== misc utils =====================
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _ensure_dir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _append_trades_csv(csv_path: str, df_new: pd.DataFrame, subset_keys: list[str]):
    """Append and dedupe trade rows to csv."""
    if df_new is None or df_new.empty:
        return
    _ensure_dir(os.path.dirname(csv_path) or ".")
    try:
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            for col in ("time_open", "time_close"):
                if col in old:
                    old[col] = pd.to_datetime(old[col], utc=True, errors="coerce").astype(str)
            for col in ("time_open", "time_close"):
                if col in df_new:
                    df_new[col] = pd.to_datetime(df_new[col], utc=True, errors="coerce").astype(str)
            all_df = pd.concat([old, df_new], ignore_index=True)
        else:
            all_df = df_new.copy()
        all_df = all_df.drop_duplicates(subset=subset_keys, keep="last")
        if "time_open" in all_df:
            all_df["time_open"] = pd.to_datetime(all_df["time_open"], utc=True, errors="coerce")
            all_df = all_df.sort_values("time_open")
        if len(all_df) > 100_000:
            all_df = all_df.iloc[-100_000:]
        for col in ("time_open", "time_close"):
            if col in all_df:
                all_df[col] = pd.to_datetime(all_df[col], utc=True, errors="coerce").astype(str)
        all_df.to_csv(csv_path, index=False)
    except Exception as e:
        st.warning(f"Trade log write failed: {e}")


def _units_to_lots(instrument: str, units: int) -> float:
    ins = (instrument or "").upper()
    if ins.startswith("XAU"):  # crude, fine for gold
        return units / 100.0
    return float(units)


def _last_atr(df: pd.DataFrame, period: int = 14) -> float:
    std = _std_ohlcv_bt(df)
    hi, lo, cl = std["high"], std["low"], std["close"]
    prev_close = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return float(atr.iloc[-1])


def _fetch_data(source: str, symbol: str, interval: str, start: str, end: str | None):
    if source == "OANDA":
        if not HAVE_OANDA:
            raise RuntimeError("OANDA selected but module/env missing.")
        ins = _to_oanda_instrument(symbol)
        df = get_ohlcv_oanda(symbol=ins, interval=interval, start=start, end=end or None)
        return df, f"OANDA {ins} {interval}"
    if source == "Yahoo":
        if yf_get is None:
            raise RuntimeError("Yahoo fetcher not available.")
        df = yf_get(symbol=symbol, interval=interval, start=start, end=end or None)
        return df, f"Yahoo {symbol} {interval}"
    raise ValueError(f"Unknown source: {source}")


def _to_display_tz(ts, tz: str = "Europe/London"):
    """Safely convert any timestamp/str to the display timezone; assume UTC if naive."""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(tz)


# -------- OANDA env helpers for smart tester --------
def _oanda_base(env: str) -> str:
    env = (env or "PRACTICE").upper()
    return "https://api-fxpractice.oanda.com" if env == "PRACTICE" else "https://api-fxtrade.oanda.com"


def _try_oanda_ping(env: str, token: str, instrument: str = "XAU_USD", interval: str = "1h", rows: int = 5):
    """
    Try to fetch a few candles from OANDA using `env` + `token`.
    Returns: (rows_count, latest_index_or_None, base_url, error_str_or_None)
    """
    try:
        os.environ["OANDA_ENV"] = env
        os.environ["OANDA_TOKEN"] = token
        if not HAVE_OANDA:
            return 0, None, _oanda_base(env), "oanda module not available"
        df = get_ohlcv_oanda(
            symbol=_to_oanda_instrument(instrument),
            interval=interval,
            count=rows,
        )
        if df is None or df.empty:
            return 0, None, _oanda_base(env), None
        return len(df), df.index[-1], _oanda_base(env), None
    except Exception as e:
        return 0, None, _oanda_base(env), str(e)


# ===================== main app =====================
def main():
    st.set_page_config(page_title="SelfLearn Trader — Prop-ready", layout="wide")
    st.title("🙂 SelfLearn Trader — Prop-ready")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Credentials (OANDA)")

        env_default = st.session_state.get("OANDA_ENV") or os.environ.get("OANDA_ENV", "PRACTICE")
        tok_default = st.session_state.get("OANDA_TOKEN") or os.environ.get("OANDA_TOKEN", "")

        colA, colB = st.columns([1, 1])
        with colA:
            ui_env = st.selectbox(
                "OANDA environment (for your token)",
                ["PRACTICE", "LIVE"],
                index=0 if env_default != "LIVE" else 1,
                key="ui_oanda_env",
            )
        with colB:
            ui_tok = st.text_input(
                "API token",
                value=tok_default,
                type="password",
                placeholder="paste your token",
                key="ui_oanda_tok",
            )

        c1, c2 = st.columns([1, 1])

        # Save to machine
        if c1.button("Save to machine"):
            if not ui_tok.strip():
                st.error("Please paste a token.")
            else:
                path = _save_secrets_toml(ui_env, ui_tok.strip())
                os.environ["OANDA_ENV"] = ui_env
                os.environ["OANDA_TOKEN"] = ui_tok.strip()
                st.session_state["OANDA_ENV"] = ui_env
                st.session_state["OANDA_TOKEN"] = ui_tok.strip()
                st.success(f"Saved to {path} — ready to use.")

        # Smart Test: try selected env first, then the other
        if c2.button("Test connection"):
            token = (ui_tok or "").strip()
            if not token:
                st.error("Paste a token first.")
            else:
                ordered = [ui_env, ("LIVE" if ui_env == "PRACTICE" else "PRACTICE")]
                lines = []
                worked = None
                latest_idx = None

                for env_try in ordered:
                    rows, latest_idx, base, err = _try_oanda_ping(env_try, token)
                    if err:
                        lines.append(f"{env_try:<9} → error: {err}  (base={base})")
                    else:
                        lines.append(f"{env_try:<9} → {rows} row(s)  (base={base})")
                        if rows > 0:
                            worked = env_try
                            break

                if worked:
                    st.success(f"OK on {worked} — latest bar {latest_idx}")
                    st.caption("Environment saved for this session.")
                    os.environ["OANDA_ENV"] = worked
                    st.session_state["OANDA_ENV"] = worked
                else:
                    st.warning("Connected, but 0 rows on both envs. Market closed, instrument disabled in this env, or permissions issue.")
                st.code("\n".join(lines))

        # ----- Data -----
        st.header("Data")
        sources = ["OANDA"] if HAVE_OANDA else ["Yahoo"]
        source = st.selectbox("Data source", sources, index=0, key="source")
        default_symbol = "XAU_USD" if source == "OANDA" else "XAUUSD=X"
        symbol = st.text_input("Symbol", value=st.session_state.get("symbol", default_symbol), key="symbol")
        interval = st.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=2, key="interval")
        start = st.text_input("Data start (UTC)", value=st.session_state.get("start", "2025-01-01"), key="start")
        end = st.text_input("Data end (optional)", value=st.session_state.get("end", ""), key="end")

        # ---- Display (timezone) ----
        st.header("Display")

        curr_tz = st.session_state.get("display_tz", "Europe/London")
        colT1, colT2, colT3, colT4 = st.columns([1, 1, 1, 2])

        def _set_tz_and_rerun(tz: str):
            st.session_state["display_tz"] = tz
            os.environ["DISPLAY_TZ"] = tz
            _rerun()

        with colT1:
            if st.button("🇬🇧 UK"):
                _set_tz_and_rerun("Europe/London")
        with colT2:
            if st.button("UTC"):
                _set_tz_and_rerun("UTC")
        with colT3:
            if st.button("🇺🇸 NY"):
                _set_tz_and_rerun("America/New_York")
        with colT4:
            custom_tz = st.text_input("Custom TZ (IANA)", value=curr_tz, key="custom_tz")
            if st.button("Apply TZ"):
                _set_tz_and_rerun(custom_tz or "Europe/London")
        st.caption(f"Showing times in: **{st.session_state.get('display_tz', 'Europe/London')}** (calcs remain UTC)")

        # ----- Decision & Learning -----
        st.header("Decision & Learning")
        threshold = st.slider("Decision threshold", 0.50, 0.75, st.session_state.get("threshold", 0.69), 0.01, key="threshold")
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            eps_start = st.number_input("Epsilon start", 0.0, 1.0, st.session_state.get("eps_start", 0.03), 0.01, key="eps_start")
        with colB:
            eps_final = st.number_input("Epsilon final", 0.0, 1.0, st.session_state.get("eps_final", 0.00), 0.01, key="eps_final")
        with colC:
            eps_decay_trades = st.number_input("Epsilon decay trades", 0, 5000, st.session_state.get("eps_decay_trades", 75), 25, key="eps_decay_trades")

        trade_start = st.text_input("Trade start date (UTC, optional)", value=st.session_state.get("trade_start", ""), key="trade_start")
        min_samples = st.number_input("Warmup bars (for scaler init)", 50, 10000, st.session_state.get("min_samples", 1000), 50, key="min_samples")

        # ----- Risk & Costs -----
        st.header("Risk & Costs")
        starting_cash = st.number_input("Starting cash", 1000.0, 5_000_000.0, st.session_state.get("starting_cash", 10_000.0), 1000.0, key="starting_cash")
        risk_per_trade_pct = st.number_input("Risk per trade (%)", 0.1, 5.0, st.session_state.get("risk_per_trade_pct", 1.0), 0.1, key="risk_per_trade_pct")
        atr_mult = st.number_input("ATR multiple for stop", 0.2, 5.0, st.session_state.get("atr_mult", 1.0), 0.1, key="atr_mult")
        spread_pips = st.number_input("Spread (pips, RT)", 0.0, 10.0, st.session_state.get("spread_pips", 0.20), 0.05, key="spread_pips")
        slippage_pips = st.number_input("Slippage (pips each side)", 0.0, 10.0, st.session_state.get("slippage_pips", 0.10), 0.05, key="slippage_pips")
        commission = st.number_input("Commission per trade (quote ccy)", 0.0, 100.0, st.session_state.get("commission", 0.0), 0.1, key="commission")

        # ----- Prop Mode -----
        st.header("Prop Mode (enforcement)")
        daily_loss_pct = st.number_input("Daily loss cap %", 0.0, 20.0, st.session_state.get("daily_loss_pct", 4.5), 0.1, key="daily_loss_pct")
        overall_loss_pct = st.number_input("Overall loss cap %", 0.0, 50.0, st.session_state.get("overall_loss_pct", 10.0), 0.1, key="overall_loss_pct")
        stop_target_on = st.checkbox("Stop at profit target", value=st.session_state.get("stop_target_on", True), key="stop_target_on")
        stop_target_pct = (st.number_input("Profit target %", 0.0, 100.0, st.session_state.get("stop_target_pct", 8.0), 0.5, key="stop_target_pct") / 100.0) if stop_target_on else None
        friday_cutoff_local = int(st.number_input("Friday cutoff hour (New York local)", 0, 23, int(st.session_state.get("friday_cutoff_local", 22)), 1, key="friday_cutoff_local"))

        # ----- Gating & Persistence -----
        st.header("Gating & Persistence")
        gate_kz = st.checkbox("Only trade in London/NY killzones (UTC)", value=st.session_state.get("gate_kz", True), key="gate_kz")
        persist_model = st.checkbox("Persist model between runs", value=st.session_state.get("persist_model", False), key="persist_model")

        # ----- News -----
        with st.expander("News filter (toggle/ICS)"):
            news_on = st.checkbox("Enable news blackout", value=st.session_state.get("news_on", False), key="news_on")
            ics_url = st.text_input("ICS / calendar URL", value=st.session_state.get("ics_url", ""), key="ics_url", placeholder="webcal:// or https:// …")
            kws_default = st.session_state.get("news_keywords", "USD;High;FOMC;CPI;NFP")
            st.text_input("Keywords (semicolon separated)", value=kws_default, key="news_keywords")
            st.slider("Blackout ± minutes", min_value=0, max_value=15, value=st.session_state.get("news_window", 2), key="news_window")
            if news_on and ics_url and st.button("Test fetch ICS"):
                if fetch_ics is None:
                    st.warning("ICS parser not available.")
                else:
                    try:
                        events = fetch_ics(ics_url)
                        st.success(f"Fetched {len(events)} events.")
                        st.write([{"start": str(e["start"]), "end": str(e["end"]), "summary": e["summary"][:80]} for e in events[:30]])
                    except Exception as e:
                        st.error(f"{e}")

        # ----- Advanced guardrails -----
        with st.expander("Advanced guardrails", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                trailing_hwm_enabled = st.checkbox("Enable trailing HWM cap", value=True, key="trailing_hwm_enabled")
                trailing_hwm_cap_pct = (st.number_input("Trailing HWM cap % (stop)", 0.0, 20.0, 3.0, 0.5, key="trailing_hwm_cap_pct_val") / 100.0) if trailing_hwm_enabled else None
                max_trades_per_day = st.number_input("Max trades per day", 0, 50, 5, 1)
                max_trades_per_day = int(max_trades_per_day) if max_trades_per_day > 0 else None
                intraday_dd_enabled = st.checkbox("Enable max intraday DD from day high", value=True, key="intraday_dd_enabled")
                max_intraday_dd_from_high_pct = (st.number_input("Max intraday DD from day high % (lock day)", 0.5, 20.0, 3.0, 0.5, key="intraday_dd_val") / 100.0) if intraday_dd_enabled else None
            with col2:
                loss_streak_trigger = st.number_input("Loss streak trigger (N losses)", 0, 10, 2, 1)
                loss_streak_pause_bars = st.number_input("Pause bars after loss streak", 0, 100, 4, 1)
                dd_adapt_enabled = st.checkbox("Enable drawdown adapt mode", value=True, key="dd_adapt_enabled")
                dd_adapt_threshold_pct = (st.number_input("Adapt mode below HWM %", 0.5, 20.0, 2.0, 0.5, key="dd_adapt_thresh_val") / 100.0) if dd_adapt_enabled else None
                dd_adapt_risk = (st.number_input("Risk in adapt mode %", 0.1, 5.0, 0.5, 0.1, key="dd_adapt_risk_val") / 100.0) if dd_adapt_enabled else None
                dd_adapt_threshold_bump = (st.number_input("Threshold bump in adapt", 0.0, 0.1, 0.01, 0.01, key="dd_adapt_bump_val")) if dd_adapt_enabled else 0.0

        # ----- Presets -----
        st.header("Presets")
        os.makedirs("presets", exist_ok=True)

        def _save_preset(name: str, payload: dict):
            with open(os.path.join("presets", f"{name}.json"), "w") as f:
                json.dump(payload, f, indent=2)

        def _load_preset(name: str) -> dict:
            with open(os.path.join("presets", f"{name}.json")) as f:
                return json.load(f)

        preset_name = st.text_input("Preset name", st.session_state.get("preset_name", "GOAT_XAU_1h_069"), key="preset_name")
        colP1, colP2, colP3 = st.columns([1, 1, 1])
        with colP1:
            if st.button("💾 Save preset"):
                payload = {
                    "symbol": symbol,
                    "interval": interval,
                    "threshold": float(threshold),
                    "epsilon_start": float(eps_start),
                    "epsilon_final": float(eps_final),
                    "epsilon_decay_trades": int(eps_decay_trades),
                    "warmup_bars": int(min_samples),
                    "trade_start_date": trade_start,
                    "starting_cash": float(starting_cash),
                    "risk_per_trade": float(risk_per_trade_pct / 100.0),
                    "atr_multiple": float(atr_mult),
                    "spread_pips": float(spread_pips),
                    "slippage_pips": float(slippage_pips),
                    "commission_per_trade": float(commission),
                    "gate_killzones": bool(gate_kz),
                    "persist_model": bool(persist_model),
                    "enforce_daily_cap": True,
                    "daily_cap_pct": float(daily_loss_pct / 100.0),
                    "enforce_overall_cap": True,
                    "overall_cap_pct": float(overall_loss_pct / 100.0),
                    "friday_cutoff_local": int(friday_cutoff_local),
                    "stop_at_profit_target_pct": (float(stop_target_pct) if stop_target_pct is not None else None),
                    # news
                    "news_on": bool(st.session_state.get("news_on", False)),
                    "ics_url": st.session_state.get("ics_url", ""),
                    "news_keywords": st.session_state.get("news_keywords", ""),
                    "news_blackout_minutes": int(st.session_state.get("news_window", 0)),
                    # guardrails
                    "trailing_hwm_cap_pct": float(trailing_hwm_cap_pct) if trailing_hwm_enabled else None,
                    "max_trades_per_day": (int(max_trades_per_day) if max_trades_per_day else None),
                    "loss_streak_trigger": int(loss_streak_trigger),
                    "loss_streak_pause_bars": int(loss_streak_pause_bars),
                    "dd_adapt_threshold_pct": float(dd_adapt_threshold_pct) if dd_adapt_enabled else None,
                    "dd_adapt_risk": float(dd_adapt_risk) if dd_adapt_enabled else None,
                    "dd_adapt_threshold_bump": float(dd_adapt_threshold_bump) if dd_adapt_enabled else 0.0,
                    "max_intraday_dd_from_high_pct": float(max_intraday_dd_from_high_pct) if intraday_dd_enabled else None,
                }
                _save_preset(preset_name, payload)
                st.success(f"Saved → presets/{preset_name}.json")
        with colP2:
            preset_files = [""] + [p[:-5] for p in os.listdir("presets") if p.endswith(".json")]
            selected = st.selectbox("Load preset", options=preset_files, index=0, key="preset_select")
        with colP3:
            if st.button("⤵️ Apply loaded preset"):
                if selected:
                    try:
                        data = _load_preset(selected)
                        st.session_state["_pending_preset"] = data  # stage for next run
                        st.success(f"Applied preset '{selected}'. Reloading UI…")
                        _rerun()
                    except Exception as e:
                        st.error(f"Failed to stage preset: {e}")

    # Keep as floats
    risk_per_trade = st.session_state.get("risk_per_trade_pct", 1.0) / 100.0

    # baseline editable BEFORE Start (no local var to keep ruff happy)
    with st.sidebar:
        st.date_input(
            "Compliance baseline (start date)",
            st.session_state.get("compliance_baseline", pd.Timestamp.utcnow().date()),
            key="compliance_baseline",
        )

    tabs = st.tabs(["Backtest", "Metrics", "Sweep", "Trades", "Compliance", "Go Live", "Debug"])

    # ---------------- Backtest ----------------
    with tabs[0]:
        if st.button("Run Backtest", type="primary"):
            try:
                with st.spinner("Fetching data and running backtest..."):
                    df, used = _fetch_data(source, symbol, interval, start, end)
                    st.caption(f"Using {used} | Range: {df.index[0].date()} → {df.index[-1].date()} (UTC)")

                    news_events_to_pass = None
                    news_keywords_list = None
                    if st.session_state.get("news_on") and st.session_state.get("ics_url") and fetch_ics:
                        try:
                            news_events_to_pass = fetch_ics(st.session_state["ics_url"])
                            news_keywords_list = [k.strip() for k in (st.session_state.get("news_keywords") or "").split(";")]
                        except Exception as e:
                            st.warning(f"News fetch skipped: {e}")

                    res = run_backtest(
                        df,
                        symbol=symbol,
                        interval=interval,
                        threshold=st.session_state.get("threshold", 0.69),
                        epsilon_start=st.session_state.get("eps_start", 0.03),
                        epsilon_final=st.session_state.get("eps_final", 0.00),
                        epsilon_decay_trades=int(st.session_state.get("eps_decay_trades", 75)),
                        warmup_bars=int(st.session_state.get("min_samples", 1000)),
                        trade_start_date=st.session_state.get("trade_start") or None,
                        starting_cash=float(st.session_state.get("starting_cash", 10_000.0)),
                        risk_per_trade=risk_per_trade,
                        atr_multiple=float(st.session_state.get("atr_mult", 1.0)),
                        spread_pips=float(st.session_state.get("spread_pips", 0.2)),
                        slippage_pips=float(st.session_state.get("slippage_pips", 0.1)),
                        commission_per_trade=float(st.session_state.get("commission", 0.0)),
                        gate_killzones=bool(st.session_state.get("gate_kz", True)),
                        persist_model=bool(st.session_state.get("persist_model", False)),
                        enforce_overall_cap=True,
                        overall_cap_pct=float(st.session_state.get("overall_loss_pct", 10.0)) / 100.0,
                        enforce_daily_cap=True,
                        daily_cap_pct=float(st.session_state.get("daily_loss_pct", 4.5)) / 100.0,
                        day_boundary_tz="America/New_York",
                        friday_cutoff_local=int(st.session_state.get("friday_cutoff_local", 22)),
                        stop_at_profit_target_pct=(
                            float(st.session_state.get("stop_target_pct", 8.0)) / 100.0
                            if st.session_state.get("stop_target_on", True)
                            else None
                        ),
                        # news
                        news_events=news_events_to_pass,
                        news_keywords=news_keywords_list,
                        news_blackout_minutes=int(st.session_state.get("news_window", 0))
                        if st.session_state.get("news_on")
                        else 0,
                        # guardrails
                        trailing_hwm_cap_pct=st.session_state.get("trailing_hwm_cap_pct")
                        if st.session_state.get("trailing_hwm_enabled", True)
                        else None,
                        max_trades_per_day=int(st.session_state.get("max_trades_per_day") or 0) or None,
                        loss_streak_pause_bars=int(st.session_state.get("loss_streak_pause_bars", 4)),
                        loss_streak_trigger=int(st.session_state.get("loss_streak_trigger", 2)),
                        dd_adapt_threshold_pct=st.session_state.get("dd_adapt_threshold_pct")
                        if st.session_state.get("dd_adapt_enabled", True)
                        else None,
                        dd_adapt_risk=st.session_state.get("dd_adapt_risk")
                        if st.session_state.get("dd_adapt_enabled", True)
                        else None,
                        dd_adapt_threshold_bump=st.session_state.get("dd_adapt_threshold_bump", 0.0)
                        if st.session_state.get("dd_adapt_enabled", True)
                        else 0.0,
                        max_intraday_dd_from_high_pct=st.session_state.get("max_intraday_dd_from_high_pct")
                        if st.session_state.get("intraday_dd_enabled", True)
                        else None,
                    )

                st.session_state["last_result"] = res
                st.session_state["last_trades"] = res.trades.copy()
                stop_reason = res.stop_reason or "end_of_data"
                stop_ts = getattr(res, "stop_ts", None) or (df.index[-1] if len(df) else None)
                st.success(f"Backtest complete. Stop reason: {stop_reason}")
                if stop_ts is not None:
                    st.caption(f"Stopped at: {stop_ts} UTC")

                # autosave
                try:
                    ts_tag = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    out_dir = "logs/backtests"
                    _ensure_dir(out_dir)
                    out_path = os.path.join(out_dir, f"{ts_tag}_{symbol}_{interval}.csv")
                    res.trades.to_csv(out_path, index=False)
                    st.caption(f"Saved backtest trades → {out_path}")
                except Exception as e:
                    st.warning(f"Could not save backtest log: {e}")

                if not res.trades.empty:
                    fig = plt.figure()
                    plt.plot(res.equity_curve)
                    plt.xlabel("Trade #")
                    plt.ylabel("Equity")
                    st.pyplot(fig)
                else:
                    st.info("No trades executed (check gates/threshold/warmup).")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.code(traceback.format_exc())

    # ---------------- Metrics ----------------
    with tabs[1]:
        st.subheader("Performance Metrics")
        tr = st.session_state.get("last_trades")
        if tr is None or tr.empty:
            st.info("Run a backtest first.")
        else:
            st.json(compute_metrics(tr))

    # ---------------- Sweep ----------------
    with tabs[2]:
        st.subheader("Threshold Sweep")
        tr = st.session_state.get("last_trades")
        if tr is None or tr.empty:
            st.info("Run a backtest first.")
        else:
            df_tr = tr.copy()
            ts = np.round(np.linspace(0.50, 0.72, 12), 3)
            rows = []
            for t in ts:
                mask = ((df_tr["side"] == 1) & (df_tr["p_up"] >= t)) | ((df_tr["side"] == -1) & (df_tr["p_up"] <= (1 - t)))
                sub = df_tr[mask]
                wins = sub["pnl"][sub["pnl"] > 0].sum()
                losses = sub["pnl"][sub["pnl"] < 0].sum()
                pf = (wins / abs(losses)) if float(losses) != 0 else float("inf")
                rows.append(
                    {
                        "threshold": float(t),
                        "trades": int(len(sub)),
                        "total_pnl": float(sub["pnl"].sum()),
                        "profit_factor": (float("inf") if pf == float("inf") else float(pf)),
                    }
                )
            sweep = pd.DataFrame(rows)
            st.dataframe(sweep, use_container_width=True)
            fig = plt.figure()
            plt.plot(sweep["threshold"], sweep["total_pnl"])
            plt.xlabel("Threshold")
            plt.ylabel("Total PnL")
            st.pyplot(fig)

    # ---------------- Trades ----------------
    with tabs[3]:
        st.subheader("Trades")
        tr = st.session_state.get("last_trades")
        if tr is None or tr.empty:
            st.info("Run a backtest first.")
        else:
            st.dataframe(tr.head(1000), use_container_width=True)
            st.download_button("Download trades.csv", tr.to_csv(index=False), "trades.csv")

    # ---------------- Compliance ----------------
    with tabs[4]:
        st.subheader("Prop Compliance (Evaluation)")
        tr = st.session_state.get("last_trades")
        if tr is None or tr.empty:
            st.info("Run a backtest first.")
        else:
            pcfg = PropConfig(
                daily_loss_pct=float(st.session_state.get("daily_loss_pct", 4.5)),
                overall_loss_pct=float(st.session_state.get("overall_loss_pct", 10.0)),
                profit_target_pct=None,
                day_boundary_tz="America/New_York",
                friday_cutoff_hour_local=int(st.session_state.get("friday_cutoff_local", 22)),
                news_blackout_mins=int(st.session_state.get("news_window", 0)) if st.session_state.get("news_on") else 0,
                relevant_currencies=set(),
            )
            board, daily, violations = evaluate_prop(tr, pcfg, news_df=None)
            left, right = st.columns([1, 1])
            with left:
                st.json(board)
            with right:
                if not daily.empty:
                    st.dataframe(daily[["prop_day", "day_min_dd_pct", "breached_daily"]], use_container_width=True)

    # ---------------- Go Live ----------------
    with tabs[5]:
        st.subheader("Go Live (paper signals for Copiix)")

        slack_url = st.text_input("Slack webhook URL (optional)", value=st.session_state.get("slack_url", ""))
        discord_url = st.text_input("Discord webhook URL (optional)", value=st.session_state.get("discord_url", ""))

        parity_mode = st.checkbox("Backtest parity mode (use exact params/ATR)", value=True)
        st.session_state["tp_r_multiple"] = st.number_input("TP multiple (R)", 0.1, 5.0, 1.0, 0.1)

        st.subheader("Live logging")
        log_on = st.checkbox("Enable trade logging", value=True, key="live_log_on")
        log_dir = st.text_input("Log directory", value=st.session_state.get("log_dir", "logs"), key="log_dir")

        colL, colR = st.columns([1, 1])
        running = st.session_state.get("live_running", False)
        with colL:
            if not running and st.button("▶️ Start"):
                st.session_state["live_running"] = True
                st.session_state["live_started_ts"] = pd.Timestamp.now(tz="UTC")
                _rerun()
        with colR:
            if running and st.button("⏹ Stop"):
                st.session_state["live_running"] = False
                _rerun()

        if st.session_state.get("live_running", False):
            try:
                # fetch fresh data (OANDA preferred)
                live_source = "OANDA" if HAVE_OANDA else source
                if live_source == "OANDA":
                    df_live = get_ohlcv_oanda(
                        symbol=_to_oanda_instrument(symbol),
                        interval=interval,
                        start=None,
                        end=None,
                        count=5000,
                    )
                    used = f"OANDA {_to_oanda_instrument(symbol)} {interval}"
                else:
                    df_live, used = _fetch_data(live_source, symbol, interval, start, None)

                if df_live is None or df_live.empty:
                    st.warning("No live data returned (market closed or token issue).")
                    st.stop()

                std = _std_ohlcv_bt(df_live)
                last_ts = std.index[-1]
                now = pd.Timestamp.now(tz="UTC")
                bar_minutes = {"15m": 15, "30m": 30, "1h": 60, "1d": 1440}.get(interval, 60)

                # last completed bar label
                expected_last = now.floor(f"{bar_minutes}min")
                if interval != "1d":
                    expected_last -= pd.Timedelta(minutes=bar_minutes)

                delay_min = float((expected_last - last_ts) / pd.Timedelta(minutes=1))

                disp_tz = st.session_state.get("display_tz", "Europe/London")
                now_local = _to_display_tz(now, disp_tz)
                last_local = _to_display_tz(last_ts, disp_tz)
                expected_local = _to_display_tz(expected_last, disp_tz)

                # Heartbeat + dynamic auto-refresh
                st.caption(
                    f"data_used={used} last_ts={last_local} expected_last={expected_local} "
                    f"now={now_local} delay_min={delay_min:.1f}"
                )
                try:
                    from streamlit_autorefresh import st_autorefresh

                    next_open = now.floor(f"{bar_minutes}min") + pd.Timedelta(minutes=bar_minutes)
                    sec_to_next = max(1, int((next_open - now).total_seconds()))
                    refresh_ms = int(min(120_000, max(5_000, (sec_to_next + 2) * 1000)))
                    st_autorefresh(interval=refresh_ms, key="live_refresh_dyn")
                    st.caption(f"Heartbeat: next auto-check in ~{sec_to_next}s")
                except Exception:
                    st.caption("Heartbeat: auto-refresh unavailable (install streamlit-autorefresh).")

                if st.button("🔄 Check now"):
                    _rerun()

                lag_threshold = max(5, bar_minutes * 0.5)
                if delay_min > lag_threshold:
                    st.info(
                        f"Data looks stale (last={last_local}, expected={expected_local}). "
                        f"Market closed or provider lag."
                    )
                    st.stop()

                # live params (parity vs adaptive)
                if parity_mode:
                    eps_start_live = 0.0
                    eps_final_live = 0.0
                    eps_decay_live = 0
                    persist_live = st.session_state.get("persist_model", False)
                else:
                    eps_start_live = st.session_state.get("eps_start", 0.03)
                    eps_final_live = st.session_state.get("eps_final", 0.00)
                    eps_decay_live = int(st.session_state.get("eps_decay_trades", 75))
                    persist_live = True

                # soft stops in live (simulate last closed bar)
                res = run_backtest(
                    df_live,
                    symbol=symbol,
                    interval=interval,
                    threshold=st.session_state.get("threshold", 0.69),
                    epsilon_start=eps_start_live,
                    epsilon_final=eps_final_live,
                    epsilon_decay_trades=eps_decay_live,
                    warmup_bars=int(st.session_state.get("min_samples", 1000)),
                    trade_start_date=st.session_state.get("trade_start") or None,
                    starting_cash=float(st.session_state.get("starting_cash", 10_000.0)),
                    risk_per_trade=(st.session_state.get("risk_per_trade_pct", 1.0) / 100.0),
                    atr_multiple=float(st.session_state.get("atr_mult", 1.0)),
                    spread_pips=float(st.session_state.get("spread_pips", 0.2)),
                    slippage_pips=float(st.session_state.get("slippage_pips", 0.1)),
                    commission_per_trade=float(st.session_state.get("commission", 0.0)),
                    gate_killzones=bool(st.session_state.get("gate_kz", True)),
                    persist_model=persist_live,
                    enforce_overall_cap=False,
                    enforce_daily_cap=False,
                    stop_at_profit_target_pct=None,
                    trailing_hwm_cap_pct=None,
                    max_intraday_dd_from_high_pct=None,
                    day_boundary_tz="America/New_York",
                    friday_cutoff_local=int(st.session_state.get("friday_cutoff_local", 22)),
                )

                # debug context (display in chosen tz)
                warmup = int(st.session_state.get("min_samples", 1000))
                first_tradable = _to_display_tz(std.index[warmup], disp_tz) if len(std) > warmup else None
                last_trade_time = (
                    _to_display_tz(pd.to_datetime(res.trades["time_open"].iloc[-1]), disp_tz)
                    if hasattr(res, "trades") and not res.trades.empty
                    else None
                )
                st.caption(
                    f"live bars={len(std)} warmup={warmup} first_tradable={first_tradable} "
                    f"last_trade_time={last_trade_time} stop_reason={getattr(res, 'stop_reason', None)}"
                )

                # ---- Compliance (soft evaluation) ----
                pcfg = PropConfig(
                    daily_loss_pct=float(st.session_state.get("daily_loss_pct", 4.5)),
                    overall_loss_pct=float(st.session_state.get("overall_loss_pct", 10.0)),
                    profit_target_pct=None,
                    day_boundary_tz="America/New_York",
                    friday_cutoff_hour_local=int(st.session_state.get("friday_cutoff_local", 22)),
                    news_blackout_mins=int(st.session_state.get("news_window", 0)) if st.session_state.get("news_on") else 0,
                    relevant_currencies=set(),
                )

                # Safe slice of trades since baseline
                base = st.session_state.get("compliance_baseline", pd.Timestamp.utcnow().date())
                trades_df = getattr(res, "trades", None)
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and "time_close" in trades_df.columns:
                    tr_eval = trades_df[pd.to_datetime(trades_df["time_close"], utc=True).dt.date >= base]
                else:
                    tr_eval = pd.DataFrame(columns=["time_open", "time_close", "pnl"])

                board, daily, _ = evaluate_prop(tr_eval, pcfg, news_df=None)

                # Today in NY (prop definition)
                today_local = std.index[-1].tz_convert("America/New_York").strftime("%Y-%m-%d")

                # Robust extraction of "locked today" and overall breach
                locked_today = False
                if isinstance(daily, pd.DataFrame) and not daily.empty:
                    if "prop_day" in daily.columns and "breached_daily" in daily.columns:
                        row_today = daily.loc[daily["prop_day"] == today_local]
                        if not row_today.empty:
                            locked_today = bool(row_today["breached_daily"].iloc[0])

                overall_breached = bool(board.get("breaches_overall", 0))
                max_dd = float(board.get("max_drawdown_pct", 0.0))

                disp_tz = st.session_state.get("display_tz", "Europe/London")
                st.caption(
                    f"compliance: locked_today={locked_today} overall_breached={overall_breached} "
                    f"max_dd={max_dd:.2f}% | shown in {disp_tz}"
                )

                require_ok = st.checkbox("Require compliance to signal", value=False)
                if require_ok and (overall_breached or locked_today):
                    st.warning("Compliance lock: signals suppressed (overall or daily breach).")
                    st.stop()

                # last-bar signal + notify + LOG
                tr_last = res.trades.tail(1)
                if tr_last.empty or int(tr_last.iloc[0]["side"]) == 0:
                    last_close = std["close"].iloc[-1]
                    last_sig_local = last_trade_time if last_trade_time else "—"
                    st.write(
                        "No new trade on last completed bar "
                        f"(last signal was {last_sig_local}, last completed bar is {last_local})."
                    )
                    st.caption(f"{used} | last bar close {last_close:.2f}")
                else:
                    row = tr_last.iloc[0]
                    side = int(row["side"])
                    last_bar_ts = std.index[-1]
                    trade_ts = pd.to_datetime(str(row["time_open"]))
                    if trade_ts != last_bar_ts:
                        st.write(
                            f"No new trade on last bar (last signal was {_to_display_tz(trade_ts, disp_tz)}, "
                            f"last bar is {_to_display_tz(last_bar_ts, disp_tz)})."
                        )
                        st.caption(f"{used} | last bar close {std['close'].iloc[-1]:.2f}")
                    else:
                        entry = float(row.get("entry_price", None) or std["close"].iloc[-1])
                        stop_dist = float(row.get("stop_dist_used", 0.0))
                        if not stop_dist:
                            atr_val = float(row.get("atr_used", 0.0)) or _last_atr(df_live)
                            stop_dist = atr_val * float(st.session_state.get("atr_mult", 1.0))

                        r_mult = float(st.session_state.get("tp_r_multiple", 1.0))
                        if side == 1:
                            sl = entry - stop_dist
                            tp = entry + stop_dist * r_mult
                            dir_txt = "LONG"
                        else:
                            sl = entry + stop_dist
                            tp = entry - stop_dist * r_mult
                            dir_txt = "SHORT"

                        units_rounded = int(round(float(row["units"])))
                        lots = _units_to_lots(symbol, units_rounded)

                        last_ts_sent = st.session_state.get("last_signal_ts")
                        is_new_bar = str(last_bar_ts) != str(last_ts_sent)
                        st.session_state["last_signal_ts"] = str(last_bar_ts)

                        msg = (
                            f"[Signal] {symbol} {interval} | {dir_txt} {units_rounded}u (~{lots:.2f} lots) "
                            f"@ {entry:.2f}  SL {sl:.2f}  TP {tp:.2f}  (p_up={row['p_up']:.2f})"
                        )
                        st.code(msg)
                        with st.expander("Last trade (raw)"):
                            st.json({k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()})

                        if is_new_bar:
                            if slack_url:
                                send_slack(slack_url, msg)
                            if discord_url:
                                send_discord(discord_url, msg)

                            if log_on:
                                rec = {
                                    "symbol": symbol,
                                    "interval": interval,
                                    "time_open": str(pd.Timestamp(trade_ts).tz_convert("UTC")),
                                    "time_close": str(row.get("time_close", "")),
                                    "side": side,
                                    "p_up": float(row.get("p_up", np.nan)),
                                    "units": units_rounded,
                                    "lots": float(lots),
                                    "entry_price": float(entry),
                                    "sl": float(sl),
                                    "tp": float(tp),
                                    "atr_used": float(row.get("atr_used", np.nan))
                                    if pd.notna(row.get("atr_used", np.nan))
                                    else np.nan,
                                    "stop_dist_used": float(stop_dist),
                                    "pnl": float(row.get("pnl", np.nan))
                                    if pd.notna(row.get("pnl", np.nan))
                                    else np.nan,
                                }
                                df_one = pd.DataFrame([rec])
                                _ensure_dir(log_dir)
                                live_csv = os.path.join(log_dir, "live_trades.csv")
                                _append_trades_csv(live_csv, df_one, ["symbol", "interval", "time_open"])
                                day_tag = pd.Timestamp(trade_ts).tz_convert("UTC").strftime("%Y-%m-%d")
                                _append_trades_csv(
                                    os.path.join(log_dir, f"live_trades_{day_tag}.csv"),
                                    df_one,
                                    ["symbol", "interval", "time_open"],
                                )

                # live metrics since Start
                with st.expander("Live run quick metrics (since Start)"):
                    try:
                        tr_all = res.trades.copy()
                        tr_all["time_close"] = pd.to_datetime(tr_all["time_close"], utc=True)
                        live_start = st.session_state.get("live_started_ts")
                        tr_since = tr_all[tr_all["time_close"] >= live_start] if live_start else tr_all.iloc[0:0]
                        st.json(compute_metrics(tr_since) if not tr_since.empty else {"trades": 0})
                    except Exception:
                        pass

                # viewer
                with st.expander("Live log (last 50)"):
                    try:
                        live_csv = os.path.join(log_dir, "live_trades.csv")
                        if os.path.exists(live_csv):
                            log_df = pd.read_csv(live_csv)
                            st.dataframe(log_df.tail(50), use_container_width=True)
                            st.download_button("Download live_trades.csv", log_df.to_csv(index=False), "live_trades.csv")
                        else:
                            st.caption("No live_trades.csv yet.")
                    except Exception as e:
                        st.warning(f"Could not read live log: {e}")

            except Exception as e:
                st.error(f"Live stream failed: {e}")
                st.code(traceback.format_exc())

    # ---------------- Debug ----------------
    with tabs[6]:
        st.subheader("Debug Info")
        st.write(
            {
                "source": st.session_state.get("source"),
                "symbol": st.session_state.get("symbol"),
                "interval": st.session_state.get("interval"),
                "threshold": st.session_state.get("threshold"),
                "epsilon_start": st.session_state.get("eps_start"),
                "epsilon_final": st.session_state.get("eps_final"),
                "epsilon_decay_trades": st.session_state.get("eps_decay_trades"),
                "trade_start_date": st.session_state.get("trade_start"),
                "warmup_bars": st.session_state.get("min_samples"),
                "risk_per_trade(%)": st.session_state.get("risk_per_trade_pct"),
                "atr_mult": st.session_state.get("atr_mult"),
                "spread_pips": st.session_state.get("spread_pips"),
                "slippage_pips": st.session_state.get("slippage_pips"),
                "commission": st.session_state.get("commission"),
                "gate_killzones": st.session_state.get("gate_kz"),
                "persist_model": st.session_state.get("persist_model"),
                "daily_cap%": st.session_state.get("daily_loss_pct"),
                "overall_cap%": st.session_state.get("overall_loss_pct"),
                "stop_target_on": st.session_state.get("stop_target_on"),
                "stop_target_pct%": st.session_state.get("stop_target_pct"),
                "OANDA_ENV": os.environ.get("OANDA_ENV"),
                "OANDA_TOKEN_set": bool(os.environ.get("OANDA_TOKEN")),
                "DISPLAY_TZ": st.session_state.get("display_tz", "Europe/London"),
            }
        )


if __name__ == "__main__":
    main()
