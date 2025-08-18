from __future__ import annotations

import pathlib
import re

p = pathlib.Path("app/ui/st_app.py")
s = p.read_text()

# 1) Insert helpers once (test conn + trickle)
ANCHOR_HELP = "# --- helpers: heartbeat (auto-added) ---"
if "def _test_oanda_connection(" not in s:
    inject = """
# --- helpers: test oanda + trickle (auto-added) ---
def _test_oanda_connection(get_ohlcv_oanda, symbol: str, interval: str):
    import streamlit as st, pandas as pd
    try:
        df = get_ohlcv_oanda(symbol=symbol, interval=interval, count=10, complete_only=True)
        if df is None or len(df) == 0:
            st.error("OANDA returned no data.")
            return
        last = pd.to_datetime(df.index[-1]).tz_convert("UTC")
        price = float(df["close"].iloc[-1])
        st.success(f"OANDA OK · last bar (UTC)={last} · close={price:.2f}")
    except Exception as e:
        st.error(f"OANDA error: {type(e).__name__}: {e}")

def _trickle_effective(threshold: float, eps: float, *, quiet_bars: int, n:int, dth:float, deps:float):
    if quiet_bars >= n:
        return max(0.0, min(1.0, threshold - dth)), max(0.0, eps + deps)
    return threshold, eps
"""
    if ANCHOR_HELP in s:
        s = s.replace(ANCHOR_HELP, inject + "\n" + ANCHOR_HELP)
    else:
        s += "\n" + inject

# 2) Settings UI: connection test + trickle controls
settings_anchor = 'st.subheader("Settings")'
if settings_anchor in s and "Test OANDA connection" not in s:
    snippet = """
    with st.expander("Connections", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            if source == "OANDA":
                if st.button("Test OANDA connection", use_container_width=True):
                    _test_oanda_connection(get_ohlcv_oanda, _to_oanda_instrument(symbol), interval)
            else:
                st.caption("Using Yahoo for data (no connection test).")
        with colB:
            st.caption("Tip: Switch Data Source to OANDA in the sidebar.")

    with st.expander("Trickle mode (quiet bars nudges)", expanded=False):
        st.checkbox(
            "Enable trickle mode",
            key="trickle_on",
            value=False,
            help=(
                "After N quiet bars, temporarily shave threshold and bump epsilon. "
                "Resets after a trade."
            ),
        )
        st.number_input("Quiet bars (N)", key="trickle_n", min_value=3, max_value=200, value=12, step=1)
        st.number_input(
            "Threshold shave (Δ)",
            key="trickle_dth",
            min_value=0.0, max_value=0.5, value=0.02, step=0.01, format="%.2f"
        )
        st.number_input(
            "Epsilon bump (Δ)",
            key="trickle_deps",
            min_value=0.0, max_value=0.5, value=0.02, step=0.01, format="%.2f"
        )
"""
    s = s.replace(settings_anchor, settings_anchor + snippet)

# 3) Go Live: show effective params with trickle applied
golive_anchor = 'st.subheader("Go Live (paper signals for Copiix)")'
if golive_anchor in s and "Effective live params" not in s:
    block = """
        st.session_state.setdefault("quiet_bars", 0)

        eff_threshold, eff_eps = float(threshold), float(st.session_state.get("epsilon_start", 0.0))
        if st.session_state.get("trickle_on", False):
            eff_threshold, eff_eps = _trickle_effective(
                eff_threshold, eff_eps,
                quiet_bars=int(st.session_state.get("quiet_bars", 0)),
                n=int(st.session_state.get("trickle_n", 12)),
                dth=float(st.session_state.get("trickle_dth", 0.02)),
                deps=float(st.session_state.get("trickle_deps", 0.02)),
            )
        st.caption(
            f"Effective live params → threshold={eff_threshold:.2f}  epsilon={eff_eps:.2f}"
        )
"""
    s = s.replace(golive_anchor, golive_anchor + block)

# 4) Increment/reset quiet bars after live decision (right after you compute `res`)
if "quiet_bars" in s and "update trickle quiet bars" not in s:
    s = re.sub(
        r"(#\s*After you compute `res`.*?explain_block\(res,.*?\)\n)",
        r"\1        # update trickle quiet bars\n"
        r"        if st.session_state.get('trickle_on', False):\n"
        r"            if getattr(res, 'opened_trade', False):\n"
        r"                st.session_state['quiet_bars'] = 0\n"
        r"            else:\n"
        r"                st.session_state['quiet_bars'] = int(st.session_state.get('quiet_bars', 0)) + 1\n",
        s,
        flags=re.S,
    )

p.write_text(s)
print("Patched st_app.py ✅")
