from __future__ import annotations

import pathlib
import re

p = pathlib.Path("app/ui/st_app.py")
s = p.read_text()

# 0) Remove any broken/escaped caption lines
s = re.sub(r"^\s*st\.sidebar\.caption\(.+\)\s*$", "", s, flags=re.M)

# 1) Split into lines; strip any leading junk indentation on import lines
lines = s.splitlines()
import_re = re.compile(r"^\s*(import\s+\w|from\s+\w)")
body = []

for ln in lines:
    if import_re.match(ln):
        continue  # drop ALL existing imports (we'll rebuild cleanly)
    body.append(ln)

# 2) Canonical import header (broad but safe; includes your internal modules)
header = """\
import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import toml
from streamlit_autorefresh import st_autorefresh

# ---------- internal modules ----------
from ..analysis.metrics import compute_metrics
from ..analysis.prop_eval import PropConfig, evaluate_prop
from ..backtest import run_backtest, _std_ohlcv as _std_ohlcv_bt
from ..dataio.oanda import get_ohlcv_oanda, _to_oanda_instrument
from ..dataio.yf import get_ohlcv_yf
from ..utils.state import load_state, save_state
"""

caption = "st.sidebar.caption(f\"Prop Copilot {os.getenv('APP_VERSION', 'dev')}\")"

# 3) Fix any emoji placeholders left over from previous patches
body_s = "\n".join(body)
body_s = body_s.replace("<0001f7e2>", "🟢").replace("<0001f7e0>", "🟠")

# 4) Reassemble: header, blank, caption, blank, body; collapse extra blank lines
new_s = header.rstrip() + "\n\n" + caption + "\n\n" + body_s.lstrip()
new_s = re.sub(r"\n{3,}", "\n\n", new_s).rstrip() + "\n"

p.write_text(new_s)
print("✅ Rebuilt imports + caption in app/ui/st_app.py")
