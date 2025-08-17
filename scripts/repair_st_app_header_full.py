from __future__ import annotations

import pathlib
import re

p = pathlib.Path("app/ui/st_app.py")
s = p.read_text()

# 0) Drop any existing sidebar caption lines anywhere (we'll reinsert once)
s = re.sub(r"^\s*st\.sidebar\.caption\(.+\)\s*$", "", s, flags=re.M)

# 1) Replace any broken emoji placeholders from earlier patches
s = s.replace("<0001f7e2>", "🟢").replace("<0001f7e0>", "🟠")

# 2) Remove ALL top-level import statements across the file (to avoid duplicates / bad order)
import_re = re.compile(r"^\s*(import\s+\w|from\s+\w)", re.M)
body = [ln for ln in s.splitlines() if not import_re.match(ln)]
body_s = "\n".join(body).lstrip("\n")

# 3) Canonical import header (stdlib → third-party → internal)
header = """\
from __future__ import annotations

# ---- stdlib
import os
import sys
import time
import json
import traceback
import contextlib
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta

# ---- third-party
import numpy as np
import pandas as pd
import streamlit as st
import toml
import matplotlib.pyplot as plt
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # optional dependency
    def st_autorefresh(*args, **kwargs):
        return None

# ---- internal modules
from ..analysis.metrics import compute_metrics
from ..analysis.prop_eval import PropConfig, evaluate_prop
from ..backtest import run_backtest, _std_ohlcv as _std_ohlcv_bt
from ..dataio.oanda import get_ohlcv_oanda, _to_oanda_instrument
from ..dataio.yf import get_ohlcv_yf
from ..utils.state import load_state, save_state
"""

# 4) Version chip (must come AFTER imports)
caption = "st.sidebar.caption(f\"Prop Copilot {os.getenv('APP_VERSION', 'dev')}\")"

# 5) Reassemble and collapse extra blank lines; ensure trailing newline
new_s = header.rstrip() + "\n\n" + caption + "\n\n" + body_s
new_s = re.sub(r"\n{3,}", "\n\n", new_s).rstrip() + "\n"

p.write_text(new_s)
print("✅ Rebuilt imports + caption in app/ui/st_app.py")
