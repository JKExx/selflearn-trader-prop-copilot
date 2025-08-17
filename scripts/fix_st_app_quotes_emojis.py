from __future__ import annotations

import pathlib
import re

p = pathlib.Path("app/ui/st_app.py")
s = p.read_text()

# Ensure `import os` after `import streamlit as st`
if re.search(r"^\s*import\s+streamlit\s+as\s+st\s*$", s, re.M) and "import os" not in s:
    s = re.sub(
        r"(^\s*import\s+streamlit\s+as\s+st\s*$)",
        r"\1\nimport os",
        s,
        count=1,
        flags=re.M,
    )

# Fix any broken sidebar caption line (remove escapes / normalize)
s = re.sub(
    r"""st\.sidebar\.caption\(.+\)""",
    "st.sidebar.caption(f\"Prop Copilot {os.getenv('APP_VERSION', 'dev')}\")",
    s,
)

# Replace broken emoji placeholders (from earlier patch attempts)
s = s.replace("<0001f7e2>", "🟢").replace("<0001f7e0>", "🟠")

# Remove accidental backslashes before quotes anywhere in that caption (belt & braces)
s = s.replace('f\\"Prop Copilot', 'f"Prop Copilot').replace("\\')", "')").replace('\\")', '")')

p.write_text(s)
print("✅ Repaired app/ui/st_app.py")
