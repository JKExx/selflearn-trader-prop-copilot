from __future__ import annotations

import re
from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()

# (A) Remove unused *_live assignments causing F841
s = re.sub(r"^\s*(threshold_live|eps_start_live|eps_final_live)\s*=.*\n", "", s, flags=re.M)

# (B) Insert an early compute block immediately before each use of threshold=eff_threshold
lines = s.splitlines(keepends=True)
needle = "threshold=eff_threshold"
injected_marker = "# --- eff params (early compute) ---"
early_block = (
    f"{injected_marker}\n"
    "try:\n"
    "    eff_threshold\n"
    "    eff_eps\n"
    "except NameError:\n"
    "    st.session_state.setdefault('quiet_bars', 0)\n"
    "    eff_threshold = float(st.session_state.get('threshold', 0.69))\n"
    "    eff_eps = float(st.session_state.get('epsilon_start', st.session_state.get('eps_start', 0.03)))\n"
    "    if st.session_state.get('trickle_on', False):\n"
    "        eff_threshold, eff_eps = _trickle_effective(\n"
    "            eff_threshold, eff_eps,\n"
    "            quiet_bars=int(st.session_state.get('quiet_bars', 0)),\n"
    "            n=int(st.session_state.get('trickle_n', 12)),\n"
    "            dth=float(st.session_state.get('trickle_dth', 0.02)),\n"
    "            deps=float(st.session_state.get('trickle_deps', 0.02)),\n"
    "        )\n"
)

out = []
i = 0
while i < len(lines):
    ln = lines[i]
    if needle in ln:
        # Check if we already injected just above
        prev = lines[i - 1] if i > 0 else ""
        if injected_marker not in prev:
            out.append(early_block + "\n")
        out.append(ln)
        i += 1
    else:
        out.append(ln)
        i += 1

new_s = "".join(out)
if new_s != s:
    p.write_text(new_s)
    print("✅ Patched: early eff_* compute and removed *_live assignments")
else:
    print("ℹ No changes (already patched)")
