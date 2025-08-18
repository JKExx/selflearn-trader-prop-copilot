import re
from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()

# 1) Remove any old "Effective live params" caption blocks (we'll reinsert once)
s = re.sub(r'\n\s*st\.caption\(f"Effective live params[^"]+"\)\n?', "\n", s)

# 2) Kill unused *_live assignments causing F841
s = re.sub(r"^\s*threshold_live\s*=.*\n", "", s, flags=re.M)
s = re.sub(r"^\s*eps_start_live\s*=.*\n", "", s, flags=re.M)
s = re.sub(r"^\s*eps_final_live\s*=.*\n", "", s, flags=re.M)

# 3) After the Go Live header, inject effective params block (once, early)
anchor = 'st.subheader("Go Live (paper signals for Copiix)")'
if anchor in s and "## eff_params START" not in s:
    block = """
        ## eff_params START
        # Baseline from sidebar/session
        st.session_state.setdefault("quiet_bars", 0)
        eff_threshold = float(st.session_state.get("threshold", 0.69))
        eff_eps = float(st.session_state.get("epsilon_start", st.session_state.get("eps_start", 0.03)))
        # Trickle adjustment (optional)
        if st.session_state.get("trickle_on", False):
            eff_threshold, eff_eps = _trickle_effective(
                eff_threshold, eff_eps,
                quiet_bars=int(st.session_state.get("quiet_bars", 0)),
                n=int(st.session_state.get("trickle_n", 12)),
                dth=float(st.session_state.get("trickle_dth", 0.02)),
                deps=float(st.session_state.get("trickle_deps", 0.02)),
            )
        st.caption(f"Effective live params → threshold={eff_threshold:.2f}  epsilon={eff_eps:.2f}")
        ## eff_params END
"""
    s = s.replace(anchor, anchor + block, 1)

p.write_text(s)
print("✅ st_app.py: effective params defined early; unused *_live vars removed")
