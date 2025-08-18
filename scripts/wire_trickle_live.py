import re
from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()

# 1) Ensure we define live aliases right after the "Effective live params" caption
anchor = 'st.caption(f"Effective live params → threshold={eff_threshold:.2f}  epsilon={eff_eps:.2f}")'
if anchor in s and "threshold_live = eff_threshold" not in s:
    s = s.replace(
        anchor,
        anchor + "\n        threshold_live = eff_threshold\n"
        "        eps_start_live = eff_eps\n"
        "        eps_final_live = eff_eps\n",
    )

# 2) Replace uses at Go Live call sites
s = re.sub(r"threshold\s*=\s*st\.session_state\.get\(\s*\"threshold\"\s*,\s*0\.69\s*\)", "threshold=threshold_live", s)
s = re.sub(
    r"epsilon_start\s*=\s*st\.session_state\.get\(\s*\"eps_start\"\s*,\s*0\.03\s*\)", "epsilon_start=eps_start_live", s
)
# Optional: if any epsilon_final still uses the old value, mirror it
s = re.sub(r"epsilon_final\s*=\s*st\.session_state\.get\(\s*\"eps_final\".*?\)", "epsilon_final=eps_final_live", s)

Path("app/ui/st_app.py").write_text(s)
print("✅ wired eff_threshold/eff_eps into live call sites")
