from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()
s = s.replace(
    "eff_threshold, eff_eps = float(threshold), float(epsilon_start)",
    "eff_threshold, eff_eps = float(threshold), float(st.session_state.get('epsilon_start', 0.0))",
)
p.write_text(s)
print("✅ patched st_app.py (epsilon_start → session_state)")
