from __future__ import annotations

import re
from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()
orig = s
changed = False

# 1) Fix multiline ternary in Backtest args
pat = re.compile(
    r'^(\s*)max_intraday_dd_from_high_pct\s*=\s*st\.session_state\.get\("max_intraday_dd_from_high_pct"\)\s*\n'
    r'^\s*if\s+st\.session_state\.get\("intraday_dd_enabled",\s*True\)\s*\n'
    r"^\s*else\s+None,\s*$",
    re.M,
)


def repl(m: re.Match) -> str:
    ind = m.group(1)
    return (
        f"{ind}max_intraday_dd_from_high_pct=(\n"
        f'{ind}    st.session_state.get("max_intraday_dd_from_high_pct")\n'
        f'{ind}    if st.session_state.get("intraday_dd_enabled", True)\n'
        f"{ind}    else None\n"
        f"{ind}),"
    )


s = pat.sub(repl, s)

# 2) Remove any prior bad 'early compute' block inside call args
lines = s.splitlines()
out = []
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.strip() == "# --- eff params (early compute) ---":
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^\s*(threshold|epsilon_start|epsilon_final)\s*=", nxt) or re.match(r"^\s*\)\s*,?\s*$", nxt):
                break
            i += 1
        continue
    out.append(ln)
    i += 1
s = "\n".join(out)

# 3) Remove unused *_live temps
s = re.sub(r"^\s*(threshold_live|eps_start_live|eps_final_live)\s*=.*\n", "", s, flags=re.M)

# 4) For every LIVE call (run_backtest with df_live,), insert eff_* compute block immediately above,
#    and force kwargs to use eff_threshold / eff_eps inside that call only.
lines = s.splitlines()
out = []
i = 0
while i < len(lines):
    ln = lines[i]
    out.append(ln)
    if re.match(r"^\s*res\s*=\s*run_backtest\(\s*$", ln):
        # confirm live call by peeking the next non-empty line
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        is_live = j < len(lines) and re.search(r"\bdf_live\b\s*,\s*$", lines[j].strip())
        if is_live:
            indent = re.match(r"^(\s*)", ln).group(1)
            block = [
                f"{indent}# --- eff params (compute before live call) ---",
                f'{indent}st.session_state.setdefault("quiet_bars", 0)',
                f'{indent}eff_threshold = float(st.session_state.get("threshold", 0.69))',
                f"{indent}eff_eps = float(",
                f'{indent}    st.session_state.get("epsilon_start", st.session_state.get("eps_start", 0.03))',
                f"{indent})",
                f'{indent}if st.session_state.get("trickle_on", False):',
                f"{indent}    eff_threshold, eff_eps = _trickle_effective(",
                f"{indent}        eff_threshold, eff_eps,",
                f'{indent}        quiet_bars=int(st.session_state.get("quiet_bars", 0)),',
                f'{indent}        n=int(st.session_state.get("trickle_n", 12)),',
                f'{indent}        dth=float(st.session_state.get("trickle_dth", 0.02)),',
                f'{indent}        deps=float(st.session_state.get("trickle_deps", 0.02)),',
                f"{indent}    )",
            ]
            out[-1:-1] = block  # insert above the call opener
            # now rewrite kwargs inside this call until closing ')'
            k = i + 1
            while k < len(lines):
                if re.match(r"^\s*\)\s*,?\s*$", lines[k]):
                    break
                repl_line = lines[k]
                repl_line = re.sub(r"(threshold\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_threshold\3", repl_line)
                repl_line = re.sub(r"(epsilon_start\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_eps\3", repl_line)
                repl_line = re.sub(r"(epsilon_final\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_eps\3", repl_line)
                lines[k] = repl_line
                k += 1
    i += 1
s = "\n".join(lines)

if s != orig:
    p.write_text(s if s.endswith("\n") else s + "\n")
    print("OK: st_app.py patched")
else:
    print("No changes needed")
