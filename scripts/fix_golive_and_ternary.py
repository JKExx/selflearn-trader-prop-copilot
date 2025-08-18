from __future__ import annotations

import re
from pathlib import Path

p = Path("app/ui/st_app.py")
s = p.read_text()

changed = False

# ---------- 1) Fix the multiline ternary in Backtest args ----------
if "max_intraday_dd_from_high_pct" in s and 'if st.session_state.get("intraday_dd_enabled"' in s:
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

    s2 = pat.sub(repl, s)
    if s2 != s:
        s = s2
        changed = True

# ---------- 2) Remove bad 'early compute' try-block inside call args ----------
lines = s.splitlines()
out: list[str] = []
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.strip() == "# --- eff params (early compute) ---":
        # Skip until next kwarg line or closing paren
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^\s*(threshold|epsilon_start|epsilon_final)\s*=", nxt) or re.match(r"^\s*\)\s*,?\s*$", nxt):
                break
            i += 1
        # do not consume the kwarg/closing line; continue normal flow
        continue
    out.append(ln)
    i += 1
s2 = "\n".join(out)
if s2 != s:
    s = s2
    changed = True

# ---------- 3) Insert eff-params compute BEFORE live call (df_live) ----------
lines = s.splitlines()
out = []
i = 0
while i < len(lines):
    ln = lines[i]
    out.append(ln)
    if re.match(r"^\s*res\s*=\s*run_backtest\(\s*$", ln):
        # Lookahead to confirm it's the LIVE call (next non-empty line has df_live,)
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and re.search(r"\bdf_live\b\s*,\s*$", lines[j].strip()):
            # Insert compute block if not already present immediately above
            prev = out[-2] if len(out) >= 2 else ""
            if "# --- eff params (compute before live call) ---" not in prev:
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
                out[-1:-1] = block  # insert before 'res = run_backtest('
    i += 1
s2 = "\n".join(out)
if s2 != s:
    s = s2
    changed = True

# ---------- 4) Force kwargs to use eff_* inside the LIVE call only ----------
# We do a localized replace in the slice between the live-call open and its closing ')'
lines = s.splitlines()
out = []
i = 0
while i < len(lines):
    ln = lines[i]
    out.append(ln)
    if re.match(r"^\s*res\s*=\s*run_backtest\(\s*$", ln):
        # Scan forward to closing paren, track if this is live (df_live,)
        start_idx = i
        j = i + 1
        is_live = False
        while j < len(lines):
            if re.search(r"\bdf_live\b\s*,\s*$", lines[j].strip()):
                is_live = True
            if re.match(r"^\s*\)\s*,?\s*$", lines[j]):  # closing of call
                end_idx = j
                break
            j += 1
        else:
            i += 1
            continue  # malformed; don't touch

        if is_live:
            slice_lines = lines[start_idx : end_idx + 1]
            for k in range(len(slice_lines)):
                sl = slice_lines[k]
                sl = re.sub(r"(threshold\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_threshold\3", sl)
                sl = re.sub(r"(epsilon_start\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_eps\3", sl)
                sl = re.sub(r"(epsilon_final\s*=\s*)(.+?)(\s*,\s*)$", r"\1eff_eps\3", sl)
                slice_lines[k] = sl
            # replace in output
            out = out[: -(len(out) - start_idx)] + slice_lines
            # skip original region in input
            i = end_idx
    i += 1
s2 = "\n".join(out)
if s2 != s:
    s = s2
    changed = True

# ---------- 5) Remove unused *_live temporary assigns ----------
s2 = re.sub(r"^\s*(threshold_live|eps_start_live|eps_final_live)\s*=.*\n", "", s, flags=re.M)
if s2 != s:
    s = s2
    changed = True

if changed:
    p.write_text(s if s.endswith("\n") else s + "\n")
    print("✅ Patched: st_app.py (ternary fixed, eff_* computed pre-call, kwargs use eff_*, junk removed)")
else:
    print("ℹ No changes (already patched)")
