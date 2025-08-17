import pathlib
import re

p = pathlib.Path("app/backtest.py")
s = p.read_text()
# Match any variable before the comma; allow arbitrary spaces/newlines
s2 = re.sub(
    r"isinstance\(\s*([^,]+?)\s*,\s*\(\s*(tuple)\s*,\s*(list)\s*\)\s*\)",
    r"isinstance(\1, tuple | list)",
    s,
    flags=re.M,
)
s2 = re.sub(
    r"isinstance\(\s*([^,]+?)\s*,\s*\(\s*(list)\s*,\s*(tuple)\s*\)\s*\)",
    r"isinstance(\1, tuple | list)",
    s2,
    flags=re.M,
)
if s2 != s:
    p.write_text(s2)
    print("Patched UP038 in app/backtest.py")
else:
    print("No UP038 patterns found (already fixed?).")
