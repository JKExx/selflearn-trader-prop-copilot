import pathlib
import re

p = pathlib.Path("app/backtest.py")
s = p.read_text()
# Replace: isinstance(feats_out, (tuple, list))  ->  isinstance(feats_out, tuple | list)
s2 = re.sub(r"isinstance\\(feats_out,\\s*\\(tuple,\\s*list\\)\\)", "isinstance(feats_out, tuple | list)", s)
if s != s2:
    p.write_text(s2)
    print("Patched app/backtest.py (UP038).")
else:
    print("No change needed; pattern not found.")
