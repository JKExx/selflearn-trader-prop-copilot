from __future__ import annotations

import pathlib
import re

p = pathlib.Path("app/ui/st_app.py")
s = p.read_text()

# 1) Remove any broken/escaped caption lines anywhere
s = re.sub(
    r"""^\s*st\.sidebar\.caption\(.+\)\s*$""",
    "",
    s,
    flags=re.M,
)

# 2) Fix emoji placeholders from earlier patches
s = s.replace("<0001f7e2>", "🟢").replace("<0001f7e0>", "🟠")

# 3) Identify top import block bounds
lines = s.splitlines()
import_re = re.compile(r"^\s*(import\s+\w|from\s+\w)")
start = None
end = None
for i, line in enumerate(lines):
    if import_re.match(line):
        start = i
        break

if start is None:
    # No imports found; just preprend imports at top
    new_imports = [
        "import os",
        "from dataclasses import dataclass",
    ]
    s = "\n".join(new_imports) + "\n\n" + s
    lines = s.splitlines()
    start = 0

# find end of contiguous import block
end = start
while end < len(lines) and (import_re.match(lines[end]) or lines[end].strip() == ""):
    end += 1

# Collect all imports across file (so we can dedupe/move to top)
all_imports = [ln for ln in lines if import_re.match(ln)]
# Ensure required ones present
if not any(re.match(r"^\s*import\s+os\s*$", ln) for ln in all_imports):
    all_imports.insert(0, "import os")
if not any(re.match(r"^\s*from\s+dataclasses\s+import\s+dataclass\s*$", ln) for ln in all_imports):
    all_imports.insert(0, "from dataclasses import dataclass")

# Dedupe while preserving order
seen = set()
dedup_imports = []
for ln in all_imports:
    key = ln.strip()
    if key and key not in seen:
        dedup_imports.append(ln.rstrip())
        seen.add(key)

# Remove all import lines from the body
body = [ln for ln in lines if not import_re.match(ln)]

# Ensure there isn't a stray dataclass import later
body = [ln for ln in body if not re.match(r"^\s*from\s+dataclasses\s+import\s+dataclass\s*$", ln)]

# Rebuild file: imports at top, then a blank, then caption, then the rest
caption = "st.sidebar.caption(f\"Prop Copilot {os.getenv('APP_VERSION', 'dev')}\")"
new_s = "\n".join(dedup_imports).rstrip() + "\n\n" + caption + "\n\n" + "\n".join(body).lstrip("\n")

# Final tidy: collapse any >2 consecutive blank lines to 2
new_s = re.sub(r"\n{3,}", "\n\n", new_s)

p.write_text(new_s)
print("✅ Fixed imports, caption position, emojis in app/ui/st_app.py")
