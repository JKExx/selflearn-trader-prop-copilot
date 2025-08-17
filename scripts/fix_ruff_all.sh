#!/usr/bin/env bash
set -euo pipefail

# Ensure ruff present (matches CI)
python -m pip install -q "ruff==0.5.7"

echo "▶ Ruff auto-fix (imports + pyupgrade)…"
# Fix just the rules we saw in CI (I = import sort, UP = pyupgrade)
ruff check --select I,UP --fix .

echo "▶ Format code (ensures stable import layout)…"
ruff format .

# Extra belt-and-braces for the two files CI flagged with typing cleanups
# Convert Optional[T] -> T | None and List/Dict -> list/dict where they slipped past
# (Scoped to the files that warned; adjust paths if needed later)

# oanda: UP007 (prefer X | Y)
if [ -f app/dataio/oanda.py ]; then
  # Optional[T] -> T | None
  sed -E -i '' 's/Optional\[([[:alnum:]_\.]+)\]/\1 | None/g' app/dataio/oanda.py 2>/dev/null || true
fi

# journal/news: List/Dict/Iterable cleanup if ruff missed anything
for f in app/journal.py app/news.py; do
  if [ -f "$f" ]; then
    # typing.List/Dict -> builtins; keep generics
    sed -E -i '' 's/\btyping\.List\b/list/g; s/\btyping\.Dict\b/dict/g' "$f" 2>/dev/null || true
    sed -E -i '' 's/\bList\[/list[/g; s/\bDict\[/dict[/g' "$f" 2>/dev/null || true
    # prefer collections.abc.Iterable import (ruff usually handles this)
    perl -0777 -pe 's/from typing import ([^\n]*\bIterable\b[^\n]*)\n/from collections.abc import Iterable\nfrom typing import \1\n/s' -i "$f" 2>/dev/null || true
    # tidy any duplicate imports lines that might result
    awk '!(seen[$0]++)' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
done

echo "▶ Final format pass…"
ruff format .

echo "✅ Done. Review with: git diff"
