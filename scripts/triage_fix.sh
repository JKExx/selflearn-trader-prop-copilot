#!/usr/bin/env bash
set -euo pipefail

OWNER="JKExx"
REPO="selflearn-trader-prop-copilot"

gh auth status >/dev/null 2>&1 || { echo "Run: gh auth login"; exit 1; }

get_or_create_ms() {
  local title="$1"
  local num
  num=$(gh api repos/$OWNER/$REPO/milestones -f state=all --jq ".[] | select(.title==\"$title\") | .number" | head -n1 || true)
  if [ -z "$num" ]; then
    num=$(gh api -X POST repos/$OWNER/$REPO/milestones \
      -f title="$title" -f state='open' --jq .number)
  fi
  echo "$num"
}

M1=$(get_or_create_ms "v0.3.1")
M2=$(get_or_create_ms "v0.3.2")

case "$M1" in (*[!0-9]*|"") echo "Bad milestone number for v0.3.1: '$M1'"; exit 1;; esac
case "$M2" in (*[!0-9]*|"") echo "Bad milestone number for v0.3.2: '$M2'"; exit 1;; esac

gh label create "type:tests"    -c "#5319e7" --description "Tests"           --force >/dev/null 2>&1 || true
gh label create "priority:high" -c "#b60205" --description "High priority"   --force >/dev/null 2>&1 || true
gh label create "type:feature"  -c "#0e8a16" --description "Feature request" --force >/dev/null 2>&1 || true

gh issue edit 2  --milestone "$M1" --add-label "type:feature" --add-label "priority:high" || true
gh issue edit 3  --milestone "$M1" --add-label "type:feature"                             || true
gh issue edit 10 --milestone "$M1" --add-label "type:tests"                               || true

gh issue edit 8  --milestone "$M2" --add-label "type:feature" --add-label "priority:high" || true
gh issue edit 9  --milestone "$M2" --add-label "type:feature"                             || true

gh issue close 4  -r "not planned" -c "Duplicate of #8 — tracking there."   || true
gh issue close 5  -r "not planned" -c "Duplicate of #9 — tracking there."   || true
gh issue close 7  -r "completed"   -c "Heartbeat shipped; track why-no-trade in #2." || true

echo "Milestones: v0.3.1=#${M1}  v0.3.2=#${M2}"
echo "OK"
