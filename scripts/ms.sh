#!/usr/bin/env zsh
set -euo pipefail
set +H   # disable history expansion so patterns like <-> work

OWNER="JKExx"
REPO="selflearn-trader-prop-copilot"

get_ms() {
  local t="$1" n=""
  n=$(gh api repos/$OWNER/$REPO/milestones -f state=open   --paginate --jq ".[] | select(.title==\"$t\") | .number" | head -n1 || true)
  if [[ -z "$n" ]]; then
    n=$(gh api repos/$OWNER/$REPO/milestones -f state=closed --paginate --jq ".[] | select(.title==\"$t\") | .number" | head -n1 || true)
  fi
  if [[ -z "$n" ]]; then
    n=$(gh api -X POST repos/$OWNER/$REPO/milestones -f title="$t" --jq .number)
  fi
  echo "$n"
}

M1="$(get_ms 'v0.3.1')"
M2="$(get_ms 'v0.3.2')"

# In zsh, <-> matches “one or more digits”
if [[ "$M1" != <-> || "$M2" != <-> ]]; then
  print -u2 "bad milestone ids: M1=$M1  M2=$M2"
  exit 1
fi

gh label create type:tests    -c "#5319e7" --force >/dev/null 2>&1 || true
gh label create priority:high -c "#b60205" --force >/dev/null 2>&1 || true
gh label create type:feature  -c "#0e8a16" --force >/dev/null 2>&1 || true

gh issue edit 2  --milestone "$M1" --add-label type:feature --add-label priority:high || true
gh issue edit 3  --milestone "$M1" --add-label type:feature                             || true
gh issue edit 10 --milestone "$M1" --add-label type:tests                                || true

gh issue edit 8  --milestone "$M2" --add-label type:feature --add-label priority:high || true
gh issue edit 9  --milestone "$M2" --add-label type:feature                             || true

gh issue close 4  -r "not planned" -c "Duplicate of #8 — tracking there." || true
gh issue close 5  -r "not planned" -c "Duplicate of #9 — tracking there." || true
gh issue close 7  -r "completed"   -c "Heartbeat shipped; track why-no-trade in #2." || true

print "OK  M1=$M1  M2=$M2"
