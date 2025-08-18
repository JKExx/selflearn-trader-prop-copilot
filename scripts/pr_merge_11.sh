#!/usr/bin/env bash
set -euo pipefail
git commit --allow-empty -m "ci: retrigger" || true
git push
gh pr checks 11
gh pr merge 11 --squash --delete-branch
