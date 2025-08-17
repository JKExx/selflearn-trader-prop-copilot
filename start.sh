#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "▶ Creating .venv …"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip >/dev/null
pip install -r requirements.txt >/dev/null

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
echo "▶ Starting app on http://localhost:8501"
exec python -m streamlit run app/ui/st_app.py --server.address 0.0.0.0 --server.port 8501
