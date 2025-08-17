#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# --- venv bootstrap ---
if [ ! -d ".venv" ]; then
  echo "▶ Creating .venv …"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip >/dev/null
pip install -r requirements.txt >/dev/null

# --- version for UI chip ---
export APP_VERSION="${APP_VERSION:-$(git describe --tags --always 2>/dev/null || echo v0.3.0)}"

# --- config ---
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BIND="${BIND_ADDR:-localhost}"
PORT="${PORT:-8501}"
URL="http://localhost:${PORT}"
HEALTH_URL="${URL}/healthz"
DETACH="${DETACH:-0}"
LAUNCH_BROWSER="${LAUNCH_BROWSER:-1}"

# --- port check ---
if command -v lsof >/dev/null 2>&1 && lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "✖ Port ${PORT} already in use."
  echo "  Tip: lsof -i :${PORT} ; kill <PID>"
  exit 1
fi

echo "▶ Starting app (bind ${BIND}:${PORT})"

_run() {
  python -m streamlit run streamlit_app.py \
    --server.address "${BIND}" \
    --server.port "${PORT}" \
    --server.headless true
}

if [ "${DETACH}" = "1" ]; then
  LOG="${LOG:-/tmp/prop-copilot.streamlit.log}"
  nohup bash -c "_run" >/dev/null 2>>"${LOG}" &
  PID=$!
  echo "${PID}" > .streamlit.pid
  echo "▶ PID ${PID} (detached). Log: ${LOG}"
  echo -n "⏳ Waiting for Streamlit to be ready"
  for _ in $(seq 1 120); do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then break; fi
    printf '.'; sleep 0.5
  done
  echo; echo "▶ Open ${URL}"
  if [ "${LAUNCH_BROWSER}" = "1" ] && command -v open >/dev/null 2>&1; then
    ( sleep 0.2; open "${URL}?t=$(date +%s)" ) &
  fi
  echo "ℹ To stop: ./stop.sh"
  exit 0
fi

cleanup(){ echo; echo "⏹ Streamlit stopped."; }
trap cleanup EXIT INT

# foreground
_run &
PID=$!

echo -n "⏳ Waiting for Streamlit to be ready"
for _ in $(seq 1 120); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then break; fi
  printf '.'; sleep 0.5
done
echo; echo "▶ Open ${URL}"
if [ "${LAUNCH_BROWSER}" = "1" ] && command -v open >/dev/null 2>&1; then
  ( sleep 0.2; open "${URL}?t=$(date +%s)" ) &
fi

wait "${PID}"
