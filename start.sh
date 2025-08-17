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

# --- config ---
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BIND="${BIND_ADDR:-localhost}"      # override with: BIND_ADDR=0.0.0.0 ./start.sh
PORT="${PORT:-8501}"
URL="http://localhost:${PORT}"      # always show localhost to the user
LAUNCH_BROWSER="${LAUNCH_BROWSER:-1}"  # set 0 to skip auto-open

# --- port check ---
if command -v lsof >/dev/null 2>&1; then
  if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "✖ Port ${PORT} already in use."
    echo "  Tip: lsof -i :${PORT}  # see process"
    echo "       kill <PID>         # stop it"
    exit 1
  fi
fi

# --- tidy on exit ---
STREAMLIT_PID=""
cleanup() {
  echo
  echo "⏹ Streamlit stopped."
  if [ -n "${STREAMLIT_PID}" ] && ps -p "${STREAMLIT_PID}" >/dev/null 2>&1; then
    kill "${STREAMLIT_PID}" >/dev/null 2>&1 || true
    sleep 0.5
    kill -9 "${STREAMLIT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT

echo "▶ Starting app (bind ${BIND}:${PORT})"
# Launch server in background so we can wait for readiness
python -m streamlit run streamlit_app.py \
  --server.address "${BIND}" \
  --server.port "${PORT}" \
  --server.headless true \
  &
STREAMLIT_PID=$!

# --- wait for health ---
echo -n "⏳ Waiting for Streamlit to be ready"
HEALTH_URL="http://localhost:${PORT}/healthz"
ATTEMPTS=120   # ~60s with 0.5s sleep
i=0
while :; do
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      break
    fi
  else
    # Fallback: try a simple TCP connect using bash builtins
    if exec 3<>"/dev/tcp/localhost/${PORT}" 2>/dev/null; then
      exec 3>&-
      break
    fi
  fi
  i=$((i+1))
  if [ "$i" -ge "$ATTEMPTS" ]; then
    echo
    echo "⚠ Gave up waiting for Streamlit health; opening anyway."
    break
  fi
  printf '.'
  sleep 0.5
done
echo

# --- open browser (after health OK) ---
if [ "${LAUNCH_BROWSER}" = "1" ] && command -v open >/dev/null 2>&1; then
  # add a cache-buster so Safari doesn't show a stale blank
  ( sleep 0.2; open "${URL}?t=$(date +%s)" ) &
fi
echo "▶ Open ${URL}"

# Keep the script attached to the server process
wait "${STREAMLIT_PID}"
