#!/usr/bin/env bash
set -euo pipefail
if [ -f .streamlit.pid ]; then
  PID=$(cat .streamlit.pid)
  echo "Killing ${PID}…"
  kill "${PID}" 2>/dev/null || true
  sleep 0.3
  kill -9 "${PID}" 2>/dev/null || true
  rm -f .streamlit.pid
  echo "✅ Stopped."
else
  echo "No .streamlit.pid; checking processes…"
  pgrep -fl "streamlit run|python -m streamlit" || echo "Nothing running."
fi
