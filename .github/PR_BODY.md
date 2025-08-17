## What
Hardening pass:
- Absolute imports + launcher (`streamlit_app.py`)
- Start scripts (macOS/Linux + Windows), Dockerfile, docker-compose
- Ruff config migrated to `[tool.ruff.lint]`, per-file ignores (UI/E402)
- UP038 fixes (`tuple | list`)
- State store for live loop (`app/utils/state.py`)
- .env.example, .gitignore updates

## Why
- One-command start for new users
- Stable imports when running Streamlit
- Green, informative CI (ruff config printed; no cache)
- Avoid re-signalling on restart

## Checks
- [x] `ruff check` passes locally
- [x] App starts: `python -m streamlit run streamlit_app.py`
- [x] Docker start: `docker compose up --build`
