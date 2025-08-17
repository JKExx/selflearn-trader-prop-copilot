# Changelog

## v0.3.0 – Hardening & Quickstart
- Absolute imports + module launcher (`streamlit_app.py`)
- Start scripts (macOS/Linux + Windows)
- Dockerfile + docker-compose (one-command start)
- Ruff config migrated to `[tool.ruff.lint]`, per-file ignores for UI/E402
- UP038 fixes (`tuple | list`)
- Persistent live state (`app/utils/state.py`)
- `.env.example`, `.gitignore` updates
- CI: lint-only ruff + compile smoke
