# v0.3.0 – Hardening & Quickstart
- Absolute imports + root launcher (`streamlit_app.py`)
- Quickstart scripts: `start.sh` / `start.ps1`
- Dockerfile + docker-compose (one-command run)
- Ruff config migrated to `[tool.ruff.lint]`, per-file relax for UI/E402
- UP038 / UP007 / typing cleanups
- Persistent live state (`app/utils/state.py`)
- `.env.example`, `.gitignore` updates
- CI: lint-only + compile-smoke
