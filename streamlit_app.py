"""
Streamlit launcher. Use:
    python -m streamlit run streamlit_app.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _main() -> None:
    # Import inside to keep E402 happy and avoid partial import side-effects
    from app.ui.st_app import main  # noqa: E402

    main()


if __name__ == "__main__":
    _main()
