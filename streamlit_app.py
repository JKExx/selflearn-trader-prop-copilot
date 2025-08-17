"""
Streamlit launcher. Run with:
    python -m streamlit run streamlit_app.py
This imports the Streamlit UI module so its top-level code runs.
"""

import importlib
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _main() -> None:
    # If st_app exposes main(), call it; otherwise just import the module
    try:
        from app.ui.st_app import main as st_main  # noqa: E402

        st_main()
    except Exception:
        importlib.import_module("app.ui.st_app")  # executes top-level Streamlit code


if __name__ == "__main__":
    _main()
