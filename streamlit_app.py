"""
Launcher so you can run:  python -m streamlit run streamlit_app.py
This imports the real app as a module, so relative imports in app/ui/st_app.py work.
"""
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ui.st_app import main

if __name__ == "__main__":
    main()
