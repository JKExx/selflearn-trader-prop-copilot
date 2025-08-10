
import streamlit as st
import traceback
try:
    from app.ui.st_app import main
    main()
except Exception:
    st.set_page_config(page_title="SelfLearn Trader", layout="wide")
    st.error("App failed to load.")
    st.code(traceback.format_exc())
