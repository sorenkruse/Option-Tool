"""
Options Tool - Multi-Page Application
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

pages_dir = Path(__file__).parent / "pages"

st.set_page_config(page_title="Options Tool", layout="wide")

pg = st.navigation([
    st.Page(str(pages_dir / "1_Pricing.py"), title="Pricing"),
    st.Page(str(pages_dir / "2_Explorer.py"), title="Explorer"),
    st.Page(str(pages_dir / "3_ShortFinder.py"), title="Short Finder"),
    st.Page(str(pages_dir / "4_Optimizer.py"), title="Optimizer"),
])
pg.run()
