"""
Options Tool - Multi-Page Application
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Options Tool", layout="wide")


# --- Password gate (uses Streamlit Cloud secrets) ---
def check_password():
    """Returns True if the user has entered the correct password."""
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return True

    try:
        correct_pw = st.secrets["auth"]["password"]
    except (KeyError, FileNotFoundError):
        # No secret configured -> skip auth (local development)
        return True

    pw = st.text_input("Password", type="password")
    if pw:
        if pw == correct_pw:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()


# --- Navigation ---
pages_dir = Path(__file__).parent / "pages"

pg = st.navigation([
    st.Page(str(pages_dir / "1_Pricing.py"), title="Pricing"),
    st.Page(str(pages_dir / "2_Explorer.py"), title="Explorer"),
    st.Page(str(pages_dir / "3_ShortFinder.py"), title="Short Finder"),
    st.Page(str(pages_dir / "6_LongFinder.py"), title="Long Finder"),
    st.Page(str(pages_dir / "4_Optimizer.py"), title="Optimizer"),
    st.Page(str(pages_dir / "5_HoldOrRoll.py"), title="Hold or Roll"),
    st.Page(str(pages_dir / "7_TradeComparator.py"), title="Trade Comparator"),
    st.Page(str(pages_dir / "8_Flyagonal.py"), title="Flyagonal"),
    st.Page(str(pages_dir / "9_DoubleDiagonal.py"), title="Double Diagonal"),
    st.Page(str(pages_dir / "10_Prognose.py"), title="Prognose"),
    st.Page(str(pages_dir / "11_Verticals.py"), title="Verticals"),
    st.Page(str(pages_dir / "12_CrawlingCrab.py"), title="Crawling Crab"),
    st.Page(str(pages_dir / "13_StockReplacement.py"), title="Stock Replacement"),
])
pg.run()
