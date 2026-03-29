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


# --- Page definitions with descriptions ---
pages_dir = Path(__file__).parent / "pages"

PAGE_INFO = {
    "Optimizer": "PoP + Target-Zone scoring for single options. "
                 "Finds the best risk/reward across strikes and DTEs.",
    "Prognose": "Forecast-driven scanner with conviction model. "
                "Enter your directional view (move %, DTE, conviction) "
                "and find the best single-leg option. 3-scenario P&L.",
    "Verticals": "Vertical spread scanner (Bull Put, Bear Call, Bull Call, Bear Put). "
                 "Same forecast model as Prognose, applied to 2-leg spreads.",
    "Crawling Crab": "Trend-following: LEAPS + Short Diagonal + Credit Spread. "
                     "Rolling short premium finances the core long position. "
                     "Auto-DTE optimization.",
    "Lizards": "Renewal trades for Crawling Crab cycles. "
               "Reverse Jade Lizard (bull) or Jade Lizard (bear). "
               "Scans all short DTEs for optimal daily income.",
    "Stock Replacement": "Replace stock with PMCC (bull) or PMCP (bear). "
                         "LEAPS + monthly short options. "
                         "Scenario analysis vs 100 shares.",
    "Double Diagonal": "Asymmetric put+call diagonal with IV-floor resilience. "
                       "Calendar or diagonal style, DTE optimization.",
    "Flyagonal": "Call Broken Wing Butterfly + Put Diagonal. "
                 "Delta-based strike placement with DTE optimization.",
    "Hold or Roll": "Evaluate existing positions: hold to expiry or roll? "
                    "Roll candidates with credit/debit scoring.",
    "Short Finder": "OTM short option scanner. Strangles, naked puts/calls. "
                    "PoP-based filtering with margin estimates.",
    "Long Finder": "Directional long scanner with short alternative comparison. "
                   "Find the best long options for a directional bet.",
    "Explorer": "Single option deep-dive with market data. "
                "Greeks, IV smile, OptionStrat link.",
    "Trade Comparator": "Compare up to 5 multi-leg trades side by side. "
                        "P&L curves, Greeks, IBKR CSV per trade.",
    "Pricing": "Black-Scholes calculator. Greeks over time chart, "
               "IV smile visualization, what-if scenarios.",
}

PAGES = [
    ("Optimizer",         "4_Optimizer.py"),
    ("Prognose",          "10_Prognose.py"),
    ("Verticals",         "11_Verticals.py"),
    ("Crawling Crab",     "12_CrawlingCrab.py"),
    ("Lizards",           "14_Lizards.py"),
    ("Stock Replacement", "13_StockReplacement.py"),
    ("Double Diagonal",   "9_DoubleDiagonal.py"),
    ("Flyagonal",         "8_Flyagonal.py"),
    ("Hold or Roll",      "5_HoldOrRoll.py"),
    ("Short Finder",      "3_ShortFinder.py"),
    ("Long Finder",       "6_LongFinder.py"),
    ("Explorer",          "2_Explorer.py"),
    ("Trade Comparator",  "7_TradeComparator.py"),
    ("Pricing",           "1_Pricing.py"),
]

pg = st.navigation([
    st.Page(str(pages_dir / filename), title=title)
    for title, filename in PAGES
])

# --- Sidebar info box ---
with st.sidebar:
    current = pg.title if hasattr(pg, "title") else ""
    info = PAGE_INFO.get(current, "")
    if info:
        st.caption(info)

pg.run()
