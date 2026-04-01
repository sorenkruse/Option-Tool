"""Landing page – Overview of all tools with clickable links."""

import streamlit as st
from pathlib import Path

pages_dir = Path(__file__).parent

st.markdown("""
<style>
.tool-desc { font-size: 13px; color: #555; line-height: 1.5; margin-bottom: 8px; }
.tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-right: 4px;
}
.tag-scan { background: #e3f2fd; color: #1565c0; }
.tag-build { background: #e8f5e9; color: #2e7d32; }
.tag-analyze { background: #fff3e0; color: #e65100; }
.tag-manage { background: #f3e5f5; color: #7b1fa2; }
.section-head {
    font-size: 14px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 24px 0 8px 0; padding-bottom: 4px;
    border-bottom: 2px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

st.title("Options Tool")
st.caption("14 specialized tools for options analysis, strategy building, and trade management.")


def tool_card(page_file, title, desc, tags):
    """Render a tool card with clickable page link."""
    st.page_link(str(pages_dir / page_file), label=f"**{title}**")
    st.markdown(f'<div class="tool-desc">{desc}</div>', unsafe_allow_html=True)
    tag_html = ""
    for label, cls in tags:
        tag_html += f'<span class="tag {cls}">{label}</span>'
    if tag_html:
        st.markdown(tag_html, unsafe_allow_html=True)


st.markdown('<div class="section-head">Decision Support</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("16_DecisionMatrix.py", "Decision Matrix",
              "Enter your market outlook (direction + timeframe) and get "
              "ranked strategy recommendations. Considers IV regime, "
              "trade type (credit/debit), and links to the right tools.",
              [("Analysis", "tag-analyze")])
with c2:
    st.empty()
with c3:
    st.empty()

# ── Scanners & Forecasting ──
st.markdown('<div class="section-head">Scanners & Forecasting</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("10_Prognose.py", "Prognose",
              "Forecast-driven single-option scanner. "
              "Enter expected move % and time horizon. "
              "3-scenario P&L (Full / Half / Flat) with weighted EV.",
              [("Scanner", "tag-scan"), ("Forecast", "tag-analyze")])
with c2:
    tool_card("15_IronCondor.py", "Iron Condor",
              "Forecast-driven asymmetric Iron Condor. "
              "Profit side OTM, risk side placed beyond target. "
              "Asymmetric widths, 4-scenario P&L, DTE optimization.",
              [("Scanner", "tag-scan"), ("Forecast", "tag-analyze")])
with c3:
    tool_card("11_Spreads.py", "Spreads",
              "Vertical and diagonal spread scanner. "
              "4 delta inputs, all spread types. "
              "Diagonal toggle for credit long leg +20% DTE.",
              [("Scanner", "tag-scan")])

# ── Strategy Builders ──
st.markdown('<div class="section-head">Strategy Builders</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("12_CrawlingCrab.py", "Crawling Crab",
              "Trend-following: LEAPS + Short Diagonal + Credit Spread. "
              "Rolling short premium finances the core long. "
              "Auto-DTE optimization for fastest funding.",
              [("Builder", "tag-build"), ("Cycles", "tag-manage")])
with c2:
    tool_card("14_Lizards.py", "Lizards",
              "Renewal trades for Crawling Crab cycles. "
              "Reverse Jade Lizard (bull) or Jade Lizard (bear). "
              "Scans all short DTEs for optimal daily income.",
              [("Builder", "tag-build"), ("Cycles", "tag-manage")])
with c3:
    tool_card("13_StockReplacement.py", "Stock Replacement",
              "Replace stock with PMCC (bull) or PMCP (bear). "
              "LEAPS + monthly short options at a fraction of cost. "
              "Scenario analysis and P&L chart vs 100 shares.",
              [("Builder", "tag-build")])

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("9_DoubleDiagonal.py", "Double Diagonal",
              "Asymmetric put+call diagonal with IV-floor resilience. "
              "Calendar or diagonal style, auto DTE optimization. "
              "Asym LC DTE toggle for bullish bias.",
              [("Builder", "tag-build")])
with c2:
    tool_card("8_Flyagonal.py", "Flyagonal",
              "Call Broken Wing Butterfly + Put Diagonal. "
              "Delta-based strike placement with DTE optimization. "
              "Theta efficiency and zone width scoring.",
              [("Builder", "tag-build")])
with c3:
    st.empty()

# ── Position Management ──
st.markdown('<div class="section-head">Position Management</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("5_HoldOrRoll.py", "Hold or Roll",
              "Evaluate existing positions: hold to expiry or roll? "
              "Roll candidates scored by credit/debit and days to 50%. "
              "IBKR CSV export for direct TWS import.",
              [("Management", "tag-manage")])
with c2:
    tool_card("7_TradeComparator.py", "Trade Comparator",
              "Compare up to 5 multi-leg trades side by side. "
              "P&L curves, Greeks aggregation, IBKR CSV per trade.",
              [("Analysis", "tag-analyze")])
with c3:
    st.empty()

# ── Single-Leg Tools ──
st.markdown('<div class="section-head">Single-Leg Tools</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("3_ShortFinder.py", "Short Finder",
              "OTM short option scanner. Naked puts, calls, strangles. "
              "PoP-based filtering with margin estimates.",
              [("Scanner", "tag-scan")])
with c2:
    tool_card("6_LongFinder.py", "Long Finder",
              "Directional long scanner with short alternative comparison. "
              "Best long calls/puts for directional bets.",
              [("Scanner", "tag-scan")])
with c3:
    tool_card("2_Explorer.py", "Explorer",
              "Single option deep-dive with market data. "
              "Greeks, IV smile curve, OptionStrat link.",
              [("Analysis", "tag-analyze")])

# ── Fundamentals ──
st.markdown('<div class="section-head">Fundamentals</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    tool_card("1_Pricing.py", "Pricing",
              "Black-Scholes calculator. Greeks over time, "
              "IV smile visualization, what-if scenarios.",
              [("Analysis", "tag-analyze")])
with c2:
    st.empty()
with c3:
    st.empty()

# ── Workflow & Concepts ──
with st.expander("Typical Workflow"):
    st.markdown("""
**1. Market View** -- Prognose or Optimizer to find opportunities.

**2. Strategy Selection:**
- Neutral/Income: Verticals (credit spreads), Double Diagonal
- Trending: Crawling Crab (initial) + Lizards (renewal cycles)
- Stock Replacement: PMCC/PMCP for long-term thesis

**3. Refinement** -- Explorer for deep-dives, Trade Comparator for alternatives.

**4. Execution** -- OptionStrat links for visualization, IBKR CSV for TWS import.

**5. Management** -- Hold or Roll evaluates positions approaching expiry.
    """)

with st.expander("Key Concepts"):
    st.markdown("""
**Diagonal Spread Toggle** -- Spreads, Iron Condor, Crawling Crab, Lizards.
Long leg gets +20% more DTE. Better vega protection in IV spikes.
Only applies to credit spreads (not debit spreads).

**3-Scenario P&L** -- Prognose, Iron Condor.
Full Move (your forecast hits), Half Move (50% of expected), Flat (no move).
Weighted P&L: 60% Full / 35% Half / 5% Flat.

**Asym LC DTE** -- Double Diagonal.
Long Call gets 1.5x Long Put DTE. Compensates IV asymmetry on rallies.

**Cycle Economics** -- Crawling Crab, Lizards.
Measures how many short-DTE cycles (closed at 50%) fund the core long.

**IV Regime** -- Decision Matrix.
VIX-based classification: Low (<15), Normal (15-22), High (22-32), Extreme (>32).
Drives strategy selection: buy in low IV, sell in high IV.
    """)
