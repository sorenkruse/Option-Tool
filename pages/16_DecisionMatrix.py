"""
Decision Matrix – Strategy recommendation based on market outlook.

Inputs: Symbol (auto-fetches spot, VIX), Outlook, Timeframe.
Outputs: Ranked strategy recommendations with rationale, IV context,
         and links to the relevant tool pages.
"""

import streamlit as st
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_provider import resolve_spot_price, get_vix

CSS = "<style>.main .block-container{font-size:14px;padding-top:1rem}" \
      "[data-testid=stMetricValue]{font-size:18px}" \
      "[data-testid=stMetricLabel]{font-size:12px}" \
      ".main h1{font-size:22px;margin-bottom:.3rem}" \
      ".main h3{font-size:16px;margin-top:.6rem;margin-bottom:.3rem}" \
      ".strat-card{border:1px solid #e0e0e0;border-radius:8px;padding:14px;" \
      "margin-bottom:10px;background:#fafafa;}" \
      ".strat-card h4{margin:0 0 4px 0;font-size:15px;}" \
      ".strat-card p{margin:0;font-size:13px;color:#555;line-height:1.45;}" \
      ".strat-rank{display:inline-block;background:#1565c0;color:white;" \
      "border-radius:50%;width:24px;height:24px;text-align:center;" \
      "line-height:24px;font-size:12px;font-weight:700;margin-right:8px;}" \
      ".iv-tag{display:inline-block;padding:2px 8px;border-radius:4px;" \
      "font-size:11px;font-weight:600;margin-left:8px;}" \
      ".iv-low{background:#e8f5e9;color:#2e7d32;}" \
      ".iv-norm{background:#e3f2fd;color:#1565c0;}" \
      ".iv-high{background:#fff3e0;color:#e65100;}" \
      ".iv-extreme{background:#ffebee;color:#c62828;}" \
      "</style>"

pages_dir = Path(__file__).parent


# ── IV Regime ────────────────────────────────────────────────────────────

def classify_iv(vix):
    """Classify IV regime from VIX."""
    if vix <= 15:
        return "Low", "iv-low", "Premium is cheap. Favor buying strategies."
    elif vix <= 22:
        return "Normal", "iv-norm", "Balanced environment. All strategies viable."
    elif vix <= 32:
        return "High", "iv-high", "Premium is expensive. Favor selling strategies."
    else:
        return "Extreme", "iv-extreme", "Crisis pricing. Sell premium or hedge with LEAPS."


# ── Strategy Database ────────────────────────────────────────────────────

def get_strategies(outlook, timeframe, iv_regime, vix):
    """
    Return ranked list of strategies for the given conditions.
    Each strategy: {name, type, description, rationale, iv_note, tool_page, tool_file}
    """
    strategies = []

    # Helper
    def add(name, stype, desc, rationale, iv_note="", tool_page=None, tool_file=None,
            score=50):
        strategies.append({
            "name": name, "type": stype, "description": desc,
            "rationale": rationale, "iv_note": iv_note,
            "tool_page": tool_page, "tool_file": tool_file,
            "score": score,
        })

    is_high_iv = iv_regime in ("High", "Extreme")
    is_low_iv = iv_regime == "Low"
    is_short = timeframe == "Short (1-2 weeks)"
    is_medium = timeframe == "Medium (1-2 months)"
    is_long = timeframe == "Long (3+ months)"

    # ══════════════════════════════════════════════════════════════════
    # STRONG BULLISH
    # ══════════════════════════════════════════════════════════════════
    if outlook == "Strong Bullish":
        add("Long Call", "Debit",
            "Buy an OTM call. Unlimited upside, limited downside.",
            "Direct directional bet with leverage. Best when IV is low.",
            "Low IV preferred (cheap premium)." if is_low_iv else
            "High IV makes this expensive. Consider spreads instead." if is_high_iv else "",
            "Prognose", "10_Prognose.py",
            score=85 if is_low_iv else 60)

        add("Bull Call Spread (Debit)", "Debit",
            "Buy ATM call, sell OTM call. Capped profit, reduced cost.",
            "Cheaper than naked call. Better in high IV (sells expensive premium).",
            "High IV: the short call offsets high premium." if is_high_iv else "",
            "Spreads", "11_Spreads.py",
            score=75 if is_high_iv else 65)

        add("Bull Put Spread (Credit)", "Credit",
            "Sell OTM put, buy further OTM put. Income if spot stays above SP.",
            "Profits from time decay AND the rally. Margin-efficient.",
            "High IV: excellent premium." if is_high_iv else "",
            "Iron Condor", "15_IronCondor.py",
            score=80 if is_high_iv else 70)

        if is_medium or is_long:
            add("PMCC / Stock Replacement", "Debit",
                "Buy LEAPS call, sell monthly short calls against it.",
                "Long-term bullish with income. Fraction of stock cost.",
                "", "Stock Replacement", "13_StockReplacement.py",
                score=90 if is_long else 75)

            add("Crawling Crab (Bull)", "Combo",
                "LEAPS call + short call diagonal + bull put spread.",
                "Trend-following with income. Short premium funds the LEAPS.",
                "", "Crawling Crab", "12_CrawlingCrab.py",
                score=85 if is_medium else 80)

        if is_short:
            add("Short Put (Naked)", "Credit",
                "Sell OTM put. Simple bullish income trade.",
                "Profits from rally + time decay. Requires margin.",
                "Excellent in high IV." if is_high_iv else
                "Moderate premium in normal IV.",
                "Short Finder", "3_ShortFinder.py",
                score=75 if is_high_iv else 55)

    # ══════════════════════════════════════════════════════════════════
    # BULLISH
    # ══════════════════════════════════════════════════════════════════
    elif outlook == "Bullish":
        add("Bull Put Spread (Credit)", "Credit",
            "Sell OTM put, buy further OTM put. Income if spot stays above SP.",
            "Conservative bullish. Defined risk, theta works for you.",
            "High IV: excellent premium." if is_high_iv else "",
            "Spreads", "11_Spreads.py",
            score=85 if is_high_iv else 80)

        add("Iron Condor (Bullish Skew)", "Credit",
            "Asymmetric IC: put spread close to spot, call spread far OTM.",
            "Profits from mild rally + time decay. Skewed for bullish bias.",
            "", "Iron Condor", "15_IronCondor.py",
            score=80)

        add("Bull Call Spread (Debit)", "Debit",
            "Buy ATM/OTM call, sell further OTM call.",
            "Directional bet with limited risk. Better than naked call in high IV.",
            "", "Spreads", "11_Spreads.py",
            score=70 if is_high_iv else 75)

        if is_medium or is_long:
            add("Covered Call / PMCC", "Combo",
                "Own stock (or LEAPS) + sell monthly calls.",
                "Income on existing position. Caps upside but generates theta.",
                "", "Stock Replacement", "13_StockReplacement.py",
                score=80 if is_long else 70)

            add("Jade Lizard (Bullish)", "Credit",
                "Short put + bear call spread. No upside risk.",
                "Credit trade with no risk on the upside. Risk only if spot drops hard.",
                "", "Lizards", "14_Lizards.py",
                score=75 if is_high_iv else 60)

        if is_short:
            add("Calendar Spread (Call)", "Debit",
                "Sell short-DTE call, buy longer-DTE call at same strike.",
                "Profits from time decay differential. Best in low IV.",
                "Low IV: long call is cheap." if is_low_iv else "",
                score=65 if is_low_iv else 45)

    # ══════════════════════════════════════════════════════════════════
    # RANGE / NEUTRAL
    # ══════════════════════════════════════════════════════════════════
    elif outlook == "Range":
        add("Iron Condor (Symmetric)", "Credit",
            "Bull put spread + bear call spread. Profit if spot stays in range.",
            "Classic theta trade. Profits from time decay in sideways market.",
            "High IV: more premium collected." if is_high_iv else "",
            "Iron Condor", "15_IronCondor.py",
            score=90 if is_high_iv else 80)

        add("Double Diagonal", "Debit",
            "Short-DTE shorts + longer-DTE longs. Calendar-like with wings.",
            "Theta income with crash protection. Profits from IV mean-reversion.",
            "High IV: ideal entry. IV compression helps." if is_high_iv else
            "Low IV: risky (IV expansion hurts)." if is_low_iv else "",
            "Double Diagonal", "9_DoubleDiagonal.py",
            score=85 if is_high_iv else 50 if is_low_iv else 70)

        add("Short Strangle", "Credit",
            "Sell OTM put + OTM call. Wide profit zone, undefined risk.",
            "Maximum premium. Requires active management and margin.",
            "Best in high IV." if is_high_iv else "",
            "Short Finder", "3_ShortFinder.py",
            score=80 if is_high_iv else 55)

        add("Iron Butterfly", "Credit",
            "ATM short straddle + OTM long wings. Maximum premium at spot.",
            "Highest credit of all condor variants. Tight profit zone.",
            "", score=65 if is_high_iv else 50)

        if is_medium or is_long:
            add("Flyagonal", "Combo",
                "BWB + put diagonal. Asymmetric profit with theta.",
                "Complex but efficient. Good for extended range-bound periods.",
                "", "Flyagonal", "8_Flyagonal.py",
                score=70)

            add("Jade Lizard / Rev. Jade Lizard", "Credit",
                "Naked short + credit spread on opposite side.",
                "No risk on one side. Good cycle trade for range markets.",
                "", "Lizards", "14_Lizards.py",
                score=65)

        if is_short:
            add("Credit Spreads (Both Sides)", "Credit",
                "Bull put spread + bear call spread as separate trades.",
                "Two independent theta trades. Manage each side separately.",
                "", "Spreads", "11_Spreads.py",
                score=75)

    # ══════════════════════════════════════════════════════════════════
    # BEARISH
    # ══════════════════════════════════════════════════════════════════
    elif outlook == "Bearish":
        add("Bear Call Spread (Credit)", "Credit",
            "Sell OTM call, buy further OTM call. Income if spot stays below SC.",
            "Conservative bearish. Defined risk, theta + direction.",
            "High IV: excellent premium." if is_high_iv else "",
            "Spreads", "11_Spreads.py",
            score=85 if is_high_iv else 80)

        add("Iron Condor (Bearish Skew)", "Credit",
            "Asymmetric IC: call spread close to spot, put spread far OTM.",
            "Profits from mild drop + time decay.",
            "", "Iron Condor", "15_IronCondor.py",
            score=80)

        add("Bear Put Spread (Debit)", "Debit",
            "Buy ATM/OTM put, sell further OTM put.",
            "Directional bet with limited risk.",
            "", "Spreads", "11_Spreads.py",
            score=70 if is_high_iv else 75)

        if is_medium or is_long:
            add("PMCP / Protective Put", "Debit",
                "Buy LEAPS put, sell monthly short puts.",
                "Long-term bearish with income from short puts.",
                "", "Stock Replacement", "13_StockReplacement.py",
                score=80 if is_long else 65)

            add("Crawling Crab (Bear)", "Combo",
                "LEAPS put + short put diagonal + bear call spread.",
                "Trend-following bearish. Income funds the core position.",
                "", "Crawling Crab", "12_CrawlingCrab.py",
                score=85 if is_medium else 75)

        if is_short:
            add("Short Call (Naked)", "Credit",
                "Sell OTM call. Simple bearish income.",
                "Profits from drop + time decay. Undefined risk.",
                "Excellent in high IV." if is_high_iv else "",
                "Short Finder", "3_ShortFinder.py",
                score=70 if is_high_iv else 50)

    # ══════════════════════════════════════════════════════════════════
    # STRONG BEARISH
    # ══════════════════════════════════════════════════════════════════
    elif outlook == "Strong Bearish":
        add("Long Put", "Debit",
            "Buy an OTM put. Profits from large drop. Limited risk.",
            "Direct bearish bet. Benefits from IV expansion during crash.",
            "Low IV: cheap entry." if is_low_iv else
            "High IV: expensive but IV may rise further in crash." if is_high_iv else "",
            "Prognose", "10_Prognose.py",
            score=85 if is_low_iv else 65)

        add("Bear Put Spread (Debit)", "Debit",
            "Buy ATM put, sell OTM put. Defined risk directional bet.",
            "Cheaper than naked put. Good in high IV.",
            "", "Spreads", "11_Spreads.py",
            score=75 if is_high_iv else 70)

        add("Bear Call Spread (Credit)", "Credit",
            "Sell OTM call, buy further OTM call.",
            "Profits from drop + theta. Call premium is high in volatile markets.",
            "High IV: excellent premium." if is_high_iv else "",
            "Iron Condor", "15_IronCondor.py",
            score=80 if is_high_iv else 70)

        if is_medium or is_long:
            add("Crawling Crab (Bear)", "Combo",
                "LEAPS put + short put diagonal + bear call spread.",
                "Sustained bearish thesis with income from short premium.",
                "", "Crawling Crab", "12_CrawlingCrab.py",
                score=90 if is_medium else 80)

            add("Collar (on existing long stock)", "Hedge",
                "Buy protective put, sell covered call. Locks in a range.",
                "Protects existing stock position. Zero or low cost.",
                "", score=70 if is_long else 55)

        if is_short:
            add("Put Ratio Backspread", "Debit",
                "Sell 1 ATM put, buy 2 OTM puts. Profits from large drop.",
                "Low or zero cost entry. Unlimited downside profit.",
                "Best when IV is low (cheap OTM puts)." if is_low_iv else "",
                score=70 if is_low_iv else 50)

            add("Short Call (Naked)", "Credit",
                "Sell OTM call. Income from drop expectation.",
                "Profits from drop + theta. Undefined upside risk.",
                "", "Short Finder", "3_ShortFinder.py",
                score=65 if is_high_iv else 45)

    # Sort by score descending
    strategies.sort(key=lambda x: x["score"], reverse=True)
    return strategies


# ── Display ──────────────────────────────────────────────────────────────

def display_strategies(strategies, iv_regime, iv_css):
    """Render strategy cards."""
    for i, s in enumerate(strategies):
        rank = i + 1
        iv_note = f' <span class="iv-tag {iv_css}">{s["iv_note"]}</span>' if s["iv_note"] else ""

        st.markdown(f"""<div class="strat-card">
            <span class="strat-rank">{rank}</span>
            <strong>{s['name']}</strong> ({s['type']}){iv_note}
            <p style="margin-top:6px;">{s['description']}</p>
            <p style="margin-top:4px;color:#333;"><em>{s['rationale']}</em></p>
        </div>""", unsafe_allow_html=True)

        if s.get("tool_file"):
            st.page_link(str(pages_dir / s["tool_file"]),
                          label=f"Open {s['tool_page']}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Decision Matrix")
    st.caption("Enter your market outlook to get strategy recommendations "
               "based on direction, timeframe, and IV environment.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="SPX",
            help="Underlying for spot and VIX lookup.").upper()
    with c2:
        outlook = st.selectbox("Outlook",
            ["Strong Bullish", "Bullish", "Range", "Bearish", "Strong Bearish"],
            index=2,
            help="Your directional view on the underlying.")
    with c3:
        timeframe = st.selectbox("Timeframe",
            ["Short (1-2 weeks)", "Medium (1-2 months)", "Long (3+ months)"],
            index=1,
            help="How long you expect the view to play out.")

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching market data..."):
            try:
                spot, _ = resolve_spot_price(symbol)
                vix = get_vix()
            except Exception as e:
                st.error(f"Error: {e}")
                return

        iv_regime, iv_css, iv_desc = classify_iv(vix)

        st.session_state["dm_data"] = {
            "symbol": symbol, "spot": spot, "vix": vix,
            "iv_regime": iv_regime, "iv_css": iv_css, "iv_desc": iv_desc,
            "outlook": outlook, "timeframe": timeframe,
        }

    if "dm_data" not in st.session_state:
        st.info("Select your outlook and timeframe, then click Get Recommendations.")
        return

    d = st.session_state["dm_data"]
    spot = d["spot"]; vix = d["vix"]
    iv_regime = d["iv_regime"]; iv_css = d["iv_css"]; iv_desc = d["iv_desc"]

    st.markdown("---")

    # Market context
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot", f"${spot:,.2f}")
    m2.metric("VIX", f"{vix:.1f}")
    m3.metric("IV Regime", iv_regime)
    m4.metric("Outlook", d["outlook"])

    st.markdown(f'<span class="iv-tag {iv_css}">{iv_desc}</span>',
                unsafe_allow_html=True)

    # Strategy recommendations
    st.markdown(f"### Recommended Strategies: {d['outlook']} / {d['timeframe']}")

    strategies = get_strategies(d["outlook"], d["timeframe"],
                                 iv_regime, vix)

    display_strategies(strategies, iv_regime, iv_css)

    # Context notes
    with st.expander("IV Regime Guide"):
        st.markdown("""
**Low (VIX < 15):** Premium is cheap. Favor buying strategies (long calls/puts,
debit spreads). Calendar spreads benefit from low entry cost on the long leg.
Avoid selling naked options (low reward for the risk).

**Normal (VIX 15-22):** Balanced environment. All strategies viable.
Credit and debit approaches both work. Choose based on direction and timeframe.

**High (VIX 22-32):** Premium is expensive. Favor selling strategies
(credit spreads, iron condors, short strangles). Bought options are costly.
Double Diagonals benefit from IV mean-reversion.

**Extreme (VIX > 32):** Crisis pricing. Premium selling is very lucrative
but risk is elevated. Consider wider spreads, LEAPS for long-term positions,
or protective structures. Avoid buying expensive short-DTE options.
        """)

    with st.expander("Strategy Type Guide"):
        st.markdown("""
**Credit (selling):** Collect premium upfront. Profit from time decay.
Defined or undefined risk depending on structure.
Best when IV is high (more premium to collect).

**Debit (buying):** Pay premium for directional exposure.
Profit from the underlying moving in your direction.
Best when IV is low (cheaper entry).

**Combo:** Multi-leg structures combining credit and debit components.
Crawling Crab, PMCC, Flyagonal. Usually medium to long term.

**Hedge:** Protective structures for existing positions.
Collars, protective puts. Reduce risk on current holdings.
        """)


main()
