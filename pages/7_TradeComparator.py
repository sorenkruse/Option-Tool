"""
Trade Comparator - Compare different trade ideas for the same market outlook.

Enter up to 5 trade ideas (each with 1-4 legs) and compare their
P&L, capital efficiency, and risk across 3 scenarios.

Leg notation: "LP 6500" = Long Put 6500, "SC 7000 x2" = 2x Short Call 7000
Supported: LP, SP, LC, SC + strike + optional quantity (x2, x3)
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    resolve_spot_price, resolve_options_chain, get_available_expirations,
    get_options_chain, get_dividend_yield, get_risk_free_rate, get_vix,
    build_smile_curve, interpolate_smile_iv,
)

CUSTOM_CSS = """
<style>
    .main .block-container { font-size: 14px; padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 18px; }
    [data-testid="stMetricLabel"] { font-size: 12px; }
    .main h1 { font-size: 22px; margin-bottom: 0.3rem; }
    .main h3 { font-size: 16px; margin-top: 0.6rem; margin-bottom: 0.3rem; }
</style>
"""

# Leg notation examples
NOTATION_HELP = (
    "Enter legs separated by comma. Examples:\n"
    "- LP 6500 = Long Put 6500\n"
    "- SC 7000 x2 = 2x Short Call 7000\n"
    "- SP 6800, SC 7100 = Short Strangle\n"
    "- LP 6400, SP 6600 = Put Credit Spread\n"
    "Prefixes: LP = Long Put, SP = Short Put, "
    "LC = Long Call, SC = Short Call"
)

# DTE override notation
DTE_HELP = (
    "Leave at 0 to use the DTE closest to the outlook period. "
    "Set a value to target a specific DTE for this trade."
)


def iv_adjustment(spot_now, spot_target, base_iv):
    """Adjust IV based on spot move."""
    pct_move = (spot_target - spot_now) / spot_now
    iv_shift = -pct_move * 0.40
    return max(base_iv + iv_shift, 0.05)


def estimate_iv(smile_df, strike, fallback):
    """Get IV from smile, with fallback."""
    if smile_df is not None and not smile_df.empty:
        iv = interpolate_smile_iv(smile_df, strike)
        if not np.isnan(iv) and 0.01 < iv < 2.0:
            return iv
    return fallback


def parse_legs(notation: str) -> list:
    """
    Parse leg notation string into list of leg dicts.

    Examples:
        "LP 6500" -> [{"type": "put", "long": True, "strike": 6500, "qty": 1}]
        "SC 7000 x2" -> [{"type": "call", "long": False, "strike": 7000, "qty": 2}]
        "SP 6800, SC 7100" -> [2 legs]
    """
    legs = []
    if not notation.strip():
        return legs

    parts = [p.strip() for p in notation.split(",")]

    for part in parts:
        # Match: prefix strike [xN]
        m = re.match(
            r'(LP|SP|LC|SC)\s+(\d+(?:\.\d+)?)\s*(?:[xX](\d+))?',
            part.strip(), re.IGNORECASE
        )
        if not m:
            continue

        prefix = m.group(1).upper()
        strike = float(m.group(2))
        qty = int(m.group(3)) if m.group(3) else 1

        is_long = prefix[0] == "L"
        opt_type = "put" if prefix[1] == "P" else "call"

        legs.append({
            "type": opt_type,
            "long": is_long,
            "strike": strike,
            "qty": qty,
        })

    return legs


def format_legs(legs: list) -> str:
    """Format legs back to readable string."""
    parts = []
    for leg in legs:
        prefix = ("L" if leg["long"] else "S") + ("P" if leg["type"] == "put" else "C")
        s = f"{prefix} {leg['strike']:,.0f}"
        if leg["qty"] > 1:
            s += f" x{leg['qty']}"
        parts.append(s)
    return ", ".join(parts)


def evaluate_trade(legs, spot, r, q, atm_iv, chain_data, expected_spot,
                    outlook_dte):
    """
    Evaluate a multi-leg trade across 3 scenarios.

    Returns dict with P&L, Greeks, cost, etc. or None on error.
    """
    chain = chain_data["chain"]
    actual_dte = chain_data["actual_dte"]
    T = actual_dte / 365.0
    remaining_after = actual_dte - outlook_dte
    if remaining_after < 1:
        remaining_after = 1
    T_after = remaining_after / 365.0

    move_delta = expected_spot - spot
    scenarios = {
        "full": expected_spot,
        "half": spot + move_delta * 0.5,
        "none": spot,
    }

    # Price each leg at entry
    total_cost = 0  # positive = debit (net paid), negative = credit (net received)
    total_delta = 0
    total_theta = 0
    total_vega = 0
    total_gamma = 0
    leg_details = []

    chain_calls = chain["calls"] if "calls" in chain else pd.DataFrame()
    chain_puts = chain["puts"] if "puts" in chain else pd.DataFrame()

    for leg in legs:
        chain_df = chain_calls if leg["type"] == "call" else chain_puts
        smile_df = build_smile_curve(chain_df, spot) if not chain_df.empty else None
        iv = estimate_iv(smile_df, leg["strike"], atm_iv)

        res = bs.calculate_all(spot, leg["strike"], T, r, iv, q, leg["type"])
        price = res["price"]

        sign = 1 if leg["long"] else -1
        qty = leg["qty"]

        total_cost += sign * price * qty
        total_delta += res["delta"] * sign * qty
        total_theta += res["theta_daily"] * sign * qty
        total_vega += res["vega_pct"] * sign * qty
        total_gamma += res["gamma"] * sign * qty

        leg_details.append({
            "leg": leg,
            "iv": iv,
            "price": price,
            "smile_df": smile_df,
        })

    # P&L per scenario
    scenario_pnls = {}
    for label, target_S in scenarios.items():
        total_value = 0
        for ld in leg_details:
            leg = ld["leg"]
            adj_iv = iv_adjustment(spot, target_S, ld["iv"])
            res_at = bs.calculate_all(target_S, leg["strike"], T_after,
                                       r, adj_iv, q, leg["type"])
            sign = 1 if leg["long"] else -1
            total_value += res_at["price"] * sign * leg["qty"]

        scenario_pnls[label] = total_value - total_cost

    weighted_pnl = (scenario_pnls["full"] * 0.50 +
                     scenario_pnls["half"] * 0.35 +
                     scenario_pnls["none"] * 0.15)

    # Capital at risk: for long-heavy trades = net debit; for short-heavy = margin estimate
    if total_cost > 0:
        capital = total_cost  # net debit
    else:
        # Net credit: capital at risk is harder to define
        # Use max loss across a range as proxy
        worst = 0
        for test_pct in [-0.10, -0.05, 0, 0.05, 0.10, 0.15]:
            test_S = spot * (1 + test_pct)
            test_val = 0
            for ld in leg_details:
                leg = ld["leg"]
                test_T = max(T_after, 1/365)
                try:
                    res_t = bs.calculate_all(test_S, leg["strike"], test_T,
                                              r, ld["iv"], q, leg["type"])
                    sign = 1 if leg["long"] else -1
                    test_val += res_t["price"] * sign * leg["qty"]
                except Exception:
                    pass
            test_pnl = test_val - total_cost
            worst = min(worst, test_pnl)
        capital = max(abs(worst), abs(total_cost), 1)

    roc = weighted_pnl / capital if capital > 0 else 0

    return {
        "cost": total_cost,
        "capital": capital,
        "delta": total_delta,
        "theta": total_theta,
        "vega": total_vega,
        "gamma": total_gamma,
        "pnl_full": scenario_pnls["full"],
        "pnl_half": scenario_pnls["half"],
        "pnl_none": scenario_pnls["none"],
        "weighted_pnl": weighted_pnl,
        "roc": roc,
        "dte": actual_dte,
        "expiration": chain_data["expiration"],
        "remaining_after": remaining_after,
        "legs_formatted": format_legs(legs),
        "leg_count": sum(l["qty"] for l in legs),
    }


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Trade Comparator")

    # ---- Market Outlook ----
    st.markdown("### Market Outlook")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
                                help="Underlying symbol.").upper()
    with c2:
        move_mode = st.radio("Target as", ["Price", "Percent"], horizontal=True)
    with c3:
        if move_mode == "Price":
            move_val = st.number_input("Expected Price", value=6300.0,
                                        min_value=100.0, max_value=99000.0,
                                        step=10.0, format="%.0f",
                                        help="Expected spot price after the move.")
        else:
            move_val = st.number_input("Expected Move %", value=-5.0,
                                        min_value=-50.0, max_value=50.0,
                                        step=0.5, format="%.1f")
    with c4:
        outlook_dte = st.number_input("Over DTE", value=20,
                                       min_value=1, max_value=365,
                                       help="Timeframe for expected move.")

    # ---- Trade Ideas ----
    st.markdown("### Trade Ideas")
    st.caption(NOTATION_HELP)

    trade_inputs = []
    for i in range(5):
        cols = st.columns([4, 1])
        with cols[0]:
            notation = st.text_input(
                f"Trade {i+1}",
                value="" if i > 0 else "LP 6500",
                key=f"trade_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
                placeholder=f"Trade {i+1}: e.g. SC 7000 x2" if i > 0 else "",
            )
        with cols[1]:
            dte_override = st.number_input(
                "DTE", value=0, min_value=0, max_value=365,
                key=f"dte_{i}",
                label_visibility="collapsed" if i > 0 else "visible",
                help=DTE_HELP if i == 0 else None,
            )
        if notation.strip():
            legs = parse_legs(notation)
            if legs:
                trade_inputs.append({
                    "notation": notation,
                    "legs": legs,
                    "dte_override": dte_override,
                    "label": format_legs(legs),
                })

    run = st.button("Compare", type="primary", use_container_width=True)

    if run:
        if not trade_inputs:
            st.warning("Enter at least one trade.")
            return
        with st.spinner("Evaluating trades..."):
            try:
                result = _compute(symbol, move_mode, move_val, outlook_dte,
                                   trade_inputs)
                st.session_state["tc_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "tc_result" not in st.session_state:
        st.info("Enter trades and click Compare.")
        return

    _display(st.session_state["tc_result"])


def _compute(symbol, move_mode, move_val, outlook_dte, trade_inputs):
    """Evaluate all trades."""
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(outlook_dte)
    vix = get_vix()
    atm_iv = vix / 100.0 if not np.isnan(vix) else 0.20

    if move_mode == "Percent":
        expected_spot = spot * (1 + move_val / 100.0)
    else:
        expected_spot = move_val

    # Get chains for each trade's DTE
    results = []
    for ti in trade_inputs:
        target_dte = ti["dte_override"] if ti["dte_override"] > 0 else int(outlook_dte * 2.5)
        target_dte = max(target_dte, outlook_dte + 5)

        try:
            chain, chain_ticker, spy_fallback = resolve_options_chain(
                symbol, target_dte)
        except Exception:
            continue

        working_spot = spot
        if spy_fallback:
            from data_provider import get_spot_price
            spy_spot = get_spot_price("SPY")
            scale = spot / spy_spot
            working_spot = spy_spot
        else:
            scale = 1.0

        chain_data = {
            "chain": chain,
            "actual_dte": chain["dte_actual"],
            "expiration": chain["expiration"],
        }

        # Adjust strikes for SPY fallback
        working_legs = []
        for leg in ti["legs"]:
            wl = leg.copy()
            if spy_fallback:
                wl["strike"] = leg["strike"] / scale
            working_legs.append(wl)

        working_expected = expected_spot / scale if spy_fallback else expected_spot

        ev = evaluate_trade(
            working_legs, working_spot, r, q, atm_iv, chain_data,
            working_expected, outlook_dte
        )

        if ev is None:
            continue

        # Scale back
        if spy_fallback:
            ev["cost"] *= scale
            ev["capital"] *= scale
            ev["pnl_full"] *= scale
            ev["pnl_half"] *= scale
            ev["pnl_none"] *= scale
            ev["weighted_pnl"] *= scale

        ev["label"] = ti["label"]
        ev["notation"] = ti["notation"]
        ev["legs"] = ti["legs"]
        results.append(ev)

    return {
        "symbol": symbol,
        "spot": spot,
        "expected_spot": expected_spot,
        "outlook_dte": outlook_dte,
        "vix": vix,
        "r": r,
        "q": q,
        "trades": results,
    }


def _display(res):
    """Display comparison results."""
    spot = res["spot"]
    move_pct = ((res["expected_spot"] / spot) - 1) * 100

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  "
                f"Target {res['expected_spot']:,.0f} ({move_pct:+.1f}%) "
                f"in {res['outlook_dte']} days  |  VIX {res['vix']:.1f}")

    trades = res["trades"]
    if not trades:
        st.warning("No trades could be evaluated.")
        return

    def pnl_fmt(val, ref):
        ref = abs(ref) if ref != 0 else 1
        pct = val / ref * 100
        return f"${val:+,.0f} ({pct:+.0f}%)"

    # ---- Comparison Table ----
    st.markdown("### Comparison")
    rows = []
    for i, t in enumerate(trades):
        cost_label = f"${t['cost']:+,.0f}"
        if t["cost"] > 0:
            cost_label += " debit"
        elif t["cost"] < 0:
            cost_label += " credit"

        ref = t["capital"]
        rows.append({
            "#": i + 1,
            "Trade": t["label"],
            "DTE": t["dte"],
            "Cost": cost_label,
            "Full Move": pnl_fmt(t["pnl_full"], ref),
            "Half Move": pnl_fmt(t["pnl_half"], ref),
            "No Move": pnl_fmt(t["pnl_none"], ref),
            "ROC": f"{t['roc']*100:+.0f}%",
            "Delta": f"{t['delta']:.2f}",
            "Theta/d": f"${t['theta']:.2f}",
            "Vega": f"${t['vega']:.2f}",
        })

    df = pd.DataFrame(rows)

    # Highlight best ROC
    best_idx = max(range(len(trades)), key=lambda i: trades[i]["roc"])

    def highlight(row):
        if row["#"] == best_idx + 1:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1),
                  use_container_width=True, hide_index=True)

    # ---- OptionStrat links ----
    link_cols = st.columns(min(len(trades), 5))
    for i, t in enumerate(trades):
        with link_cols[i]:
            os_legs = []
            for leg in t["legs"]:
                for _ in range(leg["qty"]):
                    os_legs.append({
                        "strike": leg["strike"],
                        "option_type": leg["type"],
                        "expiration": t["expiration"],
                        "long": leg["long"],
                    })
            url = bs.optionstrat_url(res["symbol"], os_legs)
            if url:
                st.caption(f"[#{i+1} OptionStrat]({url})")

    # ---- Recommendation ----
    st.markdown("---")
    best = trades[best_idx]
    st.success(
        f"Best: #{best_idx+1} {best['label']} "
        f"({best['dte']} DTE). "
        f"P&L: {pnl_fmt(best['pnl_full'], best['capital'])} (full) / "
        f"{pnl_fmt(best['pnl_half'], best['capital'])} (half) / "
        f"{pnl_fmt(best['pnl_none'], best['capital'])} (no move). "
        f"ROC: {best['roc']*100:+.0f}%"
    )

    # ---- Chart ----
    st.markdown("---")
    st.markdown("### Scenario Comparison")

    import plotly.graph_objects as go

    labels = [f"#{i+1} {t['label']}" for i, t in enumerate(trades)]
    full_vals = [t["pnl_full"] for t in trades]
    half_vals = [t["pnl_half"] for t in trades]
    none_vals = [t["pnl_none"] for t in trades]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Full Move", x=labels, y=full_vals,
        marker_color="rgba(50,180,80,0.7)",
        text=[f"${v:+,.0f}" for v in full_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Half Move", x=labels, y=half_vals,
        marker_color="rgba(100,150,255,0.7)",
        text=[f"${v:+,.0f}" for v in half_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="No Move", x=labels, y=none_vals,
        marker_color="rgba(200,200,200,0.7)",
        text=[f"${v:+,.0f}" for v in none_vals],
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig.update_layout(
        template="plotly_white",
        height=400,
        barmode="group",
        margin=dict(l=50, r=20, t=30, b=40),
        yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


main()
