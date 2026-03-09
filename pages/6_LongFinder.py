"""
Long Finder - Find the optimal long option for an expected move.

Given a market outlook (expected price + timeframe), finds the most
capital-efficient OTM option: maximum return on investment with
minimal time decay and sufficient DTE buffer.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    resolve_spot_price, get_available_expirations, get_options_chain,
    get_dividend_yield, get_risk_free_rate, get_vix,
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


def iv_adjustment(spot_now, spot_target, base_iv):
    """Adjust IV based on spot move. ~4 vol pts per 10% spot drop."""
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


def price_option(S, K, T, r, sigma, q, opt_type):
    """BS price + Greeks. Returns None on error."""
    if T <= 0 or sigma <= 0:
        return None
    try:
        return bs.calculate_all(S, K, T, r, sigma, q, opt_type)
    except Exception:
        return None


def find_candidates(symbol, spot, opt_type, expected_spot, outlook_dte,
                     r, q, atm_iv, max_budget):
    """Scan all expirations and OTM strikes for best long options."""
    expirations, chain_ticker = get_available_expirations(symbol)
    today = pd.Timestamp.now().normalize().date()

    move_delta = expected_spot - spot
    scenarios = {
        "full": expected_spot,
        "half": spot + move_delta * 0.5,
        "none": spot,
    }
    weights = {"full": 0.50, "half": 0.35, "none": 0.15}

    candidates = []

    for exp_str in expirations:
        import datetime
        exp_date = datetime.date.fromisoformat(exp_str)
        new_dte = (exp_date - today).days

        # DTE must exceed outlook + buffer
        if new_dte <= outlook_dte or new_dte > max(outlook_dte * 5, 180):
            continue

        try:
            chain = get_options_chain(chain_ticker, new_dte)
        except Exception:
            continue

        chain_df = chain["calls"] if opt_type == "call" else chain["puts"]
        if chain_df.empty:
            continue

        smile_df = build_smile_curve(chain_df, spot)
        actual_dte = chain["dte_actual"]
        T = actual_dte / 365.0
        remaining_after = actual_dte - outlook_dte
        if remaining_after < 1:
            continue
        T_after = remaining_after / 365.0

        available_strikes = sorted(chain_df["strike"].unique())

        # OTM only
        if opt_type == "put":
            otm_strikes = [k for k in available_strikes if k < spot]
        else:
            otm_strikes = [k for k in available_strikes if k > spot]

        # Limit range: not too far OTM (worthless) or too close to ATM (expensive)
        if opt_type == "put":
            otm_strikes = [k for k in otm_strikes if k > spot * 0.75]
        else:
            otm_strikes = [k for k in otm_strikes if k < spot * 1.25]

        if len(otm_strikes) > 20:
            step_n = max(1, len(otm_strikes) // 15)
            otm_strikes = otm_strikes[::step_n]

        for strike in otm_strikes:
            iv = estimate_iv(smile_df, strike, atm_iv)
            res = price_option(spot, strike, T, r, iv, q, opt_type)
            if res is None:
                continue

            price = res["price"]
            if price < 0.10:
                continue
            if max_budget > 0 and price > max_budget:
                continue

            # P&L for each scenario
            scenario_pnls = {}
            for label, target_S in scenarios.items():
                adj_iv = iv_adjustment(spot, target_S, iv)
                res_at = price_option(target_S, strike, T_after, r, adj_iv, q, opt_type)
                if res_at is None:
                    scenario_pnls[label] = -price
                    continue
                scenario_pnls[label] = res_at["price"] - price

            weighted_pnl = sum(scenario_pnls[k] * weights[k] for k in scenarios)

            # Return on capital
            roc = weighted_pnl / price

            # DTE buffer factor
            dte_ratio = min(remaining_after / max(outlook_dte, 1), 3.0)
            dte_factor = 0.5 + 0.5 * min(dte_ratio / 2.0, 1.0)

            # Theta efficiency: how much daily decay relative to potential gain
            theta_daily = abs(res["theta_daily"])
            if weighted_pnl > 0 and theta_daily > 0:
                # Days of theta to break even on weighted P&L
                theta_ratio = weighted_pnl / theta_daily
                theta_factor = min(theta_ratio / outlook_dte, 2.0) / 2.0
            else:
                theta_factor = 0.1

            score = roc * dte_factor * theta_factor * 1000

            candidates.append({
                "strike": strike,
                "dte": actual_dte,
                "expiration": chain["expiration"],
                "iv": iv,
                "price": price,
                "delta": res["delta"],
                "gamma": res["gamma"],
                "theta": res["theta_daily"],
                "vega": res["vega_pct"],
                "pnl_full": scenario_pnls["full"],
                "pnl_half": scenario_pnls["half"],
                "pnl_none": scenario_pnls["none"],
                "weighted_pnl": weighted_pnl,
                "roc": roc,
                "remaining_after": remaining_after,
                "score": score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Long Finder")

    # ---- Inputs ----
    st.markdown("### Market Outlook")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
                                help="Underlying symbol.").upper()
    with c2:
        opt_type_sel = st.selectbox("Type", ["Put", "Call"],
                                     help="Put = profit from drop. Call = profit from rise.")
    with c3:
        move_mode = st.radio("Target as", ["Price", "Percent"], horizontal=True,
                              help="Specify expected move as target price or percentage.")
    with c4:
        if move_mode == "Price":
            move_val = st.number_input("Expected Price", value=6300.0,
                                        min_value=100.0, max_value=99000.0,
                                        step=10.0, format="%.0f",
                                        help="Expected spot price after the move.")
        else:
            move_val = st.number_input("Expected Move %", value=-5.0,
                                        min_value=-50.0, max_value=50.0,
                                        step=0.5, format="%.1f",
                                        help="Expected change. -5 = down 5%.")
    with c5:
        outlook_dte = st.number_input("Over DTE", value=20,
                                       min_value=1, max_value=365,
                                       help="Timeframe for expected move.")

    c6, c7 = st.columns([1, 1])
    with c6:
        max_budget = st.number_input("Max Price ($)", value=0.0,
                                      min_value=0.0, max_value=10000.0,
                                      step=10.0, format="%.0f",
                                      help="Maximum option price. 0 = no limit.")
    with c7:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Find", type="primary", use_container_width=True)

    opt_type = opt_type_sel.lower()

    if run:
        with st.spinner("Scanning options chains..."):
            try:
                result = _compute(symbol, opt_type, move_mode, move_val,
                                   outlook_dte, max_budget)
                st.session_state["lf_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "lf_result" not in st.session_state:
        st.info("Enter your outlook and click Find.")
        return

    _display(st.session_state["lf_result"])


def _compute_short_alternatives(symbol, spot, short_type, expected_spot,
                                  outlook_dte, r, q, atm_iv, long_candidates):
    """
    Compute short option alternatives that match the top long pick's P&L.

    For each OTM short strike, calculate:
    - Premium per contract
    - How many contracts needed to match the long P&L
    - P&L at full/half/no move
    - Risk at adverse move (+/- 5%)
    """
    if not long_candidates:
        return []

    best_long = long_candidates[0]
    target_pnl = best_long["pnl_full"]  # match the full-move P&L

    if target_pnl <= 0:
        return []

    # Get chain for ~2x outlook DTE
    target_dte = max(outlook_dte * 2, 30)
    try:
        expirations, chain_ticker = get_available_expirations(symbol)
    except Exception:
        return []

    today = pd.Timestamp.now().normalize().date()
    import datetime

    results = []

    for exp_str in expirations:
        exp_date = datetime.date.fromisoformat(exp_str)
        new_dte = (exp_date - today).days
        if new_dte <= outlook_dte or new_dte > max(outlook_dte * 4, 120):
            continue

        try:
            chain = get_options_chain(chain_ticker, new_dte)
        except Exception:
            continue

        chain_df = chain["calls"] if short_type == "call" else chain["puts"]
        if chain_df.empty:
            continue

        smile_df = build_smile_curve(chain_df, spot)
        actual_dte = chain["dte_actual"]
        T = actual_dte / 365.0
        remaining_after = actual_dte - outlook_dte
        if remaining_after < 1:
            continue
        T_after = remaining_after / 365.0

        available_strikes = sorted(chain_df["strike"].unique())

        # OTM strikes only
        if short_type == "call":
            otm = [k for k in available_strikes if k > spot * 1.02 and k < spot * 1.20]
        else:
            otm = [k for k in available_strikes if k < spot * 0.98 and k > spot * 0.80]

        if len(otm) > 10:
            step_n = max(1, len(otm) // 8)
            otm = otm[::step_n]

        for strike in otm:
            iv = estimate_iv(smile_df, strike, atm_iv)
            res = price_option(spot, strike, T, r, iv, q, short_type)
            if res is None or res["price"] < 0.50:
                continue

            premium = res["price"]

            # How many to match the long P&L?
            # Short P&L at target = premium - value_at_target (per contract)
            adj_iv_full = iv_adjustment(spot, expected_spot, iv)
            res_full = price_option(expected_spot, strike, T_after, r, adj_iv_full, q, short_type)
            if res_full is None:
                continue

            pnl_per_contract = premium - res_full["price"]
            if pnl_per_contract <= 0:
                continue

            qty_needed = int(np.ceil(target_pnl / pnl_per_contract))
            qty_needed = max(1, min(qty_needed, 50))  # sanity cap

            # Actual P&L with that quantity across scenarios
            move_delta = expected_spot - spot
            scenarios_S = {
                "full": expected_spot,
                "half": spot + move_delta * 0.5,
                "none": spot,
            }

            scenario_pnls = {}
            for label, target_S in scenarios_S.items():
                adj_iv = iv_adjustment(spot, target_S, iv)
                res_at = price_option(target_S, strike, T_after, r, adj_iv, q, short_type)
                if res_at is None:
                    scenario_pnls[label] = premium * qty_needed
                    continue
                scenario_pnls[label] = (premium - res_at["price"]) * qty_needed

            # Risk at adverse move (opposite of expected direction)
            if expected_spot < spot:
                # Bearish outlook, short calls -> risk is spot going UP
                adverse_spot = spot * 1.05
            else:
                # Bullish outlook, short puts -> risk is spot going DOWN
                adverse_spot = spot * 0.95

            adj_iv_adv = iv_adjustment(spot, adverse_spot, iv)
            res_adv = price_option(adverse_spot, strike, T_after, r, adj_iv_adv, q, short_type)
            if res_adv:
                risk_adverse = (premium - res_adv["price"]) * qty_needed
            else:
                risk_adverse = -premium * qty_needed

            dist_pct = abs(strike - spot) / spot * 100

            results.append({
                "strike": strike,
                "dte": actual_dte,
                "expiration": chain["expiration"],
                "type": short_type,
                "iv": iv,
                "premium": premium,
                "delta": res["delta"],
                "qty_needed": qty_needed,
                "total_premium": premium * qty_needed,
                "pnl_full": scenario_pnls["full"],
                "pnl_half": scenario_pnls["half"],
                "pnl_none": scenario_pnls["none"],
                "risk_adverse": risk_adverse,
                "dist_pct": dist_pct,
            })

        # Only use first matching expiration to keep it simple
        if results:
            break

    results.sort(key=lambda x: x["dist_pct"])
    return results[:5]


def _compute(symbol, opt_type, move_mode, move_val, outlook_dte, max_budget):
    """Fetch data and find optimal long options."""
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(outlook_dte)
    vix = get_vix()
    atm_iv = vix / 100.0 if not np.isnan(vix) else 0.20

    if move_mode == "Percent":
        expected_spot = spot * (1 + move_val / 100.0)
    else:
        expected_spot = move_val

    candidates = find_candidates(
        symbol, spot, opt_type, expected_spot, outlook_dte,
        r, q, atm_iv, max_budget
    )

    # --- Short Alternative ---
    # For bearish outlook (Long Put): compare with Short Calls
    # For bullish outlook (Long Call): compare with Short Puts
    short_type = "call" if opt_type == "put" else "put"
    short_alt = _compute_short_alternatives(
        symbol, spot, short_type, expected_spot, outlook_dte,
        r, q, atm_iv, candidates
    )

    return {
        "symbol": symbol,
        "spot": spot,
        "opt_type": opt_type,
        "expected_spot": expected_spot,
        "outlook_dte": outlook_dte,
        "r": r,
        "q": q,
        "vix": vix,
        "candidates": candidates[:30],
        "short_alt": short_alt,
    }


def _display(res):
    """Display results."""
    spot = res["spot"]
    move_pct = ((res["expected_spot"] / spot) - 1) * 100
    type_label = res["opt_type"].upper()
    top = res["candidates"][:3]

    # Header
    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  Long {type_label}  |  "
                f"Target {res['expected_spot']:,.0f} ({move_pct:+.1f}%) "
                f"in {res['outlook_dte']} days")

    m1, m2 = st.columns(2)
    m1.metric("VIX", f"{res['vix']:.1f}")
    m2.metric("Candidates", f"{len(res['candidates'])}")

    if not top:
        st.warning("No suitable options found. Try a wider timeframe or different target.")
        return

    # ---- Top 3 Table ----
    st.markdown("---")
    st.markdown("### Top Picks")

    def pnl_fmt(val, ref):
        if ref > 0:
            pct = val / ref * 100
            return f"${val:+,.0f} ({pct:+.0f}%)"
        return f"${val:+,.0f}"

    rows = []
    for i, c in enumerate(top):
        rows.append({
            "#": i + 1,
            "Strike": f"{c['strike']:,.0f}",
            "DTE": c["dte"],
            "Exp": c["expiration"],
            "Price": f"${c['price']:.2f}",
            "Full Move": pnl_fmt(c["pnl_full"], c["price"]),
            "Half Move": pnl_fmt(c["pnl_half"], c["price"]),
            "No Move": pnl_fmt(c["pnl_none"], c["price"]),
            "ROC": f"{c['roc']*100:+.0f}%",
            "Delta": f"{c['delta']:.3f}",
            "Theta/d": f"${c['theta']:.4f}",
            "IV": f"{c['iv']*100:.1f}%",
            "Score": f"{c['score']:.0f}",
        })

    df = pd.DataFrame(rows)

    def highlight_top(row):
        if row["#"] == 1:
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight_top, axis=1),
                  use_container_width=True, hide_index=True)

    # ---- OptionStrat links ----
    link_cols = st.columns(min(len(top), 3))
    for i, c in enumerate(top):
        with link_cols[i]:
            url = bs.optionstrat_url(res["symbol"], [{
                "strike": c["strike"],
                "option_type": res["opt_type"],
                "expiration": c["expiration"],
                "long": True,
            }])
            if url:
                st.caption(f"[#{i+1} OptionStrat]({url})")

    # ---- Recommendation ----
    st.markdown("---")
    best = top[0]
    st.success(
        f"Long {type_label} {best['strike']:,.0f} @ ${best['price']:.2f} "
        f"({best['dte']} DTE, {best['expiration']}). "
        f"P&L: {pnl_fmt(best['pnl_full'], best['price'])} (full) / "
        f"{pnl_fmt(best['pnl_half'], best['price'])} (half) / "
        f"{pnl_fmt(best['pnl_none'], best['price'])} (no move). "
        f"ROC: {best['roc']*100:+.0f}%"
    )

    # ---- All candidates ----
    if len(res["candidates"]) > 3:
        with st.expander(f"All Candidates ({len(res['candidates'])})"):
            all_rows = []
            for c in res["candidates"]:
                all_rows.append({
                    "Strike": c["strike"],
                    "DTE": c["dte"],
                    "Exp": c["expiration"],
                    "Price": f"${c['price']:.2f}",
                    "IV": f"{c['iv']*100:.1f}%",
                    "Delta": f"{c['delta']:.3f}",
                    "Theta/d": f"${c['theta']:.4f}",
                    "Full": f"${c['pnl_full']:+,.0f}",
                    "Half": f"${c['pnl_half']:+,.0f}",
                    "None": f"${c['pnl_none']:+,.0f}",
                    "ROC": f"{c['roc']*100:+.0f}%",
                    "Score": f"{c['score']:.0f}",
                })
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True,
                         hide_index=True)

    # ---- Short Alternative ----
    short_alt = res.get("short_alt", [])
    if short_alt and top:
        best_long = top[0]
        short_type_label = "Short Call" if res["opt_type"] == "put" else "Short Put"

        st.markdown("---")
        st.markdown(f"### Short Alternative: {short_type_label}s")
        st.caption(
            f"How many {short_type_label}s would match the "
            f"${best_long['pnl_full']:+,.0f} P&L of the top Long pick?"
        )

        sa_rows = []
        for s in short_alt:
            sa_rows.append({
                "Strike": f"{s['strike']:,.0f}",
                "Dist": f"{s['dist_pct']:.1f}%",
                "DTE": s["dte"],
                "Premium": f"${s['premium']:.2f}",
                "Qty": s["qty_needed"],
                "Total Credit": f"${s['total_premium']:,.0f}",
                "Full Move": f"${s['pnl_full']:+,.0f}",
                "Half Move": f"${s['pnl_half']:+,.0f}",
                "No Move": f"${s['pnl_none']:+,.0f}",
                "Risk (+/-5%)": f"${s['risk_adverse']:+,.0f}",
            })

        st.dataframe(pd.DataFrame(sa_rows), use_container_width=True,
                      hide_index=True)

        # Summary comparison
        best_short = short_alt[0]
        st.markdown("#### Long vs Short")
        comp_rows = [{
            "Strategy": f"Long {res['opt_type'].upper()} {best_long['strike']:,.0f}",
            "Cost": f"${best_long['price']:.0f} debit",
            "Contracts": "1",
            "Full Move": f"${best_long['pnl_full']:+,.0f}",
            "No Move": f"${best_long['pnl_none']:+,.0f}",
            "Risk": f"${best_long['price']:.0f} (max loss)",
        }, {
            "Strategy": f"{best_short['qty_needed']}x {short_type_label} {best_short['strike']:,.0f}",
            "Cost": f"${best_short['total_premium']:,.0f} credit",
            "Contracts": str(best_short["qty_needed"]),
            "Full Move": f"${best_short['pnl_full']:+,.0f}",
            "No Move": f"${best_short['pnl_none']:+,.0f}",
            "Risk": f"${best_short['risk_adverse']:+,.0f} (at 5% adverse)",
        }]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                      hide_index=True)

    # ---- Scenario Comparison Chart ----
    if top:
        st.markdown("---")
        st.markdown("### Scenario Comparison")

        import plotly.graph_objects as go

        labels = []
        full_vals = []
        half_vals = []
        none_vals = []

        # Long picks
        for c in top:
            labels.append(f"L {c['strike']:,.0f}\n{c['dte']}d (${c['price']:.0f})")
            full_vals.append(c["pnl_full"])
            half_vals.append(c["pnl_half"])
            none_vals.append(c["pnl_none"])

        # Short alternatives
        short_alt = res.get("short_alt", [])
        for s in short_alt[:3]:
            short_label = "SC" if s["type"] == "call" else "SP"
            labels.append(f"{s['qty_needed']}x {short_label} {s['strike']:,.0f}\n{s['dte']}d (${s['total_premium']:.0f} cr)")
            full_vals.append(s["pnl_full"])
            half_vals.append(s["pnl_half"])
            none_vals.append(s["pnl_none"])

        # Colors: green/blue/gray for scenarios, distinguish long vs short by bar pattern
        n_long = len(top)
        n_total = len(labels)

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

        # Divider between long and short sections
        if short_alt and n_long < n_total:
            fig.add_vline(x=n_long - 0.5, line_dash="dot", line_color="red",
                          line_width=1.5,
                          annotation_text="Long | Short",
                          annotation_position="top")

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
        fig.update_layout(
            template="plotly_white",
            height=450,
            barmode="group",
            margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title="P&L ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)


main()
