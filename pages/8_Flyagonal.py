"""
Flyagonal Builder - Optimal positioning for the Flyagonal strategy.

The Flyagonal combines:
1. Call Broken-Wing Butterfly (short DTE): +1 LC, -2 SC, +1 LC (wider upper wing)
2. Put Diagonal: -1 SP (short DTE), +1 LP (longer DTE), same strike

Target: near-zero delta, positive theta, balanced vega, defined risk.
Typical: 8-10 DTE shorts, 16-20 DTE long put.
"""

import streamlit as st
import numpy as np
import pandas as pd
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


def estimate_iv(smile_df, strike, fallback):
    if smile_df is not None and not smile_df.empty:
        iv = interpolate_smile_iv(smile_df, strike)
        if not np.isnan(iv) and 0.01 < iv < 2.0:
            return iv
    return fallback


def round_strike(strike, step):
    """Round strike to nearest tradeable increment."""
    return round(strike / step) * step


def find_best_flyagonal(spot, r, q, short_dte, long_dte, iv_atm,
                         call_smile, put_smile_short, put_smile_long,
                         strike_step, bwb_delta, wing_steps_lower,
                         wing_steps_upper):
    """
    Find optimal Flyagonal strikes using delta + symmetry.

    1. BWB lower call at target delta -> finds strike just above spot
    2. Put at symmetric offset below spot (same distance as BWB is above)
    3. Wings defined as multiples of strike_step (scales to any underlying)
    """
    T_s = short_dte / 365.0
    T_l = long_dte / 365.0

    # --- BWB Lower Call: solve by delta ---
    try:
        bwb_k1_raw = bs.solve_strike_for_delta(bwb_delta, spot, T_s, r, iv_atm, q, "call")
    except Exception:
        return None

    bwb_k1 = round_strike(bwb_k1_raw, strike_step)

    # --- BWB Wings: relative to K1 ---
    wing_lower = strike_step * wing_steps_lower
    wing_upper = strike_step * wing_steps_upper
    bwb_k2 = bwb_k1 + wing_lower
    bwb_k3 = bwb_k2 + wing_upper

    if bwb_k1 >= bwb_k2 or bwb_k2 >= bwb_k3:
        return None

    # --- Put: symmetric offset below spot ---
    bwb_offset = bwb_k1 - spot
    put_k = round_strike(spot - bwb_offset, strike_step)

    # --- Price all legs with smile IV ---
    iv_c1 = estimate_iv(call_smile, bwb_k1, iv_atm)
    iv_c2 = estimate_iv(call_smile, bwb_k2, iv_atm)
    iv_c3 = estimate_iv(call_smile, bwb_k3, iv_atm)
    iv_sp = estimate_iv(put_smile_short, put_k, iv_atm)
    iv_lp = estimate_iv(put_smile_long, put_k, iv_atm)

    c1 = bs.calculate_all(spot, bwb_k1, T_s, r, iv_c1, q, "call")
    c2 = bs.calculate_all(spot, bwb_k2, T_s, r, iv_c2, q, "call")
    c3 = bs.calculate_all(spot, bwb_k3, T_s, r, iv_c3, q, "call")
    sp = bs.calculate_all(spot, put_k, T_s, r, iv_sp, q, "put")
    lp = bs.calculate_all(spot, put_k, T_l, r, iv_lp, q, "put")

    # Position: +1 c1, -2 c2, +1 c3, -1 sp, +1 lp
    signs = [1, -2, 1, -1, 1]
    legs = [c1, c2, c3, sp, lp]

    delta = sum(s * l["delta"] for s, l in zip(signs, legs))
    theta = sum(s * l["theta_daily"] for s, l in zip(signs, legs))
    vega = sum(s * l["vega_pct"] for s, l in zip(signs, legs))
    gamma = sum(s * l["gamma"] for s, l in zip(signs, legs))

    # Cost = what we pay (long) minus what we receive (short)
    cost = (c1["price"] - 2 * c2["price"] + c3["price"]
            - sp["price"] + lp["price"])

    # P&L at expiry range (short DTE only, diagonal still has time value)
    pnl_at_expiry = []
    test_range = np.linspace(spot * 0.92, spot * 1.08, 100)
    for test_S in test_range:
        # BWB at expiry
        bwb_val = (max(test_S - bwb_k1, 0)
                   - 2 * max(test_S - bwb_k2, 0)
                   + max(test_S - bwb_k3, 0))
        # Put diagonal: short put at expiry, long put still has time
        sp_val = -max(put_k - test_S, 0)  # short put liability
        # Long put: still has (long_dte - short_dte) days left
        lp_remaining_T = (long_dte - short_dte) / 365.0
        iv_adj = iv_lp * (1 + 0.4 * max(0, (spot - test_S) / spot))
        try:
            lp_val_res = bs.calculate_all(test_S, put_k, lp_remaining_T, r, iv_adj, q, "put")
            lp_val = lp_val_res["price"]
        except Exception:
            lp_val = max(put_k - test_S, 0)

        total_val = bwb_val + sp_val + lp_val - cost
        pnl_at_expiry.append({"spot": test_S, "pnl": total_val})

    # Max profit and max loss
    pnl_values = [p["pnl"] for p in pnl_at_expiry]
    max_profit = max(pnl_values)
    max_loss = min(pnl_values)

    # Profit zone (where pnl > 0)
    profitable = [p["spot"] for p in pnl_at_expiry if p["pnl"] > 0]
    if profitable:
        profit_zone_low = min(profitable)
        profit_zone_high = max(profitable)
    else:
        profit_zone_low = profit_zone_high = spot

    wing_lower = bwb_k2 - bwb_k1
    wing_upper = bwb_k3 - bwb_k2

    return {
        "bwb_k1": bwb_k1, "bwb_k2": bwb_k2, "bwb_k3": bwb_k3,
        "put_k": put_k,
        "iv_c1": iv_c1, "iv_c2": iv_c2, "iv_c3": iv_c3,
        "iv_sp": iv_sp, "iv_lp": iv_lp,
        "c1": c1, "c2": c2, "c3": c3, "sp": sp, "lp": lp,
        "delta": delta, "theta": theta, "vega": vega, "gamma": gamma,
        "cost": cost,
        "wing_lower": wing_lower, "wing_upper": wing_upper,
        "max_profit": max_profit, "max_loss": max_loss,
        "profit_zone_low": profit_zone_low,
        "profit_zone_high": profit_zone_high,
        "pnl_curve": pnl_at_expiry,
        "short_dte": short_dte, "long_dte": long_dte,
    }


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Flyagonal Builder")
    st.caption("Call Broken-Wing Butterfly + Put Diagonal. "
               "Targets near-zero delta, positive theta, balanced vega.")

    # ---- Inputs ----
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
                                help="Underlying symbol.").upper()
    with c2:
        strike_step = st.number_input("Strike Step", value=5, min_value=1, max_value=50,
                                       help="Strike increment for rounding.")
    with c3:
        mode = st.radio("Mode", ["Auto (optimize DTEs)", "Manual DTEs"],
                         horizontal=True,
                         help="Auto scans all DTE combinations and picks the best. "
                              "Manual lets you set specific DTEs.")

    if mode == "Manual DTEs":
        mc1, mc2 = st.columns(2)
        with mc1:
            short_dte = st.number_input("Short DTE", value=9, min_value=5, max_value=30,
                                         help="DTE for BWB calls and short put. Typical: 8-10.")
        with mc2:
            long_dte = st.number_input("Long DTE", value=18, min_value=10, max_value=60,
                                        help="DTE for the long put (diagonal). Typical: 16-20.")
    else:
        short_dte = 0
        long_dte = 0

    st.markdown("#### Positioning")
    d1, d2, d3 = st.columns(3)
    with d1:
        bwb_delta = st.number_input("BWB Delta", value=0.45,
                                     min_value=0.20, max_value=0.55, step=0.05,
                                     format="%.2f",
                                     help="Delta for the lowest BWB call. Determines distance from Spot. "
                                          "Put is placed symmetrically below Spot. "
                                          "Higher = closer to ATM. Typical: 0.40-0.50.")
    with d2:
        wing_steps_lower = st.number_input("Wing Lower (steps)", value=5,
                                            min_value=2, max_value=20,
                                            help="Lower wing width as multiples of Strike Step. "
                                                 "E.g. 5 steps x 5pt = 25pt for SPX.")
    with d3:
        wing_steps_upper = st.number_input("Wing Upper (steps)", value=6,
                                            min_value=2, max_value=20,
                                            help="Upper wing width (broken side). Should be wider than lower. "
                                                 "E.g. 6 steps x 5pt = 30pt for SPX.")

    run = st.button("Build Flyagonal", type="primary", use_container_width=True)

    if run:
        with st.spinner("Computing optimal positioning..."):
            try:
                result = _compute(symbol, short_dte, long_dte, strike_step,
                                   bwb_delta, wing_steps_lower, wing_steps_upper,
                                   auto_mode=(mode != "Manual DTEs"))
                st.session_state["fly_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "fly_result" not in st.session_state:
        st.info("Set parameters and click Build Flyagonal.")
        return

    _display(st.session_state["fly_result"])


def _score_flyagonal(result):
    """
    Score a Flyagonal configuration.

    Optimizes for:
    1. Theta efficiency: high theta relative to cost (daily income)
    2. Profit zone width: wider = more robust
    3. Vega balance: positive vega preferred (benefits from vol spike)
    4. Low absolute delta: closer to neutral = safer
    5. Diagonal edge: theta difference between short and long put
    """
    if result is None:
        return -9999

    theta = result["theta"]
    cost = result["cost"]
    vega = result["vega"]
    delta = abs(result["delta"])
    zone_width = result["profit_zone_high"] - result["profit_zone_low"]
    spot = result.get("_spot", 6000)
    max_loss = abs(result["max_loss"]) if result["max_loss"] < 0 else 1

    # 1. Theta efficiency: theta / cost (how fast you earn back the debit)
    if cost > 0 and theta > 0:
        theta_eff = theta / cost
    elif cost <= 0 and theta > 0:
        theta_eff = theta * 2  # credit trade with positive theta = great
    else:
        theta_eff = 0

    # 2. Zone width as % of spot
    zone_pct = zone_width / spot * 100

    # 3. Vega bonus: positive vega is desirable (vol spike protection)
    vega_score = 1.0 + min(max(vega, 0), 3) * 0.1

    # 4. Delta penalty: further from zero = worse
    delta_penalty = max(0, 1.0 - delta * 5)

    # 5. Diagonal theta edge
    sp_theta = abs(result["sp"]["theta_daily"])
    lp_theta = abs(result["lp"]["theta_daily"])
    if lp_theta > 0:
        diag_edge = sp_theta / lp_theta  # >1 means short decays faster
    else:
        diag_edge = 1

    score = theta_eff * zone_pct * vega_score * delta_penalty * diag_edge * 100
    return score


def _compute(symbol, short_dte, long_dte, strike_step,
             bwb_delta, wing_steps_lower, wing_steps_upper,
             auto_mode=False):
    """Fetch data and compute Flyagonal, optionally optimizing DTEs."""
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(short_dte if short_dte > 0 else 10)
    vix = get_vix()
    iv_atm = vix / 100.0 if not np.isnan(vix) else 0.20

    import datetime
    today = pd.Timestamp.now().normalize().date()

    if auto_mode:
        # Get all available expirations
        try:
            expirations, chain_ticker = get_available_expirations(symbol)
        except Exception as e:
            raise ValueError(f"Could not fetch expirations: {e}")

        if not expirations:
            raise ValueError(f"No expirations available for {symbol}. "
                             "Try Manual DTEs mode or check the symbol.")

        exp_dtes = []
        for exp_str in expirations:
            try:
                exp_date = datetime.date.fromisoformat(exp_str)
                dte = (exp_date - today).days
                if 5 <= dte <= 60:
                    exp_dtes.append((dte, exp_str))
            except Exception:
                continue

        if not exp_dtes:
            raise ValueError(
                f"No expirations between 5-60 DTE found for {symbol}. "
                f"Available: {len(expirations)} total. "
                "Try Manual DTEs mode."
            )
        # Scan all valid short/long DTE combinations
        best_result = None
        best_score = -9999
        all_combos = []

        for s_dte, s_exp in exp_dtes:
            if s_dte < 5 or s_dte > 21:
                continue
            for l_dte, l_exp in exp_dtes:
                if l_dte <= s_dte or l_dte > s_dte * 4:
                    continue
                # Ratio: long should be 1.5x - 3x the short DTE
                ratio = l_dte / s_dte
                if ratio < 1.3 or ratio > 3.5:
                    continue

                try:
                    chain_s, _, _ = resolve_options_chain(symbol, s_dte)
                    chain_l, _, _ = resolve_options_chain(symbol, l_dte)
                except Exception:
                    continue

                cs = build_smile_curve(chain_s["calls"], spot)
                ps_s = build_smile_curve(chain_s["puts"], spot)
                ps_l = build_smile_curve(chain_l["puts"], spot)

                res = find_best_flyagonal(
                    spot, r, q, chain_s["dte_actual"], chain_l["dte_actual"],
                    iv_atm, cs, ps_s, ps_l, strike_step,
                    bwb_delta, wing_steps_lower, wing_steps_upper
                )
                if res is None:
                    continue

                res["_spot"] = spot
                score = _score_flyagonal(res)

                # Store full result for this combo
                res_copy = dict(res)
                res_copy["exp_short"] = chain_s["expiration"]
                res_copy["exp_long"] = chain_l["expiration"]
                res_copy["actual_short_dte"] = chain_s["dte_actual"]
                res_copy["actual_long_dte"] = chain_l["dte_actual"]
                res_copy["symbol"] = symbol
                res_copy["spot"] = spot
                res_copy["vix"] = vix
                res_copy["r"] = r
                res_copy["q"] = q

                all_combos.append({
                    "short_dte": chain_s["dte_actual"],
                    "long_dte": chain_l["dte_actual"],
                    "exp_short": chain_s["expiration"],
                    "exp_long": chain_l["expiration"],
                    "theta": res["theta"],
                    "delta": res["delta"],
                    "vega": res["vega"],
                    "cost": res["cost"],
                    "zone_width": res["profit_zone_high"] - res["profit_zone_low"],
                    "diag_edge": (abs(res["sp"]["theta_daily"]) /
                                  max(abs(res["lp"]["theta_daily"]), 0.001)),
                    "score": score,
                    "full_result": res_copy,
                })

                if score > best_score:
                    best_score = score
                    best_result = res
                    best_result["exp_short"] = chain_s["expiration"]
                    best_result["exp_long"] = chain_l["expiration"]
                    best_result["actual_short_dte"] = chain_s["dte_actual"]
                    best_result["actual_long_dte"] = chain_l["dte_actual"]

        if best_result is None:
            raise ValueError("No valid Flyagonal found across DTE combinations.")

        # Sort combos by score
        all_combos.sort(key=lambda x: x["score"], reverse=True)
        best_result["dte_scan"] = all_combos[:10]

    else:
        # Manual mode: use specified DTEs
        chain_short, _, _ = resolve_options_chain(symbol, short_dte)
        chain_long, _, _ = resolve_options_chain(symbol, long_dte)

        call_smile_short = build_smile_curve(chain_short["calls"], spot)
        put_smile_short = build_smile_curve(chain_short["puts"], spot)
        put_smile_long = build_smile_curve(chain_long["puts"], spot)

        best_result = find_best_flyagonal(
            spot, r, q, chain_short["dte_actual"], chain_long["dte_actual"],
            iv_atm, call_smile_short, put_smile_short, put_smile_long,
            strike_step, bwb_delta, wing_steps_lower, wing_steps_upper
        )

        if best_result is None:
            raise ValueError("Could not construct Flyagonal. Check delta targets.")

        best_result["exp_short"] = chain_short["expiration"]
        best_result["exp_long"] = chain_long["expiration"]
        best_result["actual_short_dte"] = chain_short["dte_actual"]
        best_result["actual_long_dte"] = chain_long["dte_actual"]
        best_result["dte_scan"] = None

    best_result["symbol"] = symbol
    best_result["spot"] = spot
    best_result["vix"] = vix
    best_result["r"] = r
    best_result["q"] = q

    return best_result


def _display(res):
    """Display Flyagonal results."""
    spot = res["spot"]

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")

    # ---- DTE Selection ----
    dte_scan = res.get("dte_scan")
    if dte_scan:
        st.markdown("### DTE Combinations")

        # Build options for selectbox
        options = []
        for i, s in enumerate(dte_scan):
            label = (f"#{i+1} | {s['short_dte']}/{s['long_dte']}d "
                     f"({s['long_dte']/s['short_dte']:.1f}x) | "
                     f"Theta ${s['theta']:.2f} | "
                     f"Zone {s['zone_width']:.0f}pts | "
                     f"Score {s['score']:.0f}")
            options.append(label)

        selected_idx = st.selectbox(
            "Select DTE combination",
            range(len(options)),
            format_func=lambda i: options[i],
            help="Choose a DTE combination to see its full structure, P&L chart, and links."
        )

        # Use the selected combo's full result
        selected = dte_scan[selected_idx]
        r = selected["full_result"]

        # Show summary table
        scan_rows = []
        for i, s in enumerate(dte_scan):
            scan_rows.append({
                "#": i + 1,
                "Short": s["short_dte"],
                "Long": s["long_dte"],
                "Ratio": f"{s['long_dte']/s['short_dte']:.1f}x",
                "Theta/d": f"${s['theta']:.2f}",
                "Delta": f"{s['delta']:.3f}",
                "Vega": f"${s['vega']:.2f}",
                "Cost": f"${s['cost']:.2f}",
                "Zone": f"{s['zone_width']:.0f} pts",
                "Score": f"{s['score']:.0f}",
            })

        scan_df = pd.DataFrame(scan_rows)

        def hl_selected(row):
            if row["#"] == selected_idx + 1:
                return ["background-color: #d4edda"] * len(row)
            return [""] * len(row)

        st.dataframe(scan_df.style.apply(hl_selected, axis=1),
                      use_container_width=True, hide_index=True)
    else:
        r = res

    # ---- Structure (for selected combo) ----
    _display_structure(r)


def _display_structure(res):
    """Display structure, Greeks, P&L chart, and export for one Flyagonal."""
    spot = res["spot"]

    st.markdown("### Structure")

    st.markdown(f"**Legs** (Short: {res['actual_short_dte']} DTE / {res['exp_short']}, "
                f"Long Put: {res['actual_long_dte']} DTE / {res['exp_long']})")
    leg_data = [
        {"Leg": "Long Call", "Strike": f"{res['bwb_k1']:,.0f}",
         "Delta": f"{res['c1']['delta']:.3f}", "IV": f"{res['iv_c1']*100:.1f}%",
         "Price": f"${res['c1']['price']:.2f}", "Qty": "+1",
         "Exp": res["exp_short"]},
        {"Leg": "Short Call", "Strike": f"{res['bwb_k2']:,.0f}",
         "Delta": f"{res['c2']['delta']:.3f}", "IV": f"{res['iv_c2']*100:.1f}%",
         "Price": f"${res['c2']['price']:.2f}", "Qty": "-2",
         "Exp": res["exp_short"]},
        {"Leg": "Long Call", "Strike": f"{res['bwb_k3']:,.0f}",
         "Delta": f"{res['c3']['delta']:.3f}", "IV": f"{res['iv_c3']*100:.1f}%",
         "Price": f"${res['c3']['price']:.2f}", "Qty": "+1",
         "Exp": res["exp_short"]},
        {"Leg": "Short Put", "Strike": f"{res['put_k']:,.0f}",
         "Delta": f"{res['sp']['delta']:.3f}", "IV": f"{res['iv_sp']*100:.1f}%",
         "Price": f"${res['sp']['price']:.2f}", "Qty": "-1",
         "Exp": res["exp_short"]},
        {"Leg": "Long Put", "Strike": f"{res['put_k']:,.0f}",
         "Delta": f"{res['lp']['delta']:.3f}", "IV": f"{res['iv_lp']*100:.1f}%",
         "Price": f"${res['lp']['price']:.2f}", "Qty": "+1",
         "Exp": res["exp_long"]},
    ]
    st.dataframe(pd.DataFrame(leg_data), use_container_width=True, hide_index=True)

    # ---- Combined Position ----
    st.markdown("### Combined Position")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Entry Cost", f"${res['cost']:.2f}",
              help="Per point. Multiply by 100 for SPX contract.")
    m2.metric("Delta", f"{res['delta']:.3f}")
    m3.metric("Theta/day", f"${res['theta']:.2f}")
    m4.metric("Vega/1%", f"${res['vega']:.2f}")
    m5.metric("Gamma", f"{res['gamma']:.5f}")
    m6.metric("Wings", f"{res['wing_lower']:.0f} / {res['wing_upper']:.0f}")

    # Contract-level metrics
    st.caption(
        f"Per SPX contract (x100): "
        f"Cost ${res['cost']*100:,.0f}, "
        f"Theta ${res['theta']*100:,.0f}/day, "
        f"Max Profit ${res['max_profit']*100:,.0f}, "
        f"Max Loss ${res['max_loss']*100:,.0f}"
    )

    # ---- P&L Chart ----
    st.markdown("### P&L at Short Expiry")

    import plotly.graph_objects as go

    pnl_data = res["pnl_curve"]
    spots = [p["spot"] for p in pnl_data]
    pnls = [p["pnl"] for p in pnl_data]

    fig = go.Figure()

    # Profit zone fill
    fig.add_trace(go.Scatter(
        x=spots, y=[max(0, p) for p in pnls],
        fill="tozeroy", fillcolor="rgba(50,180,80,0.15)",
        line=dict(width=0), showlegend=False,
    ))

    # P&L line
    fig.add_trace(go.Scatter(
        x=spots, y=pnls,
        mode="lines", name="P&L",
        line=dict(color="#1f77b4", width=2),
    ))

    # Reference lines
    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot {spot:,.0f}")
    fig.add_vline(x=res["bwb_k1"], line_dash="dot", line_color="green",
                  annotation_text=f"LC {res['bwb_k1']:,.0f}", annotation_position="bottom left")
    fig.add_vline(x=res["bwb_k2"], line_dash="dot", line_color="red",
                  annotation_text=f"SC {res['bwb_k2']:,.0f}", annotation_position="bottom left")
    fig.add_vline(x=res["put_k"], line_dash="dot", line_color="orange",
                  annotation_text=f"Put {res['put_k']:,.0f}", annotation_position="bottom right")

    fig.add_hline(y=0, line_color="gray", line_width=0.5)

    fig.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Spot at Short Expiry",
        yaxis_title="P&L (per point)",
        margin=dict(l=50, r=20, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Profit zone: {res['profit_zone_low']:,.0f} - {res['profit_zone_high']:,.0f} "
        f"({(res['profit_zone_high']-res['profit_zone_low'])/spot*100:.1f}% width)"
    )

    # ---- Export ----
    # Shared legs definition (with qty)
    all_legs = [
        {"strike": res["bwb_k1"], "option_type": "call", "expiration": res["exp_short"],
         "long": True, "qty": 1, "price": res["c1"]["price"]},
        {"strike": res["bwb_k2"], "option_type": "call", "expiration": res["exp_short"],
         "long": False, "qty": 2, "price": res["c2"]["price"]},
        {"strike": res["bwb_k3"], "option_type": "call", "expiration": res["exp_short"],
         "long": True, "qty": 1, "price": res["c3"]["price"]},
        {"strike": res["put_k"], "option_type": "put", "expiration": res["exp_short"],
         "long": False, "qty": 1, "price": res["sp"]["price"]},
        {"strike": res["put_k"], "option_type": "put", "expiration": res["exp_long"],
         "long": True, "qty": 1, "price": res["lp"]["price"]},
    ]

    # OptionStrat: now supports qty natively
    os_legs = [
        {"strike": res["bwb_k1"], "option_type": "call", "expiration": res["exp_short"],
         "long": True, "qty": 1},
        {"strike": res["bwb_k2"], "option_type": "call", "expiration": res["exp_short"],
         "long": False, "qty": 2},
        {"strike": res["bwb_k3"], "option_type": "call", "expiration": res["exp_short"],
         "long": True, "qty": 1},
        {"strike": res["put_k"], "option_type": "put", "expiration": res["exp_short"],
         "long": False, "qty": 1},
        {"strike": res["put_k"], "option_type": "put", "expiration": res["exp_long"],
         "long": True, "qty": 1},
    ]

    ec1, ec2 = st.columns(2)
    with ec1:
        url = bs.optionstrat_url(res["symbol"], os_legs)
        if url:
            st.caption(f"[OptionStrat]({url})")
    with ec2:
        csv_data = bs.ibkr_basket_csv(res["symbol"], all_legs,
                                       tag="Flyagonal")
        st.download_button(
            "IBKR Basket CSV",
            csv_data,
            f"flyagonal_{res['symbol']}_{res['exp_short']}.csv",
            "text/csv",
        )

    # ---- Management Guidelines ----
    with st.expander("Management Guidelines"):
        tp_10pct = abs(res["max_loss"]) * 0.10
        days_to_tp = tp_10pct / res["theta"] if res["theta"] > 0 else 99
        st.markdown(
            f"- **Profit target**: ~10% of risk = ${tp_10pct*100:,.0f} per contract\n"
            f"- **Estimated days to target**: {days_to_tp:.0f} days (at current theta)\n"
            f"- **Close**: 3-4 days before short expiry to avoid gamma risk\n"
            f"- **Upside adjustment**: Roll short calls higher if spot rallies past {res['bwb_k2']:,.0f}\n"
            f"- **Downside**: Self-correcting via diagonal expansion. Monitor if spot drops below {res['put_k']:,.0f}\n"
            f"- **IV environment**: Best entered when VIX < 20 (currently {res['vix']:.1f})"
        )


main()
