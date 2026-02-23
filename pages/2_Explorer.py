"""
Options Explorer - Find options and analyze scenarios.

Step 1: Define base parameters and find an option by strike, delta, price, etc.
Step 2: See how the option changes under different scenarios (IV, spot, DTE shifts).
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from scipy.stats import norm
import charts as ch

# Compact metric styling
CUSTOM_CSS = """
<style>
[data-testid="stMetric"] {
    padding: 8px 6px;
}
[data-testid="stMetricValue"] {
    font-size: 1.1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"

def fmt_usd(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.2f}"

def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.2f}%"

POSITIONS = ["Short Put", "Short Call", "Long Put", "Long Call"]
POS_MAP = {
    "Long Call":  ("call", 1),
    "Short Call": ("call", -1),
    "Long Put":   ("put", 1),
    "Short Put":  ("put", -1),
}

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_for_strike(param, target, S, T, r, sigma, q, opt_type):
    """Find strike that produces the target value. Returns (K, success)."""
    try:
        if param == "strike":
            return float(target), True
        elif param == "price":
            K = bs.solve_strike_for_price(target, S, T, r, sigma, q, opt_type)
        elif param == "delta":
            K = bs.solve_strike_for_delta(target, S, T, r, sigma, q, opt_type)
        elif param == "theta":
            K = bs.solve_strike_for_theta(target, S, T, r, sigma, q, opt_type)
        elif param == "vega":
            K = bs.solve_strike_for_vega(target, S, T, r, sigma, q, opt_type)
        else:
            return np.nan, False
        if np.isnan(K) or K <= 0:
            return np.nan, False
        return K, True
    except Exception:
        return np.nan, False


def probability_of_profit(S, K, T, r, sigma, q, opt_type, sign, premium):
    if T <= 0 or sigma <= 0 or premium <= 0:
        return np.nan
    breakeven = K + premium if opt_type == "call" else K - premium
    if breakeven <= 0:
        return 1.0 if sign == -1 else 0.0
    d2 = (np.log(S / breakeven) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if (opt_type == "call" and sign == 1) or (opt_type == "put" and sign == -1):
        return float(norm.cdf(d2))
    else:
        return float(norm.cdf(-d2))


def compute_pnl_at_expiry(S_range, K, premium, opt_type, sign):
    if opt_type == "call":
        intrinsic = np.maximum(S_range - K, 0)
    else:
        intrinsic = np.maximum(K - S_range, 0)
    return sign * (intrinsic - premium)


# ---------------------------------------------------------------------------
# Yahoo loader
# ---------------------------------------------------------------------------

def load_yahoo_defaults(ticker, dte_target):
    from data_provider import fetch_all_data
    data = fetch_all_data(ticker, 0, dte_target)
    return {
        "spot": round(data["implied_spot"], 2),
        "r": round(data["risk_free_rate"] * 100, 2),
        "q": round(data["dividend_yield"] * 100, 2),
        "iv": round(data["atm_iv"]["avg_iv"] * 100, 1),
        "dte": data["actual_dte"],
    }


# ---------------------------------------------------------------------------
# Compute full result dict
# ---------------------------------------------------------------------------

def compute_option(S, K, T, r, sigma, q, opt_type, sign):
    """Compute all values for display."""
    res = bs.calculate_all(S, K, T, r, sigma, q, opt_type)
    pop = probability_of_profit(S, K, T, r, sigma, q, opt_type, sign, res["price"])
    res["pop"] = pop
    res["iv_used"] = sigma
    res["S"] = S
    res["K"] = K
    res["T"] = T
    res["DTE"] = round(T * 365)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Options Explorer")

    # ================================================================
    # SECTION 1: BASE PARAMETERS
    # ================================================================

    col_pos, col_ticker, col_load = st.columns([2, 2, 1])
    with col_pos:
        position_label = st.selectbox("Position", POSITIONS)
    with col_ticker:
        ticker = st.text_input("Underlying", value="SPY")
    with col_load:
        st.markdown("<br>", unsafe_allow_html=True)
        load_clicked = st.button("Load from Yahoo", use_container_width=True)

    opt_type, sign = POS_MAP[position_label]

    if load_clicked:
        with st.spinner("Loading..."):
            try:
                d = load_yahoo_defaults(ticker, 30)
                st.session_state["ex_spot"] = d["spot"]
                st.session_state["ex_r"] = d["r"]
                st.session_state["ex_q"] = d["q"]
                st.session_state["ex_iv"] = d["iv"]
                st.session_state["ex_dte"] = d["dte"]
                st.rerun()
            except Exception as e:
                st.error(f"Yahoo error: {e}")

    # Market params
    mp1, mp2, mp3, mp4, mp5 = st.columns(5)
    with mp1:
        S = st.number_input("Spot", value=st.session_state.get("ex_spot", 688.0),
                            min_value=1.0, step=1.0, format="%.2f")
    with mp2:
        dte = st.number_input("DTE", value=int(st.session_state.get("ex_dte", 30)),
                              min_value=1, max_value=730, step=1)
    with mp3:
        iv_pct = st.number_input("IV (%)",
                                 value=float(st.session_state.get("ex_iv", 15.0)),
                                 min_value=0.1, max_value=200.0, step=0.5,
                                 format="%.1f")
    with mp4:
        r_pct = st.number_input("Rate (%)",
                                value=float(st.session_state.get("ex_r", 4.0)),
                                min_value=0.0, max_value=20.0, step=0.1)
    with mp5:
        q_pct = st.number_input("Div Yield (%)",
                                value=float(st.session_state.get("ex_q", 1.0)),
                                min_value=0.0, max_value=20.0, step=0.1)

    r = r_pct / 100.0
    q = q_pct / 100.0
    sigma = iv_pct / 100.0
    T = dte / 365.0

    # ================================================================
    # SECTION 2: FIND OPTION
    # ================================================================

    st.markdown("---")
    st.markdown("### Find Option")
    st.caption("Choose how to find your option: by strike, or by a target Greek/price.")

    fc1, fc2, fc3 = st.columns([2, 3, 2])

    with fc1:
        find_by = st.selectbox("Find by", ["Strike", "Delta", "Price",
                                            "Theta (daily)", "Vega (per 1%)"])
    with fc2:
        if find_by == "Strike":
            target_val = st.number_input("Strike", value=float(round(S)),
                                         min_value=1.0, step=1.0, format="%.0f",
                                         key="find_val")
        elif find_by == "Delta":
            default_d = -0.25 if opt_type == "put" else 0.25
            target_val = st.number_input("Target Delta", value=default_d,
                                         min_value=-0.99, max_value=0.99,
                                         step=0.01, format="%.2f", key="find_val")
        elif find_by == "Price":
            target_val = st.number_input("Target Price", value=5.0,
                                         min_value=0.01, step=0.5, format="%.2f",
                                         key="find_val")
        elif find_by == "Theta (daily)":
            target_val = st.number_input("Target Theta/day", value=-0.10,
                                         min_value=-10.0, max_value=-0.001,
                                         step=0.01, format="%.4f", key="find_val")
        else:  # Vega
            target_val = st.number_input("Target Vega/1%", value=0.20,
                                         min_value=0.001, max_value=5.0,
                                         step=0.01, format="%.4f", key="find_val")

    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        find_clicked = st.button("Find", use_container_width=True,
                                 type="primary")

    # Resolve strike
    param_map = {
        "Strike": "strike",
        "Delta": "delta",
        "Price": "price",
        "Theta (daily)": "theta",
        "Vega (per 1%)": "vega",
    }

    if find_clicked:
        K, ok = solve_for_strike(param_map[find_by], target_val,
                                 S, T, r, sigma, q, opt_type)
        if ok:
            st.session_state["ex_found_K"] = round(K, 1)
        else:
            st.error(f"Could not find a strike for {find_by} = {target_val}")

    K_found = st.session_state.get("ex_found_K", round(S))

    # Compute result
    if sigma <= 0 or T <= 0 or K_found <= 0:
        st.warning("Invalid parameters.")
        return

    res = compute_option(S, K_found, T, r, sigma, q, opt_type, sign)

    # ================================================================
    # SECTION 3: OPTION RESULT
    # ================================================================

    st.markdown("---")
    st.markdown("### Result")
    st.markdown(f"**{position_label}  {ticker}  {K_found:.0f}  |  "
                f"{dte} DTE  |  IV {iv_pct:.1f}%**")

    mc = st.columns(8)
    mc[0].metric("Strike", fmt_usd(K_found))
    mc[1].metric("Price", fmt_usd(res["price"]))
    mc[2].metric("Delta", fmt(res["delta"], 4))
    mc[3].metric("Gamma", fmt(res["gamma"], 6))
    mc[4].metric("Theta/d", fmt(res["theta_daily"], 4))
    mc[5].metric("Vega/1%", fmt(res["vega_pct"], 4))
    mc[6].metric("PoP",
                 fmt(res["pop"] * 100, 1) + "%" if not np.isnan(res["pop"]) else "N/A")
    mc[7].metric("Moneyness", fmt(res["moneyness"], 4))

    mc2 = st.columns(6)
    mc2[0].metric("Intrinsic", fmt_usd(res["intrinsic"]))
    mc2[1].metric("Extrinsic", fmt_usd(res["extrinsic"]))
    mc2[2].metric("Rho/1%", fmt(res["rho_pct"], 4))
    mc2[3].metric("Vanna", fmt(res["vanna"], 4))
    mc2[4].metric("Volga", fmt(res["volga"], 4))
    mc2[5].metric("Charm", fmt(res["charm"], 6))

    # OptionStrat link
    from datetime import datetime, timedelta
    exp_date = (datetime.now() + timedelta(days=dte)).strftime("%Y-%m-%d")
    os_url = bs.optionstrat_url(ticker, [{
        "strike": K_found,
        "option_type": opt_type,
        "expiration": exp_date,
        "long": sign > 0,
    }])
    if os_url:
        st.caption(f"[OptionStrat]({os_url})")

    # ================================================================
    # SECTION 4: WHAT-IF SCENARIOS
    # ================================================================

    st.markdown("---")
    st.markdown("### What-If Scenarios")
    st.caption("How does this option change if market conditions shift?")

    # Spot shifts
    st.markdown("#### Spot Changes")
    spot_shifts = [-10, -5, -3, -1, 0, +1, +3, +5, +10]
    spot_rows = []
    for pct in spot_shifts:
        S_new = S * (1 + pct / 100.0)
        try:
            r_new = bs.calculate_all(S_new, K_found, T, r, sigma, q, opt_type)
            pop_new = probability_of_profit(S_new, K_found, T, r, sigma, q,
                                            opt_type, sign, r_new["price"])
            spot_rows.append({
                "Spot Change": f"{pct:+d}%",
                "Spot": fmt_usd(S_new),
                "Price": fmt_usd(r_new["price"]),
                "P&L": fmt_usd(sign * (r_new["price"] - res["price"])),
                "Delta": fmt(r_new["delta"], 4),
                "Gamma": fmt(r_new["gamma"], 6),
                "Theta/d": fmt(r_new["theta_daily"], 4),
                "Vega/1%": fmt(r_new["vega_pct"], 4),
                "PoP": fmt(pop_new * 100, 1) + "%" if not np.isnan(pop_new) else "N/A",
            })
        except Exception:
            pass
    if spot_rows:
        st.dataframe(pd.DataFrame(spot_rows), use_container_width=True,
                     hide_index=True)

    # IV shifts
    st.markdown("#### IV Changes")
    iv_shifts = [-10, -5, -3, -1, 0, +1, +3, +5, +10]
    iv_rows = []
    for pts in iv_shifts:
        sigma_new = sigma + pts / 100.0
        if sigma_new <= 0:
            continue
        try:
            r_new = bs.calculate_all(S, K_found, T, r, sigma_new, q, opt_type)
            pop_new = probability_of_profit(S, K_found, T, r, sigma_new, q,
                                            opt_type, sign, r_new["price"])
            iv_rows.append({
                "IV Change": f"{pts:+d} pts",
                "IV": fmt_pct(sigma_new * 100),
                "Price": fmt_usd(r_new["price"]),
                "P&L": fmt_usd(sign * (r_new["price"] - res["price"])),
                "Delta": fmt(r_new["delta"], 4),
                "Gamma": fmt(r_new["gamma"], 6),
                "Theta/d": fmt(r_new["theta_daily"], 4),
                "Vega/1%": fmt(r_new["vega_pct"], 4),
                "PoP": fmt(pop_new * 100, 1) + "%" if not np.isnan(pop_new) else "N/A",
            })
        except Exception:
            pass
    if iv_rows:
        st.dataframe(pd.DataFrame(iv_rows), use_container_width=True,
                     hide_index=True)

    # DTE shifts
    st.markdown("#### Time Decay")
    dte_points = sorted(set([max(1, dte - d) for d in [0, 1, 3, 5, 7, 14, 21]]
                            + [dte]), reverse=True)
    dte_rows = []
    for d_val in dte_points:
        t_new = d_val / 365.0
        try:
            r_new = bs.calculate_all(S, K_found, t_new, r, sigma, q, opt_type)
            pop_new = probability_of_profit(S, K_found, t_new, r, sigma, q,
                                            opt_type, sign, r_new["price"])
            dte_rows.append({
                "DTE": d_val,
                "Price": fmt_usd(r_new["price"]),
                "P&L": fmt_usd(sign * (r_new["price"] - res["price"])),
                "Delta": fmt(r_new["delta"], 4),
                "Theta/d": fmt(r_new["theta_daily"], 4),
                "Vega/1%": fmt(r_new["vega_pct"], 4),
                "PoP": fmt(pop_new * 100, 1) + "%" if not np.isnan(pop_new) else "N/A",
            })
        except Exception:
            pass
    if dte_rows:
        st.dataframe(pd.DataFrame(dte_rows), use_container_width=True,
                     hide_index=True)

    # ================================================================
    # SECTION 5: CHARTS
    # ================================================================

    st.markdown("---")
    st.markdown("### Charts")

    spot_range = np.linspace(S * 0.8, S * 1.2, 200)

    # P&L at Expiry
    pnl = compute_pnl_at_expiry(spot_range, K_found, res["price"], opt_type, sign)
    st.plotly_chart(ch.pnl_chart(spot_range, pnl, S, K_found),
                    use_container_width=True)

    # Price vs Spot at different IVs
    iv_levels = [sigma * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]]
    iv_labels = [f"IV {v*100:.0f}%{' (current)' if abs(f-1.0)<0.01 else ''}"
                 for v, f in zip(iv_levels, [0.5, 0.75, 1.0, 1.25, 1.5])]

    def compute_curves(calc_fn, sign_mult=1):
        curves = []
        for iv_val, label in zip(iv_levels, iv_labels):
            vals = []
            for s in spot_range:
                try:
                    vals.append(calc_fn(s, K_found, T, r, iv_val, q, opt_type) * sign_mult)
                except Exception:
                    vals.append(np.nan)
            curves.append((label, vals))
        return curves

    st.plotly_chart(
        ch.multi_iv_curve(spot_range,
                          compute_curves(bs.option_price, sign),
                          "Option Price vs Spot", "Price ($)",
                          spot=S, strike=K_found),
        use_container_width=True)

    st.plotly_chart(
        ch.multi_iv_curve(spot_range,
                          compute_curves(lambda s, K, T, r, iv, q, ot:
                                         bs.delta(s, K, T, r, iv, q, ot), sign),
                          "Delta vs Spot", "Delta",
                          spot=S, strike=K_found, zero_line=True),
        use_container_width=True)

    st.plotly_chart(
        ch.multi_iv_curve(spot_range,
                          compute_curves(lambda s, K, T, r, iv, q, ot:
                                         bs.theta(s, K, T, r, iv, q, ot) / 365, sign),
                          "Theta (daily) vs Spot", "Theta/day ($)",
                          spot=S, strike=K_found, zero_line=True),
        use_container_width=True)

    # Time decay
    dte_arr = np.arange(max(dte, 1), 0, -1)
    prices_decay = []
    for d_val in dte_arr:
        try:
            prices_decay.append(
                bs.option_price(S, K_found, d_val / 365.0, r, sigma, q, opt_type))
        except Exception:
            prices_decay.append(np.nan)

    st.plotly_chart(ch.time_decay_chart(dte_arr, prices_decay),
                    use_container_width=True)


main()
