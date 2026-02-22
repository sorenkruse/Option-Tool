"""
Options Pricing Tool - Phase 1
Compare Black-Scholes calculated prices with market prices.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import fetch_all_data
import charts as ch


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:,.{decimals}f}"


def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def fmt_usd(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Custom CSS for smaller fonts
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Reduce base font size */
    .main .block-container {
        font-size: 14px;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        font-size: 13px;
    }
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 13px;
    }
    /* Metric cards - smaller */
    [data-testid="stMetricValue"] {
        font-size: 18px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px;
    }
    /* Subheaders */
    .main h2 {
        font-size: 18px;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .main h3 {
        font-size: 16px;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }
    /* Title */
    .main h1 {
        font-size: 22px;
        margin-bottom: 0.3rem;
    }
    /* DataFrames */
    .stDataFrame {
        font-size: 13px;
    }
    /* Reduce padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_bs(spot, strike, dte_years, r, sigma, q, option_type):
    """Compute BS price and all Greeks. Returns dict or None."""
    if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)) or sigma <= 0:
        return None
    if dte_years <= 0:
        return None

    try:
        result = bs.calculate_all(spot, strike, dte_years, r, sigma, q, option_type)
        result["iv_used"] = sigma
        return result
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_option_section(option_type, market, result_a, result_b, iv_a_label, iv_b_label):
    """Display pricing comparison and Greeks for one option type."""
    st.markdown(f"### {option_type.upper()}")

    # Market prices in compact columns
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Bid", fmt_usd(market["bid"]))
    c2.metric("Ask", fmt_usd(market["ask"]))
    c3.metric("Mid", fmt_usd(market["mid"]))
    c4.metric("Last", fmt_usd(market["last"]))
    c5.metric("Vol", f"{market['volume']:,}")
    c6.metric("OI", f"{market['open_interest']:,}")

    mid = market["mid"]

    # Pricing comparison table
    rows = []

    def dev(result):
        if result is None:
            return np.nan, np.nan
        if np.isnan(mid) or mid <= 0:
            return np.nan, np.nan
        d_abs = result["price"] - mid
        d_pct = d_abs / mid
        return d_abs, d_pct

    dev_a = dev(result_a)
    dev_b = dev(result_b)

    rows.append({
        "": "IV Used",
        iv_a_label: fmt_pct(result_a["iv_used"]) if result_a else "N/A",
        iv_b_label: fmt_pct(result_b["iv_used"]) if result_b else "N/A",
    })
    rows.append({
        "": "BS Price",
        iv_a_label: fmt_usd(result_a["price"]) if result_a else "N/A",
        iv_b_label: fmt_usd(result_b["price"]) if result_b else "N/A",
    })
    rows.append({
        "": "Deviation ($)",
        iv_a_label: fmt(dev_a[0], 2),
        iv_b_label: fmt(dev_b[0], 2),
    })
    rows.append({
        "": "Deviation (%)",
        iv_a_label: fmt_pct(dev_a[1]),
        iv_b_label: fmt_pct(dev_b[1]),
    })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Greeks table
    if result_a or result_b:
        greeks = [
            ("Delta",              "delta",        lambda x: fmt(x, 6)),
            ("Gamma",              "gamma",        lambda x: fmt(x, 8)),
            ("Theta (daily)",      "theta_daily",  lambda x: fmt(x, 4)),
            ("Theta (annual)",     "theta_annual",  lambda x: fmt(x, 2)),
            ("Vega (per 1% IV)",   "vega_pct",     lambda x: fmt(x, 4)),
            ("Rho (per 1% rate)",  "rho_pct",      lambda x: fmt(x, 4)),
            ("Vanna",              "vanna",        lambda x: fmt(x, 6)),
            ("Volga (Vomma)",      "volga",        lambda x: fmt(x, 4)),
            ("Charm",              "charm",        lambda x: fmt(x, 6)),
            ("Speed",              "speed",        lambda x: fmt(x, 10)),
            ("Color",              "color",        lambda x: fmt(x, 8)),
            ("Zomma",              "zomma",        lambda x: fmt(x, 8)),
            ("Intrinsic",          "intrinsic",    lambda x: fmt_usd(x)),
            ("Extrinsic",          "extrinsic",    lambda x: fmt_usd(x)),
            ("Moneyness (S/K)",    "moneyness",    lambda x: fmt(x, 4)),
        ]

        greek_rows = []
        for label, key, formatter in greeks:
            row = {"Greek": label}
            row[iv_a_label] = formatter(result_a[key]) if result_a and key in result_a else "N/A"
            row[iv_b_label] = formatter(result_b[key]) if result_b and key in result_b else "N/A"
            greek_rows.append(row)

        st.dataframe(pd.DataFrame(greek_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Options Pricing Tool")

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("**Parameters**")
        ticker = st.text_input("Ticker", value="SPY",
                               help="e.g. SPY, ^SPX, AAPL, MSFT")
        strike = st.number_input("Strike", value=580.0, min_value=0.01, step=5.0)
        dte = st.number_input("DTE", value=30, min_value=1, max_value=1095, step=1)

        st.divider()
        rate_mode = st.radio("Risk-Free Rate", ["Auto", "Manual"], horizontal=True)
        manual_rate = None
        if rate_mode == "Manual":
            manual_rate = st.number_input("Rate (%)", value=4.5, min_value=0.0,
                                          max_value=20.0, step=0.1) / 100.0

        st.divider()
        run = st.button("Calculate", type="primary", use_container_width=True)

    # --- Main ---
    if not run:
        st.info("Enter parameters and click Calculate.")
        return

    with st.spinner("Fetching data..."):
        try:
            data = fetch_all_data(ticker, strike, dte)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # Risk-free rate
    r = manual_rate if manual_rate is not None else data["risk_free_rate"]
    spot = data["spot"]
    implied_spot = data["implied_spot"]
    q = data["dividend_yield"]
    T = data["dte_years"]

    # --- Market Data Summary ---
    st.markdown("### Market Data")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Spot (Yahoo)", fmt_usd(spot))
    c2.metric("Spot (Implied)", fmt_usd(implied_spot))
    c3.metric("Expiry", f"{data['expiration']} ({data['actual_dte']}d)")
    c4.metric("Rate", fmt_pct(r))
    c5.metric("Div Yield", fmt_pct(q))
    c6.metric("VIX", fmt(data["vix"], 2))
    c7.metric("SKEW", fmt(data["skew_index"], 1))

    # Sanity warning for dividend yield
    if q > 0.05:
        st.error(
            f"Dividend yield appears too high ({fmt_pct(q)}). "
            f"This is likely a data error from Yahoo Finance. "
            f"Check the Debug section for details."
        )

    if data["spy_fallback"]:
        st.warning(
            f"Yahoo does not provide SPX options. Using SPY chain "
            f"(scale factor {data['scale_factor']:.2f}x). "
            f"Prices scaled back to SPX level. IV is unchanged."
        )

    st.divider()

    # --- Compute for Call and Put ---
    for opt_type in ["call", "put"]:
        mkt = data[f"{opt_type}_market"]
        actual_strike = data[f"actual_{opt_type}_strike"]

        if abs(actual_strike - strike) > 1.0:
            st.caption(f"Nearest strike: {fmt_usd(actual_strike)} (requested: {fmt_usd(strike)})")

        mid = mkt["mid"]

        # Method A: IV derived from mid using IMPLIED spot (from put-call parity)
        # This is the most accurate because implied spot is consistent with
        # the option prices, avoiding Yahoo's delayed spot problem.
        if mid > 0 and not np.isnan(mid):
            iv_a = bs.implied_volatility(mid, implied_spot, actual_strike, T, r, q, opt_type)
            # Sanity check: IV should be between 1% and 200%
            if not np.isnan(iv_a) and (iv_a < 0.01 or iv_a > 2.0):
                st.caption(
                    f"Warning: Derived IV ({fmt_pct(iv_a)}) is outside normal range. "
                    f"This may indicate stale Yahoo data. "
                    f"(Spot={fmt_usd(implied_spot)}, K={fmt_usd(actual_strike)}, "
                    f"Mid={fmt_usd(mid)}, r={fmt_pct(r)}, q={fmt_pct(q)})"
                )
        else:
            iv_a = np.nan
        result_a = compute_bs(implied_spot, actual_strike, T, r, iv_a, q, opt_type)

        # Method B: Yahoo chain IV with Yahoo spot (for comparison)
        iv_b = mkt["market_iv"]
        result_b = compute_bs(spot, actual_strike, T, r, iv_b, q, opt_type)

        display_option_section(
            opt_type, mkt, result_a, result_b,
            "Implied Spot + Derived IV", "Yahoo Spot + Yahoo IV"
        )
        st.divider()

    # --- Put-Call Parity ---
    st.markdown("### Put-Call Parity")
    call_mid = data["call_market"]["mid"]
    put_mid = data["put_market"]["mid"]
    if call_mid > 0 and put_mid > 0 and not np.isnan(call_mid) and not np.isnan(put_mid):
        parity = bs.put_call_parity_check(spot, strike, T, r, q, call_mid, put_mid)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("C - P (Market)", fmt_usd(parity["actual_diff"]))
        pc2.metric("Se^(-qT) - Ke^(-rT)", fmt_usd(parity["theoretical_diff"]))
        pc3.metric("Deviation", fmt_usd(parity["deviation"]))
    else:
        st.caption("Insufficient data for parity check.")

    st.divider()

    # --- IV Smile Charts ---
    st.markdown("### IV Smile")
    sc1, sc2 = st.columns(2)

    with sc1:
        if not data["call_smile"].empty and "iv" in data["call_smile"].columns:
            st.plotly_chart(ch.smile_chart(data["call_smile"], "Call IV Smile",
                                           spot=implied_spot),
                            use_container_width=True)
        else:
            st.caption("Call IV: No data")

    with sc2:
        if not data["put_smile"].empty and "iv" in data["put_smile"].columns:
            st.plotly_chart(ch.smile_chart(data["put_smile"], "Put IV Smile",
                                           spot=implied_spot),
                            use_container_width=True)
        else:
            st.caption("Put IV: No data")

    # --- Debug: Parameters Used ---
    with st.expander("Debug: Parameters"):
        st.json({
            "ticker": data["ticker"],
            "chain_ticker": data["chain_ticker"],
            "spy_fallback": data["spy_fallback"],
            "scale_factor": data["scale_factor"],
            "spot_yahoo": spot,
            "spot_implied": implied_spot,
            "spot_diff": round(spot - implied_spot, 4),
            "implied_spot_method": data["implied_spot_method"],
            "requested_strike": strike,
            "actual_call_strike": data["actual_call_strike"],
            "actual_put_strike": data["actual_put_strike"],
            "dte_requested": dte,
            "dte_actual": data["actual_dte"],
            "risk_free_rate": r,
            "dividend_yield": q,
            "call_market_iv": data["call_market_iv"],
            "put_market_iv": data["put_market_iv"],
            "call_smile_iv": data["call_smile_iv"],
            "put_smile_iv": data["put_smile_iv"],
            "atm_iv": data["atm_iv"],
        })


if __name__ == "__main__":
    main()
