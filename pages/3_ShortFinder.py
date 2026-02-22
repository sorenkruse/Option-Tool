"""
Short Options Finder - Find optimal strikes for short strangles/straddles.

Uses the BS engine for Greeks and the data provider for real market IV.
Finds strikes by target delta, using actual IV from the smile curve
(not a flat VIX estimate).
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    resolve_spot_price, resolve_options_chain, get_dividend_yield,
    get_risk_free_rate, get_vix, get_skew_index, get_atm_iv,
    build_smile_curve, interpolate_smile_iv, compute_implied_spot,
)
from scipy.stats import norm

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
[data-testid="stMetric"] { padding: 8px 4px; }
[data-testid="stMetricValue"] { font-size: 1.1rem; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; }
</style>
"""

# ---------------------------------------------------------------------------
# Strike grid for different underlyings
# ---------------------------------------------------------------------------

def get_strike_step(symbol, spot):
    """Determine the strike step size based on underlying."""
    sym = symbol.upper().replace("^", "")
    if sym in ("SPX", "GSPC", "SPXW"):
        return 25
    elif sym == "SPY":
        return 1
    elif spot > 500:
        return 10
    elif spot > 100:
        return 5
    else:
        return 1


def round_to_step(val, step):
    """Round value to nearest step."""
    return round(val / step) * step


# ---------------------------------------------------------------------------
# Find strike by target delta using smile IV
# ---------------------------------------------------------------------------

def find_strike_by_delta(target_delta, S, T, r, q, smile_df, opt_type, step):
    """
    Find the strike closest to the target delta, using IV from the smile curve.

    For each candidate strike on the grid:
    1. Look up IV from the smile at that strike
    2. Compute delta with BS
    3. Pick the strike with the smallest delta error

    Returns (strike, iv_used, delta_actual).
    """
    # Candidate strikes: range around spot
    if opt_type == "call":
        # Short calls: OTM = above spot
        k_min = round_to_step(S, step)
        k_max = round_to_step(S * 1.4, step)
    else:
        # Short puts: OTM = below spot
        k_min = round_to_step(S * 0.6, step)
        k_max = round_to_step(S, step)

    candidates = np.arange(k_min, k_max + step, step)
    if len(candidates) == 0:
        return S, 0.15, target_delta

    best_k = S
    best_iv = 0.15
    best_delta = 0
    min_diff = float("inf")

    for k in candidates:
        # Get IV from smile for this strike
        iv = interpolate_smile_iv(smile_df, k)
        if np.isnan(iv) or iv <= 0.01:
            continue

        try:
            d = bs.delta(S, k, T, r, iv, q, opt_type)
        except Exception:
            continue

        diff = abs(abs(d) - abs(target_delta))
        if diff < min_diff:
            min_diff = diff
            best_k = k
            best_iv = iv
            best_delta = d

    return int(best_k), best_iv, best_delta


# ---------------------------------------------------------------------------
# OptionStrat link builder
# ---------------------------------------------------------------------------

def build_optionstrat_url(symbol, exp_date_str, call_strike, put_strike):
    """Build OptionStrat analysis URL."""
    sym = symbol.upper().replace("^", "")
    # OptionStrat uses SPXW for SPX weeklies
    leg_sym = "SPXW" if sym in ("SPX", "GSPC") else sym

    try:
        dt = datetime.strptime(exp_date_str, "%Y-%m-%d")
        date_code = dt.strftime("%y%m%d")
    except Exception:
        date_code = exp_date_str.replace("-", "")

    url = (f"https://optionstrat.com/build/custom/{sym}/"
           f"-.{leg_sym}{date_code}C{call_strike},"
           f"-.{leg_sym}{date_code}P{put_strike}")
    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Short Options Finder")

    # ---- Inputs ----
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX").upper()
    with c2:
        dte_input = st.number_input("DTE", value=40, min_value=1, max_value=365)
    with c3:
        target_delta = st.number_input("Target |Delta|", value=0.15,
                                       min_value=0.01, max_value=0.50,
                                       step=0.01, format="%.2f")
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Find Strikes", type="primary", use_container_width=True)

    if not run and "sof_result" not in st.session_state:
        st.info("Set parameters and click 'Find Strikes'.")
        return

    if run:
        with st.spinner("Fetching data..."):
            try:
                result = _compute(symbol, dte_input, target_delta)
                st.session_state["sof_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                return

    res = st.session_state.get("sof_result")
    if res is None:
        return

    _display(res, symbol)


def _compute(symbol, dte_input, target_delta):
    """Fetch data and find optimal strikes."""
    # Spot
    spot, _ = resolve_spot_price(symbol)

    # Options chain
    chain, chain_ticker, spy_fallback = resolve_options_chain(symbol, dte_input)
    scale = 1.0
    working_spot = spot

    if spy_fallback:
        from data_provider import get_spot_price
        spy_spot = get_spot_price("SPY")
        scale = spot / spy_spot
        working_spot = spy_spot

    # Market parameters
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(chain["dte_actual"])
    vix = get_vix()
    skew = get_skew_index()
    T = chain["dte_years"]
    actual_dte = chain["dte_actual"]
    expiration = chain["expiration"]

    # Implied spot
    imp_data = compute_implied_spot(chain, working_spot, r, q, T)
    S = imp_data["implied_spot"] * scale if spy_fallback else imp_data["implied_spot"]

    # ATM IV
    atm_iv = get_atm_iv(chain, working_spot)

    # Smile curves
    call_smile = build_smile_curve(chain["calls"], working_spot)
    put_smile = build_smile_curve(chain["puts"], working_spot)

    # Scale smile strikes if SPY fallback
    if spy_fallback:
        if not call_smile.empty:
            call_smile = call_smile.copy()
            call_smile["strike"] = call_smile["strike"] * scale
        if not put_smile.empty:
            put_smile = put_smile.copy()
            put_smile["strike"] = put_smile["strike"] * scale

    # Find strikes
    step = get_strike_step(symbol, S)

    call_k, call_iv, call_delta = find_strike_by_delta(
        target_delta, S, T, r, q, call_smile, "call", step)
    put_k, put_iv, put_delta = find_strike_by_delta(
        target_delta, S, T, r, q, put_smile, "put", step)

    # Compute full Greeks
    call_res = bs.calculate_all(S, call_k, T, r, call_iv, q, "call")
    put_res = bs.calculate_all(S, put_k, T, r, put_iv, q, "put")

    # PoP for short positions (profit when OTM at expiry)
    call_pop = norm.cdf(-bs._d2(S, call_k, T, r, call_iv, q)) * 100  # P(S < K)
    put_pop = norm.cdf(bs._d2(S, put_k, T, r, put_iv, q)) * 100     # P(S > K)

    return {
        "S": S, "spot_raw": spot, "vix": vix, "skew": skew,
        "r": r, "q": q, "T": T, "actual_dte": actual_dte,
        "expiration": expiration,
        "atm_iv": atm_iv,
        "call_k": call_k, "call_iv": call_iv, "call_delta": call_delta,
        "call_res": call_res, "call_pop": call_pop,
        "put_k": put_k, "put_iv": put_iv, "put_delta": put_delta,
        "put_res": put_res, "put_pop": put_pop,
        "step": step, "scale": scale, "symbol": symbol,
        "target_delta": target_delta,
    }


def _display(res, symbol):
    """Display results."""
    S = res["S"]
    call_k, put_k = res["call_k"], res["put_k"]
    call_res, put_res = res["call_res"], res["put_res"]

    dist_call = ((call_k / S) - 1) * 100
    dist_put = (1 - (put_k / S)) * 100
    combined_theta = call_res["theta_daily"] + put_res["theta_daily"]
    combined_premium = call_res["price"] + put_res["price"]
    avg_pop = (res["call_pop"] + res["put_pop"]) / 2

    # ---- Header ----
    st.markdown(f"#### {symbol} @ {S:,.2f}  |  VIX: {res['vix']:.1f}  |  "
                f"Exp: {res['expiration']}  ({res['actual_dte']} DTE)")

    # ---- Key metrics ----
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Call Strike", f"{call_k:,}", f"+{dist_call:.1f}%")
    m2.metric("Put Strike", f"{put_k:,}", f"-{dist_put:.1f}%")
    m3.metric("Premium", f"${combined_premium:,.2f}")
    m4.metric("Theta/day", f"${combined_theta:,.2f}")
    m5.metric("Avg PoP", f"{avg_pop:.1f}%")
    m6.metric("SKEW", f"{res['skew']:.1f}" if not np.isnan(res["skew"]) else "N/A")

    # ---- Distribution chart ----
    atm_sigma = res["atm_iv"]["avg_iv"]
    if np.isnan(atm_sigma) or atm_sigma <= 0:
        atm_sigma = res["vix"] / 100

    std_dev = S * atm_sigma * np.sqrt(res["T"])
    x = np.linspace(S - 3.5 * std_dev, S + 3.5 * std_dev, 600)
    y = norm.pdf(x, S, std_dev)

    fig = go.Figure()

    # Profit zone (between strikes)
    mask = (x >= put_k) & (x <= call_k)
    fig.add_trace(go.Scatter(
        x=x[mask], y=y[mask], fill="tozeroy",
        fillcolor="rgba(0,180,0,0.12)", line=dict(width=0),
        name="Profit Zone", showlegend=False
    ))

    # Distribution curve
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="rgba(150,150,150,0.7)", width=1.5),
        name="Distribution", showlegend=False
    ))

    # Strike lines
    fig.add_vline(x=S, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot {S:,.0f}", annotation_position="top")
    fig.add_vline(x=call_k, line_color="red",
                  annotation_text=f"Call {call_k:,}", annotation_position="top left")
    fig.add_vline(x=put_k, line_color="green",
                  annotation_text=f"Put {put_k:,}", annotation_position="top right")

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white",
        xaxis_title="Price at Expiry",
        yaxis_showticklabels=False,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Greeks table ----
    st.markdown("#### Greeks")
    greeks_df = pd.DataFrame({
        "": ["Short Call", "Short Put", "Combined"],
        "Strike": [f"{call_k:,}", f"{put_k:,}", ""],
        "Distance": [f"+{dist_call:.1f}%", f"-{dist_put:.1f}%", ""],
        "IV": [f"{res['call_iv']*100:.1f}%", f"{res['put_iv']*100:.1f}%", ""],
        "Price": [f"${call_res['price']:.2f}", f"${put_res['price']:.2f}",
                  f"${combined_premium:.2f}"],
        "Delta": [f"{call_res['delta']:.4f}", f"{put_res['delta']:.4f}",
                  f"{call_res['delta']+put_res['delta']:.4f}"],
        "Gamma": [f"{call_res['gamma']:.6f}", f"{put_res['gamma']:.6f}",
                  f"{call_res['gamma']+put_res['gamma']:.6f}"],
        "Theta/d": [f"{call_res['theta_daily']:.4f}",
                    f"{put_res['theta_daily']:.4f}",
                    f"{combined_theta:.4f}"],
        "Vega/1%": [f"{call_res['vega_pct']:.4f}",
                    f"{put_res['vega_pct']:.4f}",
                    f"{call_res['vega_pct']+put_res['vega_pct']:.4f}"],
        "PoP": [f"{res['call_pop']:.1f}%", f"{res['put_pop']:.1f}%",
                f"{avg_pop:.1f}%"],
    })
    st.dataframe(greeks_df, use_container_width=True, hide_index=True)

    # ---- What-if: VIX shock ----
    st.markdown("#### IV Shock Scenarios")
    shock_rows = []
    for shock_pct in [-20, -10, -5, 0, +5, +10, +20, +30, +50]:
        c_iv_new = res["call_iv"] * (1 + shock_pct / 100)
        p_iv_new = res["put_iv"] * (1 + shock_pct / 100)
        if c_iv_new <= 0 or p_iv_new <= 0:
            continue
        try:
            c_new = bs.calculate_all(S, call_k, res["T"], res["r"],
                                     c_iv_new, res["q"], "call")
            p_new = bs.calculate_all(S, put_k, res["T"], res["r"],
                                     p_iv_new, res["q"], "put")
            new_premium = c_new["price"] + p_new["price"]
            # Short position: profit when premium decreases
            pnl = -(new_premium - combined_premium)
            shock_rows.append({
                "IV Shock": f"{shock_pct:+d}%",
                "Call IV": f"{c_iv_new*100:.1f}%",
                "Put IV": f"{p_iv_new*100:.1f}%",
                "Premium": f"${new_premium:.2f}",
                "P&L": f"${pnl:+.2f}",
                "Theta/d": f"${c_new['theta_daily']+p_new['theta_daily']:.2f}",
            })
        except Exception:
            pass
    if shock_rows:
        st.dataframe(pd.DataFrame(shock_rows), use_container_width=True,
                     hide_index=True)

    # ---- OptionStrat link ----
    url = build_optionstrat_url(symbol, res["expiration"], call_k, put_k)
    st.caption(f"[Open in OptionStrat]({url})")

    # ---- Log line ----
    with st.expander("Log"):
        log = ";".join([
            datetime.now().strftime("%d.%m.%Y"),
            symbol, f"{S:.2f}",
            str(res["actual_dte"]),
            str(call_k), str(put_k),
            f"{res['target_delta']}",
            f"{res['vix']:.1f}",
            url
        ])
        st.code(log)


main()
