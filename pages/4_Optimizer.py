"""
Short Option Optimizer with VRP Model + Monte Carlo Simulation.

1. VRP Model: Compares implied vol (market price) vs realized vol (actual moves).
   The difference is the volatility risk premium - the edge of selling options.
   Options priced at RV instead of IV gives the expected profit.

2. Monte Carlo: Simulates spot paths using realized vol, marks to market daily,
   exits at 50% take profit or 200% stop loss. Computes expected holding days,
   win rate, and annualized return.

3. Combines both to find the DTE where the VRP edge is harvested most efficiently.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    fetch_chains_for_dte_range, get_dividend_yield, get_risk_free_rate,
    get_vix, get_skew_index, interpolate_smile_iv, resolve_spot_price,
    get_realized_volatility,
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
# Strike helpers
# ---------------------------------------------------------------------------

def get_strike_step(symbol, spot):
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
    return round(val / step) * step


def find_strike_for_delta(target_delta, S, T, r, q, smile_df, opt_type, step):
    if opt_type == "call":
        k_min, k_max = round_to_step(S, step), round_to_step(S * 1.4, step)
    else:
        k_min, k_max = round_to_step(S * 0.6, step), round_to_step(S, step)

    candidates = np.arange(k_min, k_max + step, step)
    best_k, best_iv, best_delta, min_diff = None, None, None, float("inf")

    for k in candidates:
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
            best_k, best_iv, best_delta = int(k), iv, d

    if best_k is None:
        return None
    return {"strike": best_k, "iv": best_iv, "delta": best_delta}


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

def simulate_early_exit(S, K, T_years, r, iv, rv, q, opt_type, dte,
                        n_paths=5000, tp_pct=0.50, sl_pct=2.00,
                        rng=None):
    """
    Monte Carlo simulation of short option with early exit rules.

    Simulates spot paths using REALIZED vol (what actually happens),
    then marks option to market using IMPLIED vol (how market prices it).
    This captures the VRP: the option decays at IV speed while spot
    moves at RV speed.

    Args:
        S: current spot
        K: strike
        T_years: time to expiry in years
        r: risk-free rate
        iv: implied volatility (for option pricing / mark to market)
        rv: realized volatility (for simulating spot movement)
        q: dividend yield
        opt_type: 'call' or 'put'
        dte: days to expiry
        n_paths: number of simulation paths
        tp_pct: take profit at this fraction of premium (0.50 = 50%)
        sl_pct: stop loss at this multiple of premium (2.0 = 200%)

    Returns dict with simulation results.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Entry price (what we sell for)
    entry_price = bs.option_price(S, K, T_years, r, iv, q, opt_type)
    if entry_price <= 0:
        return None

    tp_target = entry_price * tp_pct          # close when option worth this
    sl_target = entry_price * (1 + sl_pct)    # close when option worth this

    # For short: profit = entry - current. TP when current <= entry*(1-tp_pct).
    # SL when current >= entry*(1+sl_pct).
    tp_price = entry_price * (1 - tp_pct)     # option price at 50% profit
    sl_price = entry_price * (1 + sl_pct)     # option price at 200% loss

    dt = 1.0 / 365.0
    n_days = dte

    # Generate all paths at once: (n_paths, n_days) random draws
    z = rng.standard_normal((n_paths, n_days))

    # Simulate spot paths using REALIZED vol
    drift = (r - q - 0.5 * rv**2) * dt
    diffusion = rv * np.sqrt(dt)

    # Spot paths: (n_paths, n_days+1), starting at S
    spots = np.zeros((n_paths, n_days + 1))
    spots[:, 0] = S
    for t in range(n_days):
        spots[:, t + 1] = spots[:, t] * np.exp(drift + diffusion * z[:, t])

    # Track results per path
    pnl = np.zeros(n_paths)
    hold_days = np.zeros(n_paths, dtype=int)
    exit_type = np.empty(n_paths, dtype="U10")  # "TP", "SL", "EXP"

    for i in range(n_paths):
        exited = False
        for t in range(1, n_days + 1):
            remaining_T = (n_days - t) / 365.0

            if remaining_T <= 0.5 / 365.0:
                # Last day: use intrinsic
                if opt_type == "call":
                    cur_price = max(spots[i, t] - K, 0)
                else:
                    cur_price = max(K - spots[i, t], 0)
            else:
                # Mark to market using IV (market still prices at IV)
                # IV could shift, but we assume constant IV for simplicity
                try:
                    cur_price = bs.option_price(
                        spots[i, t], K, remaining_T, r, iv, q, opt_type)
                except Exception:
                    cur_price = entry_price  # fallback

            # Check exit conditions (short position)
            if cur_price <= tp_price:
                # Take profit: buy back cheaper
                pnl[i] = entry_price - cur_price
                hold_days[i] = t
                exit_type[i] = "TP"
                exited = True
                break
            elif cur_price >= sl_price:
                # Stop loss: buy back more expensive
                pnl[i] = entry_price - cur_price
                hold_days[i] = t
                exit_type[i] = "SL"
                exited = True
                break

        if not exited:
            # Hold to expiry
            if opt_type == "call":
                final_val = max(spots[i, -1] - K, 0)
            else:
                final_val = max(K - spots[i, -1], 0)
            pnl[i] = entry_price - final_val
            hold_days[i] = n_days
            exit_type[i] = "EXP"

    # Aggregate results
    tp_count = np.sum(exit_type == "TP")
    sl_count = np.sum(exit_type == "SL")
    exp_count = np.sum(exit_type == "EXP")

    win_count = np.sum(pnl > 0)

    avg_pnl = float(np.mean(pnl))
    avg_hold = float(np.mean(hold_days))
    median_hold = float(np.median(hold_days))

    # Annualized return (based on average holding period)
    capital = K if opt_type == "put" else S
    if avg_hold > 0 and capital > 0:
        ann_return = (avg_pnl / capital) * (365 / avg_hold) * 100
    else:
        ann_return = 0

    # Pnl per day (using average holding period)
    pnl_per_day = avg_pnl / avg_hold if avg_hold > 0 else 0

    return {
        "entry_price": entry_price,
        "avg_pnl": avg_pnl,
        "median_pnl": float(np.median(pnl)),
        "pnl_std": float(np.std(pnl)),
        "win_rate": win_count / n_paths * 100,
        "tp_rate": tp_count / n_paths * 100,
        "sl_rate": sl_count / n_paths * 100,
        "exp_rate": exp_count / n_paths * 100,
        "avg_hold_days": avg_hold,
        "median_hold_days": median_hold,
        "ann_return": ann_return,
        "pnl_per_day": pnl_per_day,
        "pnl_distribution": pnl,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Short Option Optimizer")

    # ---- Inputs ----
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX").upper()
    with c2:
        dte_min = st.number_input("DTE min", value=20, min_value=5, max_value=180)
    with c3:
        dte_max = st.number_input("DTE max", value=50, min_value=10, max_value=365)
    with c4:
        target_delta = st.number_input("|Delta|", value=0.15,
                                       min_value=0.03, max_value=0.45,
                                       step=0.01, format="%.2f")
    with c5:
        opt_type_sel = st.selectbox("Type", ["Put", "Call"])

    c6, c7, c8, c9 = st.columns([1, 1, 1, 1])
    with c6:
        tp_pct = st.number_input("Take Profit %", value=50, min_value=10,
                                 max_value=90, step=5)
    with c7:
        sl_pct = st.number_input("Stop Loss %", value=200, min_value=50,
                                 max_value=500, step=50)
    with c8:
        n_paths = st.number_input("MC Paths", value=5000, min_value=1000,
                                  max_value=20000, step=1000)
    with c9:
        rv_override = st.number_input(
            "RV Override % (0 = use actual)",
            value=0.0, min_value=0.0, max_value=60.0, step=0.5, format="%.1f",
            help="Override the realized vol used for simulation. "
                 "0 = use 20d historical RV from Yahoo.")

    run_col1, run_col2 = st.columns([3, 2])
    with run_col1:
        run = st.button("Run Optimization", type="primary",
                        use_container_width=True)
    with run_col2:
        run_multi = st.button("Multi-RV Scenario",
                              use_container_width=True,
                              help="Run at multiple RV levels to see "
                                   "how the optimum shifts")

    opt_type = opt_type_sel.lower()
    rv_ov = rv_override / 100.0 if rv_override > 0 else None

    if not run and not run_multi and "opt3_result" not in st.session_state:
        st.info("Configure parameters and click 'Run Optimization'.")
        return

    if run:
        with st.spinner("Loading data and running Monte Carlo simulation..."):
            try:
                result = _run_scan(symbol, dte_min, dte_max, target_delta,
                                   opt_type, tp_pct / 100, sl_pct / 100,
                                   int(n_paths), rv_ov)
                st.session_state["opt3_result"] = result
                st.session_state.pop("opt3_multi", None)
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if run_multi:
        with st.spinner("Running volatility scenarios (this takes a while)..."):
            try:
                multi, base_iv = _run_multi_rv(symbol, dte_min, dte_max,
                                               target_delta, opt_type,
                                               tp_pct / 100, sl_pct / 100,
                                               int(n_paths))
                st.session_state["opt3_multi"] = (multi, base_iv)
                if "opt3_result" not in st.session_state and multi:
                    st.session_state["opt3_result"] = multi[0]["result"]
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    res = st.session_state.get("opt3_result")
    if res is None:
        return

    _display(res)

    # Multi-RV scenarios
    multi_data = st.session_state.get("opt3_multi")
    if multi_data:
        multi, base_iv = multi_data
        _display_multi_rv(multi, base_iv)


def _run_scan(symbol, dte_min, dte_max, target_delta, opt_type,
              tp_pct, sl_pct, n_paths, rv_override=None):
    """Full scan with VRP + Monte Carlo."""

    chains, spot, working_spot, scale = fetch_chains_for_dte_range(
        symbol, dte_min, dte_max)
    if not chains:
        raise ValueError(f"No expirations found between {dte_min}-{dte_max} DTE")

    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(35)
    vix = get_vix()
    skew = get_skew_index()
    step = get_strike_step(symbol, spot)
    S = spot

    # Realized volatility
    rv_data = get_realized_volatility(symbol, windows=[20, 30])
    rv_20 = rv_data.get("rv_20", np.nan)
    rv_30 = rv_data.get("rv_30", np.nan)

    if rv_override is not None and rv_override > 0:
        rv_sim = rv_override
    else:
        rv_sim = rv_20 if not np.isnan(rv_20) else rv_30

    if np.isnan(rv_sim):
        raise ValueError("Could not compute realized volatility")

    rng = np.random.default_rng(42)
    rows = []

    progress = st.progress(0, text="Scanning expirations...")
    total = len(chains)

    for idx, chain_data in enumerate(chains):
        dte = chain_data["dte"]
        T = chain_data["dte_years"]
        exp = chain_data["expiration"]
        smile = chain_data["put_smile"] if opt_type == "put" \
            else chain_data["call_smile"]

        progress.progress((idx + 1) / total,
                          text=f"Simulating {exp} ({dte} DTE)...")

        if smile.empty:
            continue

        result = find_strike_for_delta(target_delta, S, T, r, q,
                                       smile, opt_type, step)
        if result is None:
            continue

        K = result["strike"]
        iv = result["iv"]
        delta = result["delta"]

        # BS metrics
        try:
            bs_res = bs.calculate_all(S, K, T, r, iv, q, opt_type)
        except Exception:
            continue

        price_at_iv = bs_res["price"]
        if price_at_iv <= 0:
            continue

        # VRP: price at IV vs price at RV
        try:
            price_at_rv = bs.option_price(S, K, T, r, rv_sim, q, opt_type)
        except Exception:
            price_at_rv = price_at_iv

        vrp_edge = price_at_iv - price_at_rv  # expected profit from VRP
        vrp_edge_daily = vrp_edge / dte

        # Monte Carlo simulation
        mc = simulate_early_exit(S, K, T, r, iv, rv_sim, q, opt_type, dte,
                                 n_paths=n_paths, tp_pct=tp_pct, sl_pct=sl_pct,
                                 rng=rng)
        if mc is None:
            continue

        # Distance
        distance_pct = abs(K - S) / S * 100

        rows.append({
            "expiration": exp,
            "dte": dte,
            "strike": K,
            "iv": iv,
            "delta": delta,
            "price": price_at_iv,
            "distance_pct": distance_pct,
            "theta_daily": bs_res["theta_daily"],
            "vega_pct": bs_res["vega_pct"],
            "gamma": bs_res["gamma"],
            # VRP
            "price_at_rv": price_at_rv,
            "vrp_edge": vrp_edge,
            "vrp_edge_daily": vrp_edge_daily,
            # MC results
            "mc_avg_pnl": mc["avg_pnl"],
            "mc_pnl_per_day": mc["pnl_per_day"],
            "mc_win_rate": mc["win_rate"],
            "mc_tp_rate": mc["tp_rate"],
            "mc_sl_rate": mc["sl_rate"],
            "mc_exp_rate": mc["exp_rate"],
            "mc_avg_hold": mc["avg_hold_days"],
            "mc_median_hold": mc["median_hold_days"],
            "mc_ann_return": mc["ann_return"],
            "mc_pnl_std": mc["pnl_std"],
            "mc_pnl_dist": mc["pnl_distribution"],
        })

    progress.empty()

    if not rows:
        raise ValueError("No valid options found.")

    df = pd.DataFrame(rows).sort_values("dte")

    return {
        "df": df, "S": S, "vix": vix, "skew": skew, "r": r, "q": q,
        "rv_20": rv_20, "rv_30": rv_30, "rv_sim": rv_sim,
        "symbol": symbol, "target_delta": target_delta, "opt_type": opt_type,
        "tp_pct": tp_pct, "sl_pct": sl_pct, "n_paths": n_paths,
    }


def _display(res):
    """Display results."""
    df = res["df"]
    S = res["S"]
    opt_label = "Short Put" if res["opt_type"] == "put" else "Short Call"

    # ---- Header ----
    st.markdown(f"#### {res['symbol']} @ {S:,.2f}  |  VIX: {res['vix']:.1f}  |  "
                f"{opt_label}  |  Target |Delta|: {res['target_delta']:.2f}")

    # VRP info
    vrp_col1, vrp_col2, vrp_col3, vrp_col4 = st.columns(4)
    vrp_col1.metric("IV (ATM)", f"{res['vix']:.1f}%")
    vrp_col2.metric("RV 20d", f"{res['rv_20']*100:.1f}%"
                     if not np.isnan(res['rv_20']) else "N/A")
    vrp_col3.metric("RV 30d", f"{res['rv_30']*100:.1f}%"
                     if not np.isnan(res['rv_30']) else "N/A")
    vrp = res['vix'] - res['rv_sim'] * 100
    vrp_col4.metric("VRP (IV - RV)", f"{vrp:.1f} pts")

    st.caption(f"Simulation: {res['n_paths']:,} paths  |  "
               f"TP: {res['tp_pct']*100:.0f}%  |  "
               f"SL: {res['sl_pct']*100:.0f}%  |  "
               f"Spot moves simulated at RV ({res['rv_sim']*100:.1f}%), "
               f"options priced at IV")

    # ---- Main comparison chart ----
    st.markdown("---")
    st.markdown("### DTE Comparison")

    dte_vals = df["dte"].values

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "MC P&L / Day ($)",
            "MC Win Rate (%)",
            "MC Avg Hold (days)",
            "VRP Edge / Day ($)",
            "MC Ann. Return (%)",
            "MC TP / SL / Expiry (%)",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    # MC P&L per day
    colors_pnl = ["rgba(50,180,80,0.7)" if v >= 0 else "rgba(255,80,80,0.7)"
                   for v in df["mc_pnl_per_day"]]
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_pnl_per_day"].values,
                         marker_color=colors_pnl, showlegend=False),
                  row=1, col=1)

    # Win rate
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_win_rate"].values,
                         marker_color="rgba(50,180,80,0.7)", showlegend=False),
                  row=1, col=2)

    # Avg holding days
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_avg_hold"].values,
                         marker_color="rgba(100,150,255,0.7)", showlegend=False),
                  row=1, col=3)

    # VRP edge per day
    fig.add_trace(go.Bar(x=dte_vals, y=df["vrp_edge_daily"].values,
                         marker_color="rgba(255,150,50,0.7)", showlegend=False),
                  row=2, col=1)

    # Annualized return
    colors_ann = ["rgba(50,180,80,0.7)" if v >= 0 else "rgba(255,80,80,0.7)"
                  for v in df["mc_ann_return"]]
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_ann_return"].values,
                         marker_color=colors_ann, showlegend=False),
                  row=2, col=2)

    # TP / SL / Expiry stacked bars
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_tp_rate"].values,
                         name="TP", marker_color="rgba(50,180,80,0.7)"),
                  row=2, col=3)
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_sl_rate"].values,
                         name="SL", marker_color="rgba(255,80,80,0.7)"),
                  row=2, col=3)
    fig.add_trace(go.Bar(x=dte_vals, y=df["mc_exp_rate"].values,
                         name="Expiry", marker_color="rgba(150,150,150,0.5)"),
                  row=2, col=3)

    fig.update_layout(
        template="plotly_white", height=550, barmode="stack",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Table ----
    st.markdown("---")
    st.markdown("### All Expirations")

    SORT_OPTIONS = {
        "mc_pnl_per_day": "MC P&L / Day (recommended)",
        "mc_ann_return": "MC Annualized Return (%)",
        "mc_win_rate": "MC Win Rate (%)",
        "vrp_edge_daily": "VRP Edge / Day ($)",
        "mc_avg_hold": "Avg Holding Days (shortest)",
        "price": "Option Price ($)",
    }

    sort_key = st.selectbox("Sort by", list(SORT_OPTIONS.keys()),
                            format_func=lambda x: SORT_OPTIONS[x])

    ascending = sort_key == "mc_avg_hold"
    sorted_df = df.sort_values(sort_key, ascending=ascending)

    display_df = pd.DataFrame({
        "Exp": sorted_df["expiration"],
        "DTE": sorted_df["dte"],
        "Strike": sorted_df["strike"].apply(lambda x: f"{x:,}"),
        "Dist": sorted_df["distance_pct"].apply(lambda x: f"{x:.1f}%"),
        "IV": sorted_df["iv"].apply(lambda x: f"{x*100:.1f}%"),
        "Delta": sorted_df["delta"].apply(lambda x: f"{x:.3f}"),
        "Price": sorted_df["price"].apply(lambda x: f"${x:.2f}"),
        "VRP $": sorted_df["vrp_edge"].apply(lambda x: f"${x:.2f}"),
        "VRP/d": sorted_df["vrp_edge_daily"].apply(lambda x: f"${x:.2f}"),
        "MC P&L/d": sorted_df["mc_pnl_per_day"].apply(lambda x: f"${x:.2f}"),
        "Win%": sorted_df["mc_win_rate"].apply(lambda x: f"{x:.1f}%"),
        "TP%": sorted_df["mc_tp_rate"].apply(lambda x: f"{x:.1f}%"),
        "SL%": sorted_df["mc_sl_rate"].apply(lambda x: f"{x:.1f}%"),
        "Avg Hold": sorted_df["mc_avg_hold"].apply(lambda x: f"{x:.1f}d"),
        "Ann%": sorted_df["mc_ann_return"].apply(lambda x: f"{x:.1f}%"),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---- Best pick ----
    best = df.loc[df["mc_pnl_per_day"].idxmax()]
    st.markdown("---")
    st.markdown("### Recommendation (highest MC P&L/Day)")

    mc = st.columns(8)
    mc[0].metric("Expiration", best["expiration"])
    mc[1].metric("DTE", f"{int(best['dte'])}")
    mc[2].metric("Strike", f"{int(best['strike']):,}")
    mc[3].metric("Price", f"${best['price']:.2f}")
    mc[4].metric("MC P&L/Day", f"${best['mc_pnl_per_day']:.2f}")
    mc[5].metric("Win Rate", f"{best['mc_win_rate']:.1f}%")
    mc[6].metric("Avg Hold", f"{best['mc_avg_hold']:.1f}d")
    mc[7].metric("Ann. Return", f"{best['mc_ann_return']:.1f}%")

    mc2 = st.columns(6)
    mc2[0].metric("VRP Edge", f"${best['vrp_edge']:.2f}")
    mc2[1].metric("TP Rate", f"{best['mc_tp_rate']:.1f}%")
    mc2[2].metric("SL Rate", f"{best['mc_sl_rate']:.1f}%")
    mc2[3].metric("Delta", f"{best['delta']:.3f}")
    mc2[4].metric("IV", f"{best['iv']*100:.1f}%")
    mc2[5].metric("Dist", f"{best['distance_pct']:.1f}%")

    # OptionStrat link
    os_url = bs.optionstrat_url(res["symbol"], [{
        "strike": int(best["strike"]),
        "option_type": res["opt_type"],
        "expiration": best["expiration"],
        "long": False,
    }])
    if os_url:
        st.caption(f"[OptionStrat]({os_url})")

    # ---- P&L distribution of best ----
    if "mc_pnl_dist" in best and best["mc_pnl_dist"] is not None:
        st.markdown("---")
        st.markdown(f"### P&L Distribution ({best['expiration']}, "
                    f"{int(best['dte'])} DTE)")

        pnl_data = best["mc_pnl_dist"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pnl_data, nbinsx=80,
            marker_color="rgba(100,150,255,0.6)",
            marker_line_color="rgba(100,150,255,0.9)",
            marker_line_width=0.5,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_hist.add_vline(x=np.mean(pnl_data), line_color="green",
                           annotation_text=f"Mean: ${np.mean(pnl_data):.2f}")
        fig_hist.update_layout(
            template="plotly_white", height=300,
            xaxis_title="P&L ($)", yaxis_title="Count",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Multi-RV Scenario
# ---------------------------------------------------------------------------

def _run_multi_rv(symbol, dte_min, dte_max, target_delta, opt_type,
                  tp_pct, sl_pct, n_paths):
    """Run the scan at multiple vol levels relative to current IV."""
    # Vol shift scenarios: percentage changes from current IV
    vol_shifts = [-0.50, -0.30, -0.10, 0, +0.10, +0.20, +0.30, +0.40, +0.50]

    # First get the base data to know current IV
    base = _run_scan(symbol, dte_min, dte_max, target_delta, opt_type,
                     tp_pct, sl_pct, 1000, rv_override=None)
    # Use median IV across expirations as reference
    base_iv = float(base["df"]["iv"].median())

    results = []
    progress = st.progress(0, text="Running volatility scenarios...")
    paths_per = max(n_paths * 3 // (5 * len(vol_shifts)), 1000)  # ~3000

    for i, shift in enumerate(vol_shifts):
        rv_level = base_iv * (1 + shift)
        if rv_level <= 0.02:
            continue

        label = f"{shift*100:+.0f}%"
        progress.progress((i + 1) / len(vol_shifts),
                          text=f"Vol {label} (RV={rv_level*100:.1f}%)...")
        try:
            res = _run_scan(symbol, dte_min, dte_max, target_delta, opt_type,
                            tp_pct, sl_pct, paths_per,
                            rv_override=rv_level)
            results.append({
                "shift": shift,
                "shift_label": label,
                "rv": rv_level,
                "result": res,
            })
        except Exception:
            continue

    progress.empty()
    return results, base_iv


def _display_multi_rv(multi, base_iv):
    """Display multi-RV scenario comparison with relative shift labels."""
    st.markdown("---")
    st.markdown("### Volatility Scenario Analysis")
    st.caption(f"Base IV: {base_iv*100:.1f}% — Scenarios show what happens "
               f"when realized vol shifts relative to current IV.")

    summary_rows = []
    for entry in multi:
        df = entry["result"]["df"]
        if df.empty:
            continue

        best = df.loc[df["mc_pnl_per_day"].idxmax()]

        summary_rows.append({
            "shift": entry["shift"],
            "label": entry["shift_label"],
            "RV": entry["rv"],
            "Best DTE": int(best["dte"]),
            "Strike": int(best["strike"]),
            "P&L/Day": best["mc_pnl_per_day"],
            "Win Rate": best["mc_win_rate"],
            "SL Rate": best["mc_sl_rate"],
            "Avg Hold": best["mc_avg_hold"],
            "Ann Return": best["mc_ann_return"],
        })

    if not summary_rows:
        st.warning("No scenario results.")
        return

    summary = pd.DataFrame(summary_rows)
    x_labels = [f"{r['label']}\n({r['RV']*100:.0f}%)" for r in summary_rows]

    # Chart: Best DTE, Win Rate, P&L/Day
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Best DTE", "Win Rate (%)", "P&L / Day ($)"),
    )

    fig.add_trace(go.Bar(
        x=x_labels, y=summary["Best DTE"].values,
        marker_color="rgba(100,150,255,0.7)",
        text=summary["Best DTE"].values, textposition="auto",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_labels, y=summary["Win Rate"].values,
        marker_color="rgba(50,180,80,0.7)", showlegend=False,
    ), row=1, col=2)

    colors_pnl = ["rgba(50,180,80,0.7)" if v >= 0 else "rgba(255,80,80,0.7)"
                   for v in summary["P&L/Day"].values]
    fig.add_trace(go.Bar(
        x=x_labels, y=summary["P&L/Day"].values,
        marker_color=colors_pnl, showlegend=False,
    ), row=1, col=3)

    fig.update_layout(template="plotly_white", height=350,
                      margin=dict(l=50, r=20, t=40, b=60))
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display = pd.DataFrame({
        "Vol Shift": [r["label"] for r in summary_rows],
        "RV": summary["RV"].apply(lambda x: f"{x*100:.1f}%"),
        "Best DTE": summary["Best DTE"],
        "Strike": summary["Strike"].apply(lambda x: f"{x:,}"),
        "P&L/Day": summary["P&L/Day"].apply(lambda x: f"${x:.2f}"),
        "Win%": summary["Win Rate"].apply(lambda x: f"{x:.1f}%"),
        "SL%": summary["SL Rate"].apply(lambda x: f"{x:.1f}%"),
        "Avg Hold": summary["Avg Hold"].apply(lambda x: f"{x:.1f}d"),
        "Ann%": summary["Ann Return"].apply(lambda x: f"{x:.1f}%"),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Line chart: P&L/Day by DTE across all scenarios
    st.markdown("#### P&L/Day by DTE across Volatility Scenarios")
    fig2 = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

    for i, entry in enumerate(multi):
        df = entry["result"]["df"].sort_values("dte")
        fig2.add_trace(go.Scatter(
            x=df["dte"].values,
            y=df["mc_pnl_per_day"].values,
            mode="lines+markers",
            name=f"{entry['shift_label']} ({entry['rv']*100:.0f}%)",
            line=dict(color=colors[i % len(colors)],
                      width=3 if entry["shift"] == 0 else 1.5),
        ))

    fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
    fig2.update_layout(
        template="plotly_white", height=400,
        xaxis_title="DTE", yaxis_title="MC P&L / Day ($)",
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)


main()
