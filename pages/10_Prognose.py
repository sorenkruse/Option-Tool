"""
Prognose – Forecast-driven Options Scanner

1. Load underlying data + IV
2. User provides directional forecast (% move, DTE)
3. System projects IV change from spot move
4. Scans all single-leg options (LP, LC, SP, SC) across strikes/DTEs
5. Ranks by P&L and risk-adjusted score (P&L × PoP)
6. HeatMap visualization (Strike × DTE)
"""

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    resolve_spot_price, resolve_options_chain, get_available_expirations,
    get_dividend_yield, get_risk_free_rate, get_vix,
    build_smile_curve, interpolate_smile_iv,
)

CSS = "<style>.main .block-container{font-size:14px;padding-top:1rem}" \
      "[data-testid=stMetricValue]{font-size:18px}" \
      "[data-testid=stMetricLabel]{font-size:12px}" \
      ".main h1{font-size:22px;margin-bottom:.3rem}" \
      ".main h3{font-size:16px;margin-top:.6rem;margin-bottom:.3rem}</style>"


def _round(v, step):
    return round(v / step) * step


def _iv(smile, strike, fallback):
    if smile is not None and not smile.empty:
        v = interpolate_smile_iv(smile, strike)
        if not np.isnan(v) and 0.01 < v < 2.0:
            return v
    return fallback


def project_iv(iv_now, spot_now, spot_target):
    """
    Project IV change from spot move.
    Empirical: ~4 vol pts per 10% spot drop, asymmetric.
    """
    pct_move = (spot_target - spot_now) / spot_now
    # Negative moves increase IV more than positive moves decrease it
    if pct_move < 0:
        iv_shift = -pct_move * 0.40  # 10% drop → +4% IV
    else:
        iv_shift = -pct_move * 0.25  # 10% rally → -2.5% IV
    return max(iv_now + iv_shift, 0.05)


def calc_pop(spot, strike, T, r, iv, q, opt_type, is_long, premium):
    """
    Probability of Profit for a single-leg option.

    For long options: P(option value at expiry > premium paid)
    For short options: P(option expires worthless or partial, keeping premium)
    """
    from scipy.stats import norm

    if T <= 0 or iv <= 0:
        return 0.0

    d2 = (np.log(spot / strike) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))

    if opt_type == "call":
        # P(S_T > strike) = N(d2)
        p_itm = norm.cdf(d2)
    else:
        # P(S_T < strike) = N(-d2)
        p_itm = norm.cdf(-d2)

    if is_long:
        # Need enough move to cover premium
        if opt_type == "call":
            be = strike + premium
            d2_be = (np.log(spot / be) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            return norm.cdf(d2_be)
        else:
            be = strike - premium
            if be <= 0:
                return 0.0
            d2_be = (np.log(spot / be) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            return norm.cdf(-d2_be)
    else:
        # Short: profit if option expires worthless
        return 1.0 - p_itm


def scan_options(spot, r, q, iv_atm, target_spot, target_iv,
                 forecast_dte, strike_step, expirations_data,
                 scan_dtes):
    """
    Scan all single-leg options across strikes and DTEs.

    Returns list of dicts with P&L, PoP, Score for each option.
    """
    results = []

    # Strike range: ±15% from spot, rounded
    strike_low = _round(spot * 0.85, strike_step)
    strike_high = _round(spot * 1.15, strike_step)
    strikes = np.arange(strike_low, strike_high + strike_step, strike_step)

    for exp_str, chain_data in expirations_data.items():
        dte = chain_data["dte_actual"]
        if dte not in scan_dtes:
            continue

        call_smile = build_smile_curve(chain_data["calls"], spot)
        put_smile = build_smile_curve(chain_data["puts"], spot)

        # Time remaining after forecast period
        remaining_dte = max(dte - forecast_dte, 1)
        T_entry = dte / 365.0
        T_exit = remaining_dte / 365.0

        for strike in strikes:
            for opt_type in ["call", "put"]:
                smile = call_smile if opt_type == "call" else put_smile
                iv_entry = _iv(smile, strike, iv_atm)

                # Entry price
                try:
                    entry = bs.calculate_all(spot, strike, T_entry, r, iv_entry, q, opt_type)
                except Exception:
                    continue

                price = entry["price"]
                if price < 0.10:
                    continue

                # Exit price at forecast target
                # IV at exit: project from target spot, adjusted for strike
                iv_exit_atm = target_iv
                # Skew adjustment: OTM puts get more IV in drops
                if opt_type == "put" and target_spot < spot:
                    moneyness = strike / target_spot
                    iv_exit = iv_exit_atm * (1 + 0.15 * max(0, moneyness - 1))
                elif opt_type == "call" and target_spot > spot:
                    moneyness = target_spot / strike
                    iv_exit = iv_exit_atm * (1 - 0.05 * max(0, moneyness - 1))
                else:
                    iv_exit = iv_exit_atm

                iv_exit = max(iv_exit, 0.05)

                try:
                    exit_res = bs.calculate_all(target_spot, strike, T_exit, r, iv_exit, q, opt_type)
                except Exception:
                    continue

                exit_price = exit_res["price"]

                for is_long in [True, False]:
                    if is_long:
                        pnl = exit_price - price
                        label = f"L{'C' if opt_type == 'call' else 'P'}"
                    else:
                        pnl = price - exit_price
                        label = f"S{'C' if opt_type == 'call' else 'P'}"

                    pop = calc_pop(spot, strike, T_entry, r, iv_entry, q,
                                   opt_type, is_long, price)

                    # Score: expected value = P&L × PoP
                    ev = pnl * pop

                    # ROC
                    capital = price if is_long else max(abs(entry["delta"]) * spot * 0.1, price)
                    roc = pnl / capital if capital > 0 else 0

                    results.append({
                        "leg": label,
                        "strike": strike,
                        "dte": dte,
                        "exp": exp_str,
                        "opt_type": opt_type,
                        "is_long": is_long,
                        "entry": price,
                        "exit": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl / price * 100 if price > 0 else 0,
                        "pop": pop,
                        "ev": ev,
                        "roc": roc,
                        "delta": entry["delta"],
                        "iv_entry": iv_entry,
                        "iv_exit": iv_exit,
                    })

    return results


# ── Display ──────────────────────────────────────────────────────────────

def _fmt_table(df_subset):
    """Format a dataframe for display."""
    d = df_subset[["strike", "dte", "entry", "exit", "pnl", "pnl_pct",
                    "pop", "ev", "delta", "iv_entry"]].copy()
    d.columns = ["Strike", "DTE", "Entry", "Exit", "P&L", "P&L%",
                  "PoP", "EV", "Delta", "IV"]
    d["Strike"] = d["Strike"].map(lambda x: f"{x:,.0f}")
    d["Entry"]  = d["Entry"].map(lambda x: f"${x:.2f}")
    d["Exit"]   = d["Exit"].map(lambda x: f"${x:.2f}")
    d["P&L"]    = d["P&L"].map(lambda x: f"${x:+.2f}")
    d["P&L%"]   = d["P&L%"].map(lambda x: f"{x:+.0f}%")
    d["PoP"]    = d["PoP"].map(lambda x: f"{x*100:.0f}%")
    d["EV"]     = d["EV"].map(lambda x: f"${x:+.2f}")
    d["Delta"]  = d["Delta"].map(lambda x: f"{x:.3f}")
    d["IV"]     = d["IV"].map(lambda x: f"{x*100:.1f}%")
    return d


def display(res):
    spot = res["spot"]
    target = res["target_spot"]
    move_pct = res["move_pct"]

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target", f"{target:,.0f}", f"{move_pct:+.1f}%")
    m2.metric("IV Now", f"{res['iv_now']*100:.1f}%")
    m3.metric("IV Projected", f"{res['iv_target']*100:.1f}%",
              f"{(res['iv_target']-res['iv_now'])*100:+.1f}%")
    m4.metric("Options Scanned", f"{len(res['all_results']):,}")

    df = pd.DataFrame(res["all_results"])
    if df.empty:
        st.warning("No options found.")
        return

    # ── Top 4 per type (summary) ──
    st.markdown("### Best Picks (Top 1 per Type)")
    top_picks = []
    for leg in ["LC", "LP", "SC", "SP"]:
        sub = df[df["leg"] == leg].sort_values("ev", ascending=False)
        if not sub.empty:
            row = sub.iloc[0]
            top_picks.append({
                "Leg": leg,
                "Strike": f"{row['strike']:,.0f}",
                "DTE": row["dte"],
                "Entry": f"${row['entry']:.2f}",
                "P&L": f"${row['pnl']:+.2f}",
                "P&L%": f"{row['pnl_pct']:+.0f}%",
                "PoP": f"{row['pop']*100:.0f}%",
                "EV": f"${row['ev']:+.2f}",
                "Delta": f"{row['delta']:.3f}",
            })
    if top_picks:
        st.dataframe(pd.DataFrame(top_picks), use_container_width=True,
                      hide_index=True)

    # OptionStrat + IBKR for all top picks
    if top_picks:
        cols = st.columns(len(top_picks))
        for i, pick_row in enumerate(top_picks):
            leg_type = pick_row["Leg"]
            sub = df[df["leg"] == leg_type].sort_values("ev", ascending=False)
            if sub.empty:
                continue
            top = sub.iloc[0]
            os_legs = [{"strike": top["strike"], "option_type": top["opt_type"],
                         "expiration": top["exp"], "long": top["is_long"], "qty": 1}]
            with cols[i]:
                url = bs.optionstrat_url(res["symbol"], os_legs)
                if url:
                    st.markdown(f"[{leg_type} OptionStrat]({url})")
                csv = bs.ibkr_basket_csv(res["symbol"], os_legs, tag=leg_type)
                st.download_button(f"{leg_type} CSV", csv,
                                    f"prognose_{leg_type}_{res['symbol']}.csv",
                                    "text/csv", key=f"csv_{leg_type}")

    # ── Sort control ──
    sort_by = st.selectbox("Sort all tables by",
                            ["ev", "pnl", "pop", "roc"],
                            format_func=lambda x: {"ev": "Expected Value",
                                "pnl": "P&L", "pop": "Prob. of Profit",
                                "roc": "Return on Capital"}[x])

    # ── Per-type tables ──
    for leg in ["LC", "LP", "SC", "SP"]:
        label = {"LC": "Long Call", "LP": "Long Put",
                 "SC": "Short Call", "SP": "Short Put"}[leg]
        sub = df[df["leg"] == leg].sort_values(sort_by, ascending=False).head(15)
        if sub.empty:
            continue
        st.markdown(f"### {label}")
        st.dataframe(_fmt_table(sub), use_container_width=True, hide_index=True)

    # ── HeatMap ──
    st.markdown("### HeatMap")
    hc1, hc2 = st.columns(2)
    with hc1:
        hm_leg = st.selectbox("Leg Type", ["LC", "LP", "SC", "SP"], key="hm_leg")
    with hc2:
        hm_color = st.radio("Color", ["P&L", "Score (EV)"],
                              horizontal=True, key="hm_color")

    hm_data = df[df["leg"] == hm_leg].copy()
    if hm_data.empty:
        st.info(f"No data for {hm_leg}.")
        return

    color_col = "pnl" if hm_color == "P&L" else "ev"
    pivot = hm_data.pivot_table(index="strike", columns="dte",
                                 values=color_col, aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    # Trim to strikes with meaningful values (not all NaN or near zero)
    row_max = pivot.abs().max(axis=1)
    pivot = pivot[row_max > 0.5]

    if pivot.empty:
        st.info("No meaningful data for HeatMap.")
        return

    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{int(c)}d" for c in pivot.columns],
        y=[f"{int(s):,}" for s in pivot.index],
        colorscale=[[0, "#d32f2f"], [0.5, "#ffffff"], [1, "#388e3c"]],
        zmid=0,
        text=[[f"${v:.0f}" if not np.isnan(v) else "" for v in row]
              for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Strike: %{y}<br>DTE: %{x}<br>Value: $%{z:.1f}<extra></extra>",
        colorbar_title=hm_color,
    ))

    fig.update_layout(
        template="plotly_white",
        height=max(350, min(len(pivot) * 20, 800)),
        xaxis_title="DTE",
        yaxis_title="Strike",
        margin=dict(l=80, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, move_pct, forecast_dte, strike_step, scan_dte_targets):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(forecast_dte)
    vix = get_vix()
    iv_now = vix / 100.0 if not np.isnan(vix) else 0.20

    target_spot = spot * (1 + move_pct / 100.0)
    iv_target = project_iv(iv_now, spot, target_spot)

    # Load chains for scan DTEs
    today = datetime.date.today()
    try:
        exps, _ = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")

    # Map expirations to DTEs
    exp_dte = {}
    for e in exps:
        try:
            d = (datetime.date.fromisoformat(e) - today).days
            if d >= 3:
                exp_dte[e] = d
        except Exception:
            continue

    # Find closest expiration for each target DTE
    scan_dtes = set()
    expirations_data = {}
    for target_dte in scan_dte_targets:
        best_exp = None
        best_diff = 999
        for e, d in exp_dte.items():
            diff = abs(d - target_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = e
        if best_exp and best_exp not in expirations_data:
            try:
                ch, _, _ = resolve_options_chain(symbol, exp_dte[best_exp])
                expirations_data[ch["expiration"]] = ch
                scan_dtes.add(ch["dte_actual"])
            except Exception:
                continue

    if not expirations_data:
        raise ValueError("Could not load option chains.")

    results = scan_options(spot, r, q, iv_now, target_spot, iv_target,
                           forecast_dte, strike_step, expirations_data,
                           scan_dtes)

    return {
        "symbol": symbol, "spot": spot, "vix": vix,
        "iv_now": iv_now, "iv_target": iv_target,
        "target_spot": target_spot, "move_pct": move_pct,
        "forecast_dte": forecast_dte,
        "all_results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Prognose")
    st.caption("Forecast-driven options scanner. "
               "Enter your directional view and find the best option to express it.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX").upper()
    with c2:
        move_pct = st.number_input("Move %", value=-2.0,
            min_value=-20.0, max_value=20.0, step=0.5, format="%.1f",
            help="Expected % move. Negative = bearish, positive = bullish.")
    with c3:
        forecast_dte = st.number_input("Forecast DTE", value=5,
            min_value=1, max_value=30, step=1,
            help="Days until expected move completes.")
    with c4:
        default_step = 25 if "SPX" in symbol else 5
        strike_step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50)

    # Scan DTEs: forecast + 14d + 30d
    scan_targets = sorted(set([forecast_dte,
                                max(forecast_dte, 14),
                                max(forecast_dte, 30)]))

    if st.button("Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning options..."):
            try:
                result = compute(symbol, move_pct, forecast_dte,
                                  strike_step, scan_targets)
                st.session_state["prog_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "prog_result" not in st.session_state:
        st.info("Enter your forecast and click Scan.")
        return

    display(st.session_state["prog_result"])


main()
