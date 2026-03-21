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
                 scan_dtes, conviction=0.5):
    """
    Scan all single-leg options across strikes and DTEs.

    conviction: 0.0-1.0, determines scenario weights:
        Full move weight  = 0.15 + 0.45 * conviction
        Half move weight  = 0.25 + 0.10 * conviction
        Flat weight       = 0.60 - 0.55 * conviction
    """
    results = []

    # Scenario weights from conviction
    w_full = 0.15 + 0.45 * conviction
    w_half = 0.25 + 0.10 * conviction
    w_flat = 1.0 - w_full - w_half

    # Scenario spots and IVs
    half_spot = spot + (target_spot - spot) * 0.5
    half_iv = project_iv(iv_atm, spot, half_spot)
    flat_iv = max(iv_atm - 0.005, 0.05)  # slight IV decay when flat

    # Strike range: ±8% from spot (tradeable range)
    strike_low = _round(spot * 0.92, strike_step)
    strike_high = _round(spot * 1.08, strike_step)
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
                # OTM only: calls above spot, puts below spot
                if opt_type == "call" and strike <= spot:
                    continue
                if opt_type == "put" and strike >= spot:
                    continue

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

                if abs(entry["delta"]) > 0.85:
                    continue

                # Compute exit price for 3 scenarios
                # Key: exit IV = entry smile IV + projected IV CHANGE (not raw ATM IV)
                iv_change_full = target_iv - iv_atm
                iv_change_half = half_iv - iv_atm
                iv_change_flat = flat_iv - iv_atm

                def _exit_price(s_target, iv_shift):
                    iv_e = iv_entry + iv_shift  # smile IV + change
                    # Skew adjustment for large moves
                    if opt_type == "put" and s_target < spot:
                        m = strike / s_target
                        iv_e *= (1 + 0.15 * max(0, m - 1))
                    elif opt_type == "call" and s_target > spot:
                        m = s_target / strike
                        iv_e *= (1 - 0.05 * max(0, m - 1))
                    iv_e = max(iv_e, 0.05)
                    try:
                        return bs.calculate_all(s_target, strike, T_exit, r, iv_e, q, opt_type)["price"]
                    except Exception:
                        return None

                exit_full = _exit_price(target_spot, iv_change_full)
                exit_half = _exit_price(half_spot, iv_change_half)
                exit_flat = _exit_price(spot, iv_change_flat)

                if exit_full is None or exit_half is None or exit_flat is None:
                    continue

                for is_long in [True, False]:
                    # Short positions: enforce 2% minimum distance from spot
                    if not is_long:
                        if opt_type == "call" and strike < spot * 1.02:
                            continue
                        if opt_type == "put" and strike > spot * 0.98:
                            continue

                    sign = 1 if is_long else -1
                    pnl_full = sign * (exit_full - price)
                    pnl_half = sign * (exit_half - price)
                    pnl_flat = sign * (exit_flat - price)

                    # Weighted P&L
                    pnl_weighted = w_full * pnl_full + w_half * pnl_half + w_flat * pnl_flat

                    label = ("L" if is_long else "S") + ("C" if opt_type == "call" else "P")

                    pop = calc_pop(spot, strike, T_entry, r, iv_entry, q,
                                   opt_type, is_long, price)

                    ev = pnl_weighted * pop

                    capital = price if is_long else max(abs(entry["delta"]) * spot * 0.1, price)
                    roc = pnl_weighted / capital if capital > 0 else 0

                    results.append({
                        "leg": label,
                        "strike": strike,
                        "dte": dte,
                        "exp": exp_str,
                        "opt_type": opt_type,
                        "is_long": is_long,
                        "entry": price,
                        "exit": exit_full,
                        "pnl": pnl_full,
                        "pnl_half": pnl_half,
                        "pnl_flat": pnl_flat,
                        "pnl_w": pnl_weighted,
                        "pnl_pct": pnl_weighted / price * 100 if price > 0 else 0,
                        "pop": pop,
                        "ev": ev,
                        "roc": roc,
                        "delta": entry["delta"],
                        "iv_entry": iv_entry,
                    })

    return results


# ── Display ──────────────────────────────────────────────────────────────

def _fmt_table(df_subset):
    """Format a dataframe for display. P&L values multiplied by 100 (contract size)."""
    d = df_subset[["strike", "dte", "entry", "pnl", "pnl_half",
                    "pnl_flat", "pnl_w", "pnl_pct",
                    "pop", "ev", "delta", "iv_entry"]].copy()
    d.columns = ["Strike", "DTE", "Entry", "Full", "Half", "Flat",
                  "Weighted", "W%", "PoP", "EV", "Delta", "IV"]
    d["Strike"]   = d["Strike"].map(lambda x: f"{x:,.0f}")
    d["Entry"]    = d["Entry"].map(lambda x: f"${x*100:,.0f}")
    d["Full"]     = d["Full"].map(lambda x: f"${x*100:+,.0f}")
    d["Half"]     = d["Half"].map(lambda x: f"${x*100:+,.0f}")
    d["Flat"]     = d["Flat"].map(lambda x: f"${x*100:+,.0f}")
    d["Weighted"] = d["Weighted"].map(lambda x: f"${x*100:+,.0f}")
    d["W%"]       = d["W%"].map(lambda x: f"{x:+.0f}%")
    d["PoP"]      = d["PoP"].map(lambda x: f"{x*100:.0f}%")
    d["EV"]       = d["EV"].map(lambda x: f"${x*100:+,.0f}")
    d["Delta"]    = d["Delta"].map(lambda x: f"{x:.3f}")
    d["IV"]       = d["IV"].map(lambda x: f"{x*100:.1f}%")
    return d


def display(res):
    spot = res["spot"]
    target = res["target_spot"]
    move_pct = res["move_pct"]

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Target", f"{target:,.0f}", f"{move_pct:+.1f}%",
              help="Projected spot price = current spot × (1 + move%)")
    m2.metric("IV Now", f"{res['iv_now']*100:.1f}%",
              help="Current implied volatility from VIX")
    m3.metric("IV Projected", f"{res['iv_target']*100:.1f}%",
              f"{(res['iv_target']-res['iv_now'])*100:+.1f}%",
              help="Projected IV at target spot. Drops increase IV (~4pts/10%), "
                   "rallies decrease IV (~2.5pts/10%). Asymmetric like real markets.")
    conv = res.get("conviction", 0.5)
    w_full = 0.15 + 0.45 * conv
    w_half = 0.25 + 0.10 * conv
    w_flat = 1.0 - w_full - w_half
    m4.metric("Conviction", f"{conv*100:.0f}%",
              help="Your confidence level. Determines how much weight "
                   "the full-move scenario gets vs. flat (no move).")
    m5.metric("Weights", f"{w_full:.0%}/{w_half:.0%}/{w_flat:.0%}",
              help="Scenario weights: Full Move / Half Move / Flat. "
                   "These weight the 3 P&L scenarios into the Weighted P&L and EV.")

    df = pd.DataFrame(res["all_results"])
    if df.empty:
        st.warning("No options found.")
        return

    # ── Top picks per type + underlying benchmark ──
    st.markdown("### Best Picks (Top 1 per Type)")

    # Underlying benchmark
    spot = res["spot"]
    target = res["target_spot"]
    move_pct = res["move_pct"]

    # Determine futures proxy
    sym_upper = res["symbol"].upper().replace("^", "")
    is_bearish = move_pct < 0

    top_picks = []

    # Underlying row first - direction follows forecast
    is_bearish = move_pct < 0
    if sym_upper in ("SPX", "GSPC"):
        fut_label = "10 MES Short" if is_bearish else "10 MES Long"
        fut_multiplier = 5.0
        fut_qty = 10
        fut_margin = 1_500
    elif sym_upper == "SPY":
        fut_label = "100 Shares Short" if is_bearish else "100 Shares Long"
        fut_multiplier = 1.0
        fut_qty = 100
        fut_margin = spot * 100 * 0.5
    elif sym_upper in ("NDX", "QQQ"):
        fut_label = "10 MNQ Short" if is_bearish else "10 MNQ Long"
        fut_multiplier = 2.0
        fut_qty = 10
        fut_margin = 2_000
    else:
        fut_label = "100 Shares Short" if is_bearish else "100 Shares Long"
        fut_multiplier = 1.0
        fut_qty = 100
        fut_margin = spot * 100 * 0.5

    fut_pnl_per_pt = fut_multiplier * fut_qty
    # Short futures profit from drops, long from rallies
    if is_bearish:
        fut_pnl = (spot - target) * fut_pnl_per_pt  # short: profit when spot drops
    else:
        fut_pnl = (target - spot) * fut_pnl_per_pt
    fut_capital = fut_margin * fut_qty if fut_multiplier > 1 else fut_margin
    fut_roc = fut_pnl / fut_capital * 100 if fut_capital > 0 else 0

    top_picks.append({
        "Leg": fut_label,
        "Strike": f"{spot:,.0f}",
        "DTE": res["forecast_dte"],
        "Entry": f"${fut_capital:,.0f}",
        "Full": f"${fut_pnl:+,.0f}",
        "Half": f"${fut_pnl/2:+,.0f}",
        "Flat": "$0",
        "Weighted": f"${fut_pnl * (w_full + w_half*0.5):+,.0f}",
        "PoP": "n/a",
        "EV": "n/a",
    })

    # Options rows (P&L multiplied by 100 for real contract value)
    for leg in ["LC", "LP", "SC", "SP"]:
        sub = df[df["leg"] == leg].sort_values("ev", ascending=False)
        if not sub.empty:
            row = sub.iloc[0]
            top_picks.append({
                "Leg": leg,
                "Strike": f"{row['strike']:,.0f}",
                "DTE": row["dte"],
                "Entry": f"${row['entry']*100:,.0f}",
                "Full": f"${row['pnl']*100:+,.0f}",
                "Half": f"${row['pnl_half']*100:+,.0f}",
                "Flat": f"${row['pnl_flat']*100:+,.0f}",
                "Weighted": f"${row['pnl_w']*100:+,.0f}",
                "PoP": f"{row['pop']*100:.0f}%",
                "EV": f"${row['ev']*100:+,.0f}",
            })
    if top_picks:
        st.dataframe(pd.DataFrame(top_picks), use_container_width=True,
                      hide_index=True)

    # Combined + per-leg OptionStrat links (options only, skip underlying)
    option_legs = ["LC", "LP", "SC", "SP"]
    if top_picks:
        all_legs = []
        per_leg_data = []
        for pick_row in top_picks:
            leg_type = pick_row["Leg"]
            if leg_type not in option_legs:
                continue
            sub = df[df["leg"] == leg_type].sort_values("ev", ascending=False)
            if sub.empty:
                continue
            top = sub.iloc[0]
            leg_dict = {
                "strike": int(float(top["strike"])),
                "option_type": str(top["opt_type"]),
                "expiration": str(top["exp"]),
                "long": bool(top["is_long"]),
                "qty": 1,
                "price": float(top["entry"]),
            }
            all_legs.append(leg_dict)
            per_leg_data.append((leg_type, leg_dict))

        # Combined link (shorts first, puts before calls - OptionStrat convention)
        all_legs_sorted = sorted(all_legs,
                                  key=lambda l: (l["long"], l["option_type"] != "put"))
        url_all = bs.optionstrat_url(res["symbol"], all_legs_sorted)
        csv_all = bs.ibkr_basket_csv(res["symbol"], all_legs_sorted, tag="Prognose")
        clean_sym = res["symbol"].replace("^", "")
        e1, e2 = st.columns(2)
        with e1:
            if url_all:
                st.markdown(f"[All Picks: OptionStrat]({url_all})")
        with e2:
            st.download_button("All Picks: IBKR CSV", csv_all,
                                f"prognose_all_{clean_sym}.csv",
                                "text/csv", key="csv_all")

        # Per-leg links
        cols = st.columns(len(per_leg_data))
        for i, (leg_type, leg_dict) in enumerate(per_leg_data):
            with cols[i]:
                url = bs.optionstrat_url(res["symbol"], [leg_dict])
                if url:
                    st.markdown(f"[{leg_type}]({url})")
                csv = bs.ibkr_basket_csv(res["symbol"], [leg_dict], tag=leg_type)
                st.download_button(f"{leg_type} CSV", csv,
                                    f"prognose_{leg_type}_{res['symbol']}.csv",
                                    "text/csv", key=f"csv_{leg_type}")

    # ── Sort control + column legend ──
    with st.expander("Column Legend"):
        st.markdown(
            "- **Strike**: Option strike price (OTM only: calls above spot, puts below)\n"
            "- **DTE**: Days to expiration of the option\n"
            "- **Entry**: Option premium at current market (×100 for SPX contract)\n"
            "- **Full**: P&L if your full forecast move happens (spot → target, IV adjusts)\n"
            "- **Half**: P&L if only 50% of the move happens\n"
            "- **Flat**: P&L if spot stays unchanged (pure theta + slight IV decay)\n"
            "- **Weighted**: Conviction-weighted average of Full/Half/Flat scenarios\n"
            "- **W%**: Weighted P&L as % of entry premium\n"
            "- **PoP**: Probability of Profit at expiry (BS-based, break-even adjusted)\n"
            "- **EV**: Expected Value = Weighted P&L × PoP. "
            "Best risk-adjusted metric. Higher = better.\n"
            "- **Delta**: Option delta at entry. "
            "Long Puts have negative delta (profit from drops), "
            "Short Calls also profit from drops (positive theta, negative P&L exposure)\n"
            "- **IV**: Implied volatility at entry from the smile curve. "
            "Exit IV = Entry IV + projected IV change (not ATM swap)"
        )

    sort_by = st.selectbox("Sort all tables by",
                            ["ev", "pnl", "pop", "roc"],
                            format_func=lambda x: {"ev": "Expected Value",
                                "pnl": "P&L (full move)", "pop": "Prob. of Profit",
                                "roc": "Return on Capital"}[x],
                            help="EV (Expected Value) is the recommended sort: "
                                 "it balances P&L potential with probability. "
                                 "P&L sorts by full-move profit only. "
                                 "PoP sorts by highest probability regardless of profit size. "
                                 "ROC sorts by return on capital deployed.")

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
        hm_leg = st.selectbox("Leg Type", ["LC", "LP", "SC", "SP"], key="hm_leg",
            help="LC = Long Call (profit from rally), "
                 "LP = Long Put (profit from drop), "
                 "SC = Short Call (profit from flat/drop, theta income), "
                 "SP = Short Put (profit from flat/rally, theta income)")
    with hc2:
        hm_color = st.radio("Color", ["P&L", "Score (EV)"],
                              horizontal=True, key="hm_color",
                              help="P&L: conviction-weighted profit/loss per contract. "
                                   "Score (EV): P&L × Probability of Profit. "
                                   "Green = profitable, Red = loss, White = break-even.")

    hm_data = df[df["leg"] == hm_leg].copy()
    if hm_data.empty:
        st.info(f"No data for {hm_leg}.")
        return

    color_col = "pnl_w" if hm_color == "P&L" else "ev"
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

def compute(symbol, move_pct, forecast_dte, strike_step, scan_dte_targets,
            conviction=0.5):
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
        exps, chain_ticker = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")

    if not exps:
        raise ValueError(f"No expirations for {symbol}.")

    # Map expirations to DTEs
    exp_dte = {}
    for e in exps:
        try:
            d = (datetime.date.fromisoformat(e) - today).days
            if d >= 3:
                exp_dte[e] = d
        except Exception:
            continue

    if not exp_dte:
        raise ValueError("No valid expirations found.")

    # Find closest expiration for each target DTE, load chains
    scan_dtes = set()
    expirations_data = {}
    errors = []
    for target_dte in scan_dte_targets:
        best_exp = min(exp_dte.keys(), key=lambda e: abs(exp_dte[e] - target_dte))
        if best_exp in expirations_data:
            continue
        try:
            ch, _, _ = resolve_options_chain(symbol, exp_dte[best_exp])
            expirations_data[ch["expiration"]] = ch
            scan_dtes.add(ch["dte_actual"])
        except Exception as e:
            errors.append(f"DTE {target_dte} ({best_exp}): {e}")
            continue

    if not expirations_data:
        raise ValueError(
            f"Could not load any option chains for {symbol}.\n"
            f"Tried DTEs: {scan_dte_targets}\n"
            f"Available expirations: {len(exp_dte)}\n"
            f"Errors: {'; '.join(errors)}"
        )

    results = scan_options(spot, r, q, iv_now, target_spot, iv_target,
                           forecast_dte, strike_step, expirations_data,
                           scan_dtes, conviction)

    return {
        "symbol": symbol, "spot": spot, "vix": vix,
        "iv_now": iv_now, "iv_target": iv_target,
        "target_spot": target_spot, "move_pct": move_pct,
        "forecast_dte": forecast_dte,
        "conviction": conviction,
        "all_results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Prognose")
    st.caption("Forecast-driven options scanner. "
               "Enter your directional view and find the best option to express it.")

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
            help="Underlying to scan. ^SPX uses SPXW options via Yahoo Finance, "
                 "with SPY fallback if SPXW unavailable.").upper()
    with c2:
        move_pct = st.number_input("Move %", value=-2.0,
            min_value=-20.0, max_value=20.0, step=0.5, format="%.1f",
            help="Your expected % move of the underlying. "
                 "Negative = bearish (drop), positive = bullish (rally). "
                 "Example: -2.0 means you expect a 2% drop. "
                 "The system projects IV change automatically: "
                 "drops increase IV (~4pts per 10% drop), rallies decrease IV (~2.5pts per 10%).")
    with c3:
        forecast_dte = st.number_input("Forecast DTE", value=5,
            min_value=1, max_value=30, step=1,
            help="Days until you expect the move to complete. "
                 "This determines how much theta decay to factor in. "
                 "Options are scanned at 3 DTEs: this value, ~14d, and ~30d. "
                 "Shorter forecast = more theta impact on results.")
    with c4:
        conviction = st.number_input("Conviction %", value=50,
            min_value=10, max_value=90, step=10,
            help="How confident you are in your forecast. Controls scenario weighting:\n"
                 "- Full Move: your forecast hits exactly\n"
                 "- Half Move: only 50% of expected move\n"
                 "- Flat: market doesn't move, only theta works\n\n"
                 "Low conviction (20%): 24% Full / 27% Half / 49% Flat → favors short options (theta income)\n"
                 "Medium (50%): 38% Full / 30% Half / 32% Flat → balanced\n"
                 "High (80%): 51% Full / 33% Half / 16% Flat → favors long options (directional bet)")
    with c5:
        default_step = 25 if "SPX" in symbol else 5
        strike_step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50,
            help="Strike increment for scanning. Larger = faster but fewer results. "
                 "SPX: 25 (scans every 25 pts), SPY: 5.")

    # Scan DTEs: forecast + 14d + 30d
    scan_targets = sorted(set([forecast_dte,
                                max(forecast_dte, 14),
                                max(forecast_dte, 30)]))

    if st.button("Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning options..."):
            try:
                result = compute(symbol, move_pct, forecast_dte,
                                  strike_step, scan_targets,
                                  conviction / 100.0)
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
