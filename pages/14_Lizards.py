"""
Lizards – Renewal trades for Crawling Crab cycles.

Reverse Jade Lizard (Bull): Short Call + Bull Put Spread (SP + LP)
Jade Lizard (Bear): Short Put + Bear Call Spread (SC + LC)

These are the 3 short-DTE legs that get rolled every cycle.
The existing LEAPS from Crawling Crab stays open.
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


# ── Scan ─────────────────────────────────────────────────────────────────

def scan_lizards(spot, r, q, iv_atm, step, direction,
                 dte_actual, smile_call, smile_put,
                 diag_delta, spread_d_short, spread_d_long):
    """Build one lizard configuration for a given DTE."""
    is_bull = direction == "Bull"
    T = dte_actual / 365.0

    # Naked short (same direction as trend)
    # Bull: Short Call (above spot) | Bear: Short Put (below spot)
    naked_type = "call" if is_bull else "put"
    naked_smile = smile_call if is_bull else smile_put
    try:
        if is_bull:
            naked_k = _round(bs.solve_strike_for_delta(
                diag_delta, spot, T, r, iv_atm, q, "call"), step)
        else:
            naked_k = _round(bs.solve_strike_for_delta(
                -diag_delta, spot, T, r, iv_atm, q, "put"), step)
    except Exception:
        return None
    iv_naked = _iv(naked_smile, naked_k, iv_atm)
    naked = bs.calculate_all(spot, naked_k, T, r, iv_naked, q, naked_type)

    # Credit spread (opposite direction)
    # Bull: Bull Put Spread (SP + LP) | Bear: Bear Call Spread (SC + LC)
    spread_type = "put" if is_bull else "call"
    spread_smile = smile_put if is_bull else smile_call
    try:
        if is_bull:
            sp_k = _round(bs.solve_strike_for_delta(
                -spread_d_short, spot, T, r, iv_atm, q, "put"), step)
            lp_k = _round(bs.solve_strike_for_delta(
                -spread_d_long, spot, T, r, iv_atm, q, "put"), step)
            if sp_k <= lp_k:
                return None
        else:
            sp_k = _round(bs.solve_strike_for_delta(
                spread_d_short, spot, T, r, iv_atm, q, "call"), step)
            lp_k = _round(bs.solve_strike_for_delta(
                spread_d_long, spot, T, r, iv_atm, q, "call"), step)
            if lp_k <= sp_k:
                return None
    except Exception:
        return None

    iv_sp = _iv(spread_smile, sp_k, iv_atm)
    iv_lp = _iv(spread_smile, lp_k, iv_atm)
    sp = bs.calculate_all(spot, sp_k, T, r, iv_sp, q, spread_type)
    lp = bs.calculate_all(spot, lp_k, T, r, iv_lp, q, spread_type)

    spread_credit = sp["price"] - lp["price"]
    if spread_credit <= 0:
        return None

    total_credit = naked["price"] + spread_credit
    income_50 = total_credit * 0.50

    # Greeks
    delta = (-naked["delta"]
             - sp["delta"] + lp["delta"])
    theta = (-naked["theta_daily"]
             + sp["theta_daily"] - lp["theta_daily"])
    vega = (-naked["vega_pct"]
            + sp["vega_pct"] - lp["vega_pct"])

    # Max loss on spread side
    spread_width = abs(sp_k - lp_k)
    max_loss_spread = spread_width - spread_credit

    # Days to 50%
    days_to_50 = income_50 / theta if theta > 0 else dte_actual * 0.67

    return {
        "dte": dte_actual,
        "naked_k": naked_k, "sp_k": sp_k, "lp_k": lp_k,
        "naked": naked, "sp": sp, "lp": lp,
        "iv_naked": iv_naked, "iv_sp": iv_sp, "iv_lp": iv_lp,
        "naked_type": naked_type, "spread_type": spread_type,
        "naked_premium": naked["price"],
        "spread_credit": spread_credit,
        "total_credit": total_credit,
        "income_50": income_50,
        "delta": delta, "theta": theta, "vega": vega,
        "max_loss_spread": max_loss_spread,
        "spread_width": spread_width,
        "days_to_50": days_to_50,
    }


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, direction, diag_delta, spread_d_short, spread_d_long):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(30)
    vix = get_vix()
    iv = vix / 100.0 if not np.isnan(vix) else 0.30
    today = datetime.date.today()

    try:
        exps, _ = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")
    if not exps:
        raise ValueError(f"No expirations for {symbol}.")

    # Scan all DTEs from 10d to 50d
    results = []
    seen_dtes = set()
    for exp_str in exps:
        try:
            d = (datetime.date.fromisoformat(exp_str) - today).days
            if d < 10 or d > 50:
                continue
            ch, _, _ = resolve_options_chain(symbol, d)
            dte = ch["dte_actual"]
            if dte in seen_dtes:
                continue
            seen_dtes.add(dte)

            sm_call = build_smile_curve(ch["calls"], spot)
            sm_put = build_smile_curve(ch["puts"], spot)

            res = scan_lizards(spot, r, q, iv, step, direction,
                                dte, sm_call, sm_put,
                                diag_delta, spread_d_short, spread_d_long)
            if res:
                res["exp"] = ch["expiration"]
                results.append(res)
        except Exception:
            continue

    if not results:
        raise ValueError("No valid lizard configurations found.")

    # Sort by income_50 / days_to_50 (best daily income)
    results.sort(key=lambda x: x["income_50"] / max(x["days_to_50"], 1), reverse=True)

    # Also load chains for diagonal offset
    chain_by_exp = {}
    for exp_str in exps:
        try:
            d = (datetime.date.fromisoformat(exp_str) - today).days
            if d >= 7 and d <= 70:
                ch, _, _ = resolve_options_chain(symbol, d)
                chain_by_exp[ch["expiration"]] = ch
        except Exception:
            continue

    return {
        "symbol": symbol, "spot": spot, "vix": vix, "iv": iv,
        "r": r, "q": q, "direction": direction,
        "all_results": results,
        "chain_by_exp": chain_by_exp,
    }


# ── Display ──────────────────────────────────────────────────────────────

SCAN_CFG = {
    "DTE": st.column_config.NumberColumn("DTE", help="Days to expiration for all 3 legs"),
    "Naked": st.column_config.TextColumn("Naked",
        help="Naked short strike (call for bull, put for bear)"),
    "Spread": st.column_config.TextColumn("Spread",
        help="Credit spread strikes (short/long)"),
    "Credit": st.column_config.TextColumn("Credit",
        help="Total credit received from all 3 legs (×100)"),
    "50% Inc": st.column_config.TextColumn("50% Inc",
        help="Income when closing at 50% profit (×100)"),
    "Days50": st.column_config.TextColumn("Days50",
        help="Estimated days to reach 50% profit"),
    "$/day": st.column_config.TextColumn("$/day",
        help="Daily income rate = 50% income / days to 50% (×100)"),
    "Delta": st.column_config.TextColumn("Delta",
        help="Net position delta"),
    "Theta": st.column_config.TextColumn("Theta",
        help="Daily theta income (×100)"),
    "Max Loss": st.column_config.TextColumn("Max Loss",
        help="Max loss on the spread side (×100)"),
}


def display(res):
    spot = res["spot"]
    direction = res["direction"]
    is_bull = direction == "Bull"
    name = "Reverse Jade Lizard" if is_bull else "Jade Lizard"
    naked_label = "Short Call" if is_bull else "Short Put"
    spread_label = "Bull Put Spread" if is_bull else "Bear Call Spread"

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  {name}")

    results = res["all_results"]
    diag_on = res.get("diag_spread", False)

    # Results table
    st.markdown("### Best DTEs")
    st.caption("Sorted by daily income rate (50% profit / days to reach it).")

    rows = []
    for r in results:
        daily = r["income_50"] / max(r["days_to_50"], 1)
        rows.append({
            "DTE": r["dte"],
            "Naked": f"{r['naked_k']:,.0f}",
            "Spread": f"{r['sp_k']:,.0f}/{r['lp_k']:,.0f}",
            "Credit": f"${r['total_credit']*100:,.0f}",
            "50% Inc": f"${r['income_50']*100:,.0f}",
            "Days50": f"{r['days_to_50']:.0f}",
            "$/day": f"${daily*100:,.0f}",
            "Delta": f"{r['delta']:.3f}",
            "Theta": f"${r['theta']*100:+,.0f}",
            "Max Loss": f"${r['max_loss_spread']*100:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                  hide_index=True, column_config=SCAN_CFG)

    # Select
    options = [f"{r['dte']}d: {naked_label} {r['naked_k']:.0f} + "
               f"{spread_label} {r['sp_k']:.0f}/{r['lp_k']:.0f} → "
               f"${r['income_50']*100:,.0f}/cycle"
               for r in results]
    idx = st.selectbox("Select configuration", range(len(options)),
                        format_func=lambda i: options[i], key="liz_sel")
    pick = results[idx]

    # Diagonal offset for LP
    sc_dte_val = pick["dte"]
    if diag_on:
        avail_dtes = sorted(c["dte_actual"] for c in res["chain_by_exp"].values())
        lp_dte_min = round(sc_dte_val * 1.20)
        lp_dte_val = sc_dte_val
        for d in avail_dtes:
            if d >= lp_dte_min and d > sc_dte_val:
                lp_dte_val = d
                break
        lp_role = f"{spread_label} (Diag +{lp_dte_val - sc_dte_val}d)"
    else:
        lp_dte_val = sc_dte_val
        lp_role = spread_label

    # Position
    st.markdown("### Position")
    sp_type = "Put" if is_bull else "Call"
    leg_rows = [
        {"Leg": naked_label, "Strike": f"${pick['naked_k']:,.0f}",
         "Delta": f"{pick['naked']['delta']:.3f}",
         "IV": f"{pick['iv_naked']*100:.1f}%",
         "Price": f"${pick['naked']['price']*100:,.0f}",
         "DTE": sc_dte_val, "Role": "Naked Short"},
        {"Leg": f"Short {sp_type}", "Strike": f"${pick['sp_k']:,.0f}",
         "Delta": f"{pick['sp']['delta']:.3f}",
         "IV": f"{pick['iv_sp']*100:.1f}%",
         "Price": f"${pick['sp']['price']*100:,.0f}",
         "DTE": sc_dte_val, "Role": spread_label},
        {"Leg": f"Long {sp_type}", "Strike": f"${pick['lp_k']:,.0f}",
         "Delta": f"{pick['lp']['delta']:.3f}",
         "IV": f"{pick['iv_lp']*100:.1f}%",
         "Price": f"${pick['lp']['price']*100:,.0f}",
         "DTE": lp_dte_val, "Role": lp_role},
    ]
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # Metrics
    st.markdown("### Economics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Credit", f"${pick['total_credit']*100:,.0f}",
              help="Premium received from all 3 legs (×100)")
    m2.metric("50% Income", f"${pick['income_50']*100:,.0f}",
              help="Target profit when closing at 50%")
    m3.metric("Days to 50%", f"{pick['days_to_50']:.0f}d",
              help="Estimated days to reach 50% profit target")
    daily = pick["income_50"] / max(pick["days_to_50"], 1)
    m4.metric("$/day", f"${daily*100:,.0f}",
              help="Daily income rate (×100)")
    m5.metric("Net Delta", f"{pick['delta']:.3f}",
              help="Net position delta")
    m6.metric("Max Loss (Spread)", f"${pick['max_loss_spread']*100:,.0f}",
              help="Max loss if spot moves through the spread. "
                   "The naked short has theoretically unlimited risk.")

    st.caption(
        f"Sell all 3 legs → wait ~{pick['days_to_50']:.0f}d → "
        f"close at 50% profit (${pick['income_50']*100:,.0f}) → "
        f"repeat with new lizard at current spot."
    )

    # Export
    st.markdown("### Export")
    sc_exp = pick["exp"]
    lp_exp = min(res["chain_by_exp"].keys(),
                  key=lambda e: abs(res["chain_by_exp"][e]["dte_actual"] - lp_dte_val))
    os_legs = [
        {"strike": int(pick["naked_k"]), "option_type": pick["naked_type"],
         "expiration": str(sc_exp), "long": False, "qty": 1},
        {"strike": int(pick["sp_k"]), "option_type": pick["spread_type"],
         "expiration": str(sc_exp), "long": False, "qty": 1},
        {"strike": int(pick["lp_k"]), "option_type": pick["spread_type"],
         "expiration": str(lp_exp), "long": True, "qty": 1},
    ]
    os_sorted = sorted(os_legs, key=lambda l: (l["long"], l["option_type"] != "put"))

    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], os_sorted)
        if url:
            st.markdown(f"[OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], os_sorted, tag="Lizard")
        st.download_button("IBKR Basket CSV", csv,
                            f"lizard_{res['symbol'].replace('^','')}.csv", "text/csv")

    with st.expander("Strategy Guide"):
        st.markdown(
            f"**{name}** = {naked_label} + {spread_label}\n\n"
            f"**3 Legs:**\n"
            f"- {naked_label} {pick['naked_k']:.0f}: "
            f"${pick['naked']['price']*100:,.0f} premium\n"
            f"- Short {sp_type} {pick['sp_k']:.0f}: "
            f"${pick['sp']['price']*100:,.0f} premium\n"
            f"- Long {sp_type} {pick['lp_k']:.0f}: "
            f"${pick['lp']['price']*100:,.0f} cost\n\n"
            f"**Usage in Crawling Crab cycle:**\n"
            f"1. Your LEAPS from Crawling Crab is still open\n"
            f"2. Open this lizard as the renewal trade\n"
            f"3. Wait ~{pick['days_to_50']:.0f}d for 50% profit\n"
            f"4. Close all 3 legs\n"
            f"5. Open a new lizard at current spot\n\n"
            f"**Risk:**\n"
            f"- {naked_label} has uncapped risk on its side "
            f"(but the existing LEAPS provides partial hedge)\n"
            f"- Spread max loss: ${pick['max_loss_spread']*100:,.0f}\n"
            f"- Total credit received ({pick['total_credit']*100:.0f}) "
            f"offsets some of the spread risk"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Lizards")
    st.caption("Renewal trades for Crawling Crab cycles. "
               "Reverse Jade Lizard (bull) or Jade Lizard (bear).")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="GOOGL",
            help="Underlying (same as your Crawling Crab).").upper()
    with c2:
        direction = st.selectbox("Direction", ["Bull", "Bear"],
            help="Bull: Reverse Jade Lizard (Short Call + Bull Put Spread). "
                 "Bear: Jade Lizard (Short Put + Bear Call Spread).")
    with c3:
        diag_spread = st.toggle("Diagonal Spread", value=True,
            help="Long leg of credit spread gets +20% DTE. "
                 "Better vega protection in IV spikes.")

    c4, c5, c6 = st.columns(3)
    with c4:
        diag_delta = st.number_input("Naked Short Δ", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for the naked short leg. "
                 "Matches the Crawling Crab 'Diag Short Δ'.")
    with c5:
        spread_d_short = st.number_input("Spread Short Δ", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta of the short leg in the credit spread.")
    with c6:
        spread_d_long = st.number_input("Spread Long Δ", value=0.20,
            min_value=0.05, max_value=0.35, step=0.05, format="%.2f",
            help="Delta of the long (protective) leg in the spread.")

    step = 5 if spot_guess(symbol) > 50 else 1

    if st.button(f"Scan {'Reverse Jade' if direction == 'Bull' else 'Jade'} Lizards",
                  type="primary", use_container_width=True):
        with st.spinner("Scanning lizard configurations..."):
            try:
                result = compute(symbol, step, direction,
                                  diag_delta, spread_d_short, spread_d_long)
                result["diag_spread"] = diag_spread
                st.session_state["liz_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "liz_result" not in st.session_state:
        st.info("Set parameters and click Scan.")
        return

    display(st.session_state["liz_result"])


def spot_guess(symbol):
    s = symbol.upper().replace("^", "")
    if s in ("SPX", "GSPC", "NDX"):
        return 5000
    if s in ("SPY", "QQQ", "IWM"):
        return 400
    return 200


main()
