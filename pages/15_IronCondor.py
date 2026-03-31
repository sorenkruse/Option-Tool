"""
Iron Condor – Forecast-driven asymmetric Iron Condor builder.

Builds a skewed Iron Condor based on your directional view:
- Profit side: positioned with conviction-based delta (benefits from move)
- Risk side: short strike placed beyond the target price with buffer
- Asymmetric widths available for better risk/reward on the move side
- DTE optimized for the forecast horizon (2-3x forecast DTE)
- Diagonal Spread toggle for credit long legs
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


def project_iv(iv_atm, spot, target):
    """Project IV change based on spot move (asymmetric skew)."""
    pct = (target - spot) / spot
    if pct < 0:
        return iv_atm + abs(pct) * 0.40  # +4 IV pts per 10% drop
    else:
        return iv_atm - pct * 0.25       # -2.5 IV pts per 10% rally


# ── Build Iron Condor ────────────────────────────────────────────────────

def build_condor(spot, r, q, iv_atm, step, target_spot, forecast_dte,
                 conviction, dte, smile_call, smile_put,
                 smile_call_long, smile_put_long, long_dte,
                 profit_width_mult=1, risk_width_mult=1):
    """
    Build one Iron Condor configuration.

    Returns dict with all legs and metrics, or None.
    """
    is_bearish = target_spot < spot
    move_pct = (target_spot - spot) / spot
    T = dte / 365.0
    T_long = long_dte / 365.0

    # Conviction → delta mapping for profit side
    # Higher conviction = more aggressive (closer to ATM)
    profit_delta = 0.15 + 0.30 * conviction  # 0.15 @0% → 0.45 @100%
    profit_delta = round(profit_delta, 2)

    # Risk side: short strike placed beyond target with buffer
    # Buffer decreases with conviction (more confident = tighter)
    buffer_pct = 0.02 - 0.01 * conviction  # 2% @0% → 1% @100%
    buffer_pct = max(buffer_pct, 0.005)     # minimum 0.5%

    try:
        if is_bearish:
            # Profit side = Bear Call (above spot)
            sc_k = _round(bs.solve_strike_for_delta(
                profit_delta, spot, T, r, iv_atm, q, "call"), step)
            lc_k = sc_k + step * profit_width_mult

            # Risk side = Bull Put (below spot, beyond target)
            risk_level = target_spot * (1 - buffer_pct)
            sp_k = _round(risk_level, step)
            # Ensure SP is below target
            if sp_k >= target_spot:
                sp_k = _round(target_spot - step, step)
            lp_k = sp_k - step * risk_width_mult

        else:
            # Profit side = Bull Put (below spot)
            sp_k = _round(bs.solve_strike_for_delta(
                -profit_delta, spot, T, r, iv_atm, q, "put"), step)
            lp_k = sp_k - step * profit_width_mult

            # Risk side = Bear Call (above spot, beyond target)
            risk_level = target_spot * (1 + buffer_pct)
            sc_k = _round(risk_level, step)
            # Ensure SC is above target
            if sc_k <= target_spot:
                sc_k = _round(target_spot + step, step)
            lc_k = sc_k + step * risk_width_mult

    except Exception:
        return None

    # Validate strike ordering
    if sp_k <= lp_k or sc_k <= sp_k or lc_k <= sc_k:
        return None
    if sp_k >= spot or sc_k <= spot:
        return None

    # Price all legs
    iv_sp = _iv(smile_put, sp_k, iv_atm)
    iv_lp = _iv(smile_put_long, lp_k, iv_atm)
    iv_sc = _iv(smile_call, sc_k, iv_atm)
    iv_lc = _iv(smile_call_long, lc_k, iv_atm)

    try:
        sp = bs.calculate_all(spot, sp_k, T, r, iv_sp, q, "put")
        lp = bs.calculate_all(spot, lp_k, T_long, r, iv_lp, q, "put")
        sc = bs.calculate_all(spot, sc_k, T, r, iv_sc, q, "call")
        lc = bs.calculate_all(spot, lc_k, T_long, r, iv_lc, q, "call")
    except Exception:
        return None

    # Credits
    put_credit = sp["price"] - lp["price"]
    call_credit = sc["price"] - lc["price"]
    total_credit = put_credit + call_credit

    # Fallback: if diagonal offset makes credit negative, use same DTE
    if total_credit <= 0 and T_long != T:
        try:
            iv_lp_f = _iv(smile_put, lp_k, iv_atm)
            iv_lc_f = _iv(smile_call, lc_k, iv_atm)
            lp = bs.calculate_all(spot, lp_k, T, r, iv_lp_f, q, "put")
            lc = bs.calculate_all(spot, lc_k, T, r, iv_lc_f, q, "call")
            put_credit = sp["price"] - lp["price"]
            call_credit = sc["price"] - lc["price"]
            total_credit = put_credit + call_credit
            T_long = T
            long_dte = dte
        except Exception:
            return None

    if total_credit <= 0:
        return None

    # Max loss
    put_width = sp_k - lp_k
    call_width = lc_k - sc_k
    max_loss_put = put_width - total_credit
    max_loss_call = call_width - total_credit
    max_loss = max(max_loss_put, max_loss_call)
    if max_loss <= 0:
        return None

    # Greeks
    signs = [-1, 1, -1, 1]
    greeks = [sp, lp, sc, lc]
    delta = sum(s * g["delta"] for s, g in zip(signs, greeks))
    theta = sum(s * g["theta_daily"] for s, g in zip(signs, greeks))
    vega = sum(s * g["vega_pct"] for s, g in zip(signs, greeks))

    # P&L scenarios
    remaining_put = max(long_dte - forecast_dte, 1) / 365.0
    remaining_call = remaining_put
    target_iv = project_iv(iv_atm, spot, target_spot)
    half_spot = spot + (target_spot - spot) * 0.5
    half_iv = project_iv(iv_atm, spot, half_spot)
    flat_iv = max(iv_atm - 0.005, iv_atm * 0.98)
    wrong_spot = spot + (spot - target_spot)  # opposite move
    wrong_iv = project_iv(iv_atm, spot, wrong_spot)

    def _pnl_at_forecast(s_end, iv_end):
        """P&L after forecast_dte days."""
        T_rem_s = max(dte - forecast_dte, 1) / 365.0
        T_rem_l = max(long_dte - forecast_dte, 1) / 365.0
        try:
            sp_e = bs.calculate_all(s_end, sp_k, T_rem_s, r, iv_end, q, "put")["price"]
            lp_e = bs.calculate_all(s_end, lp_k, T_rem_l, r, iv_end, q, "put")["price"]
            sc_e = bs.calculate_all(s_end, sc_k, T_rem_s, r, iv_end, q, "call")["price"]
            lc_e = bs.calculate_all(s_end, lc_k, T_rem_l, r, iv_end, q, "call")["price"]
            close = (sp_e - lp_e) + (sc_e - lc_e)
            return total_credit - close
        except Exception:
            return 0

    pnl_full = _pnl_at_forecast(target_spot, target_iv)
    pnl_half = _pnl_at_forecast(half_spot, half_iv)
    pnl_flat = _pnl_at_forecast(spot, flat_iv)
    pnl_wrong = _pnl_at_forecast(wrong_spot, wrong_iv)

    # Weighted P&L (conviction-based)
    w_full = 0.15 + 0.45 * conviction
    w_half = 0.25 + 0.10 * conviction
    w_flat = 1.0 - w_full - w_half
    pnl_w = w_full * pnl_full + w_half * pnl_half + w_flat * pnl_flat

    # Profit zone
    be_put = sp_k - total_credit
    be_call = sc_k + total_credit
    zone_width = (be_call - be_put) / spot * 100

    # Score
    roc = pnl_w / max_loss if max_loss > 0 else 0
    days_to_50 = (total_credit * 0.50) / theta if theta > 0 else 999

    return {
        "sp_k": sp_k, "lp_k": lp_k, "sc_k": sc_k, "lc_k": lc_k,
        "sp": sp, "lp": lp, "sc": sc, "lc": lc,
        "iv_sp": iv_sp, "iv_lp": iv_lp, "iv_sc": iv_sc, "iv_lc": iv_lc,
        "put_credit": put_credit, "call_credit": call_credit,
        "total_credit": total_credit,
        "put_width": put_width, "call_width": call_width,
        "max_loss": max_loss,
        "delta": delta, "theta": theta, "vega": vega,
        "pnl_full": pnl_full, "pnl_half": pnl_half,
        "pnl_flat": pnl_flat, "pnl_wrong": pnl_wrong, "pnl_w": pnl_w,
        "be_put": be_put, "be_call": be_call, "zone_width": zone_width,
        "roc": roc, "days_to_50": days_to_50,
        "dte": dte, "long_dte": long_dte,
        "profit_delta": profit_delta,
        "profit_width": step * profit_width_mult,
        "risk_width": step * risk_width_mult,
    }


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, move_pct, forecast_dte, conviction,
            diag_spread=True, profit_width_mult=1, risk_width_mult=1):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(forecast_dte * 2)
    vix = get_vix()
    iv = vix / 100.0 if not np.isnan(vix) else 0.20
    target_spot = spot * (1 + move_pct / 100.0)

    today = datetime.date.today()
    try:
        exps, _ = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")
    if not exps:
        raise ValueError(f"No expirations for {symbol}.")

    # Load chains for DTEs from forecast*1.5 to forecast*4
    exp_dte = {}
    for e in exps:
        try:
            d = (datetime.date.fromisoformat(e) - today).days
            if d >= 3:
                exp_dte[e] = d
        except Exception:
            continue

    chain_by_exp = {}
    for exp_str, d in exp_dte.items():
        if d < max(forecast_dte, 5) or d > forecast_dte * 5 + 15:
            continue
        try:
            ch, _, _ = resolve_options_chain(symbol, d)
            actual = ch["expiration"]
            if actual not in chain_by_exp:
                chain_by_exp[actual] = ch
        except Exception:
            continue

    if not chain_by_exp:
        raise ValueError("Could not load option chains.")

    all_exps = sorted(chain_by_exp.keys())
    results = []

    for exp_str in all_exps:
        ch = chain_by_exp[exp_str]
        dte = ch["dte_actual"]
        # DTE should be 1.5x to 4x forecast
        if dte < forecast_dte * 1.5 or dte > forecast_dte * 4:
            continue

        sm_call = build_smile_curve(ch["calls"], spot)
        sm_put = build_smile_curve(ch["puts"], spot)

        # Long leg chain: +20% offset if diagonal enabled
        if diag_spread:
            long_target = round(dte * 1.2)
            cands = [(e, chain_by_exp[e]["dte_actual"])
                     for e in all_exps
                     if chain_by_exp[e]["dte_actual"] >= long_target
                     and chain_by_exp[e]["dte_actual"] > dte]
            if cands:
                long_exp = min(cands, key=lambda x: x[1])[0]
                long_ch = chain_by_exp[long_exp]
                long_dte = long_ch["dte_actual"]
                sm_call_l = build_smile_curve(long_ch["calls"], spot)
                sm_put_l = build_smile_curve(long_ch["puts"], spot)
            else:
                long_dte = dte
                sm_call_l = sm_call
                sm_put_l = sm_put
        else:
            long_dte = dte
            sm_call_l = sm_call
            sm_put_l = sm_put

        res = build_condor(spot, r, q, iv, step, target_spot, forecast_dte,
                            conviction, dte, sm_call, sm_put,
                            sm_call_l, sm_put_l, long_dte,
                            profit_width_mult, risk_width_mult)
        if res:
            res["exp_short"] = exp_str
            # Find long expiration
            if diag_spread and long_dte != dte:
                res["exp_long"] = min(
                    chain_by_exp.keys(),
                    key=lambda e: abs(chain_by_exp[e]["dte_actual"] - long_dte))
            else:
                res["exp_long"] = exp_str
            results.append(res)

    if not results:
        raise ValueError("No valid Iron Condor configurations found.")

    results.sort(key=lambda x: x["roc"], reverse=True)

    return {
        "symbol": symbol, "spot": spot, "vix": vix, "iv": iv,
        "r": r, "q": q, "move_pct": move_pct,
        "target_spot": target_spot, "forecast_dte": forecast_dte,
        "conviction": conviction, "diag_spread": diag_spread,
        "all_results": results,
    }


# ── Display ──────────────────────────────────────────────────────────────

def display(res):
    spot = res["spot"]
    target = res["target_spot"]
    is_bearish = target < spot
    direction = "Bearish" if is_bearish else "Bullish"

    st.markdown("---")
    move_str = f"{res['move_pct']:+.1f}%"
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  "
                f"Target ${target:,.0f} ({move_str}) in {res['forecast_dte']}d  |  "
                f"{direction} IC")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target", f"${target:,.0f}", f"{move_str}")
    m2.metric("IV Now", f"{res['iv']*100:.1f}%")
    iv_proj = project_iv(res['iv'], spot, target)
    iv_chg = iv_proj - res['iv']
    m3.metric("IV Projected", f"{iv_proj*100:.1f}%",
              f"{iv_chg*100:+.1f}%")
    m4.metric("Conviction", f"{res['conviction']*100:.0f}%")

    results = res["all_results"]

    # DTE scan table
    st.markdown("### DTE Optimization")
    rows = []
    for i, r in enumerate(results[:10]):
        rows.append({
            "#": i+1,
            "S DTE": r["dte"], "L DTE": r["long_dte"],
            "Credit": f"${r['total_credit']*100:,.0f}",
            "Put": f"{r['sp_k']:,.0f}/{r['lp_k']:,.0f}",
            "Call": f"{r['sc_k']:,.0f}/{r['lc_k']:,.0f}",
            "Zone": f"{r['zone_width']:.1f}%",
            "Full": f"${r['pnl_full']*100:+,.0f}",
            "Half": f"${r['pnl_half']*100:+,.0f}",
            "Flat": f"${r['pnl_flat']*100:+,.0f}",
            "Wrong": f"${r['pnl_wrong']*100:+,.0f}",
            "Weighted": f"${r['pnl_w']*100:+,.0f}",
            "ROC": f"{r['roc']*100:+.0f}%",
            "Days50": f"{r['days_to_50']:.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Select
    options = [f"#{i+1}: {r['dte']}/{r['long_dte']}d  "
               f"Credit ${r['total_credit']*100:,.0f}  "
               f"ROC {r['roc']*100:+.0f}%  "
               f"Zone {r['zone_width']:.1f}%"
               for i, r in enumerate(results[:10])]
    idx = st.selectbox("Select configuration", range(len(options)),
                        format_func=lambda i: options[i], key="ic_sel")
    pick = results[idx]

    # Position table
    st.markdown("### Position")
    exp_s = pick.get("exp_short", "")
    exp_l = pick.get("exp_long", exp_s)

    profit_label = "Profit side" if is_bearish else "Profit side"
    risk_label = "Risk side"

    put_role = risk_label if is_bearish else profit_label
    call_role = profit_label if is_bearish else risk_label

    leg_rows = [
        {"Leg": "Short Put", "Strike": f"${pick['sp_k']:,.0f}",
         "Delta": f"{pick['sp']['delta']:.3f}",
         "IV": f"{pick['iv_sp']*100:.1f}%",
         "Price": f"${pick['sp']['price']*100:,.0f}",
         "DTE": pick["dte"], "Exp": exp_s,
         "Role": f"{put_role} ({pick['put_width']:,.0f}w)"},
        {"Leg": "Long Put", "Strike": f"${pick['lp_k']:,.0f}",
         "Delta": f"{pick['lp']['delta']:.3f}",
         "IV": f"{pick['iv_lp']*100:.1f}%",
         "Price": f"${pick['lp']['price']*100:,.0f}",
         "DTE": pick["long_dte"], "Exp": exp_l,
         "Role": put_role},
        {"Leg": "Short Call", "Strike": f"${pick['sc_k']:,.0f}",
         "Delta": f"{pick['sc']['delta']:.3f}",
         "IV": f"{pick['iv_sc']*100:.1f}%",
         "Price": f"${pick['sc']['price']*100:,.0f}",
         "DTE": pick["dte"], "Exp": exp_s,
         "Role": f"{call_role} ({pick['call_width']:,.0f}w)"},
        {"Leg": "Long Call", "Strike": f"${pick['lc_k']:,.0f}",
         "Delta": f"{pick['lc']['delta']:.3f}",
         "IV": f"{pick['iv_lc']*100:.1f}%",
         "Price": f"${pick['lc']['price']*100:,.0f}",
         "DTE": pick["long_dte"], "Exp": exp_l,
         "Role": call_role},
    ]
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # Economics
    st.markdown("### Economics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Credit", f"${pick['total_credit']*100:,.0f}",
              help=f"Put: ${pick['put_credit']*100:,.0f} + "
                   f"Call: ${pick['call_credit']*100:,.0f}")
    m2.metric("Max Loss", f"${pick['max_loss']*100:,.0f}")
    m3.metric("ROC", f"{pick['roc']*100:+.0f}%",
              help="Weighted P&L / Max Loss")
    m4.metric("Days to 50%", f"{pick['days_to_50']:.0f}d")
    m5.metric("Net Delta", f"{pick['delta']:.3f}")
    m6.metric("Theta/day", f"${pick['theta']*100:+,.0f}")

    # Profit zone
    st.caption(f"Profit zone: {pick['be_put']:,.0f} - {pick['be_call']:,.0f} "
               f"({pick['zone_width']:.1f}% wide)  |  "
               f"Breakevens: "
               f"put side ${pick['be_put']:,.0f} ({(pick['be_put']/spot-1)*100:+.1f}%), "
               f"call side ${pick['be_call']:,.0f} ({(pick['be_call']/spot-1)*100:+.1f}%)")

    # Scenario P&L
    st.markdown("### Scenario P&L")
    sc_rows = [
        {"Scenario": f"Full Move ({res['move_pct']:+.1f}%)",
         "Spot": f"${target:,.0f}", "P&L": f"${pick['pnl_full']*100:+,.0f}",
         "vs Credit": f"{pick['pnl_full']/pick['total_credit']*100:+.0f}%"},
        {"Scenario": f"Half Move ({res['move_pct']/2:+.1f}%)",
         "Spot": f"${spot+(target-spot)*0.5:,.0f}",
         "P&L": f"${pick['pnl_half']*100:+,.0f}",
         "vs Credit": f"{pick['pnl_half']/pick['total_credit']*100:+.0f}%"},
        {"Scenario": "Flat",
         "Spot": f"${spot:,.0f}", "P&L": f"${pick['pnl_flat']*100:+,.0f}",
         "vs Credit": f"{pick['pnl_flat']/pick['total_credit']*100:+.0f}%"},
        {"Scenario": f"Wrong ({-res['move_pct']:+.1f}%)",
         "Spot": f"${spot+(spot-target):,.0f}",
         "P&L": f"${pick['pnl_wrong']*100:+,.0f}",
         "vs Credit": f"{pick['pnl_wrong']/pick['total_credit']*100:+.0f}%"},
    ]
    st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    # Weighted
    st.caption(f"Weighted P&L: ${pick['pnl_w']*100:+,.0f}  "
               f"(Full {(0.15+0.45*res['conviction'])*100:.0f}% / "
               f"Half {(0.25+0.10*res['conviction'])*100:.0f}% / "
               f"Flat {(1-(0.15+0.45*res['conviction'])-(0.25+0.10*res['conviction']))*100:.0f}%)")

    # Export
    st.markdown("### Export")
    legs = [
        {"strike": int(pick["sp_k"]), "option_type": "put",
         "expiration": str(exp_s), "long": False, "qty": 1},
        {"strike": int(pick["lp_k"]), "option_type": "put",
         "expiration": str(exp_l), "long": True, "qty": 1},
        {"strike": int(pick["sc_k"]), "option_type": "call",
         "expiration": str(exp_s), "long": False, "qty": 1},
        {"strike": int(pick["lc_k"]), "option_type": "call",
         "expiration": str(exp_l), "long": True, "qty": 1},
    ]
    legs_sorted = sorted(legs, key=lambda l: (l["long"], l["option_type"] != "put"))

    clean = res["symbol"].replace("^", "")
    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], legs_sorted)
        if url:
            st.markdown(f"[OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], legs_sorted, tag="IronCondor")
        st.download_button("IBKR Basket CSV", csv,
                            f"ic_{clean}.csv", "text/csv")

    # Put + Call spread links
    put_legs = [l for l in legs if l["option_type"] == "put"]
    call_legs = [l for l in legs if l["option_type"] == "call"]
    c1, c2 = st.columns(2)
    with c1:
        url_p = bs.optionstrat_url(res["symbol"], put_legs)
        if url_p:
            st.markdown(f"[Bull Put Spread]({url_p})")
    with c2:
        url_c = bs.optionstrat_url(res["symbol"], call_legs)
        if url_c:
            st.markdown(f"[Bear Call Spread]({url_c})")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Iron Condor")
    st.caption("Forecast-driven asymmetric Iron Condor. "
               "Skewed positioning based on your directional view.")

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
            help="Underlying to trade.").upper()
    with c2:
        move_pct = st.number_input("Move %", value=-2.0,
            min_value=-20.0, max_value=20.0, step=0.5, format="%.1f",
            help="Expected move. Negative = bearish. "
                 "The risk-side spread is placed beyond this target.")
    with c3:
        forecast_dte = st.number_input("Forecast DTE", value=5,
            min_value=1, max_value=30, step=1,
            help="Days until expected move. "
                 "Scanner uses 2-4x this for option DTE.")
    with c4:
        conviction = st.number_input("Conviction %", value=50,
            min_value=10, max_value=90, step=10,
            help="Confidence in the move. "
                 "High → profit side closer to ATM (more credit). "
                 "Low → profit side further OTM (safer).")
    with c5:
        default_step = 25 if "SPX" in symbol else 5
        strike_step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50)

    c6, c7, c8 = st.columns(3)
    with c6:
        diag_spread = st.toggle("Diagonal Spread", value=True,
            help="Long legs get +20% more DTE. Better IV protection.")
    with c7:
        profit_w = st.number_input("Profit Width (steps)", value=1,
            min_value=1, max_value=4,
            help="Width of profit-side spread in strike steps. "
                 "Wider = more cushion if wrong direction.")
    with c8:
        risk_w = st.number_input("Risk Width (steps)", value=1,
            min_value=1, max_value=4,
            help="Width of risk-side spread in strike steps. "
                 "Narrower = less max loss on the move side.")

    if st.button("Build Iron Condor", type="primary", use_container_width=True):
        with st.spinner("Scanning configurations..."):
            try:
                result = compute(symbol, strike_step, move_pct,
                                  forecast_dte, conviction / 100.0,
                                  diag_spread, profit_w, risk_w)
                st.session_state["ic_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "ic_result" not in st.session_state:
        st.info("Set your forecast and click Build.")
        return

    display(st.session_state["ic_result"])


def spot_guess(symbol):
    s = symbol.upper().replace("^", "")
    if s in ("SPX", "GSPC", "NDX"):
        return 5000
    return 200


main()
