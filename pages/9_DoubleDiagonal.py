"""
Double Diagonal Builder

Put Diagonal: Short Put (short DTE) + Long Put (long DTE, further OTM)
Call Diagonal: Short Call (short DTE) + Long Call (long DTE, further OTM)

Both shorts share one expiration, both longs share another.
Style: Diagonal (long strikes offset) or Calendar (same strikes).
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


def _round(strike, step):
    return round(strike / step) * step


def _iv(smile, strike, fallback):
    if smile is not None and not smile.empty:
        v = interpolate_smile_iv(smile, strike)
        if not np.isnan(v) and 0.01 < v < 2.0:
            return v
    return fallback


# ── Core builder ──────────────────────────────────────────────────────────

def build(spot, r, q, iv_atm, step, put_delta, call_delta,
          short_dte, long_dte,
          smile_put_s, smile_put_l, smile_call_s, smile_call_l,
          wing=1, iv_floor=0.15, fast=False, lc_long_dte=None,
          smile_call_lc=None):
    """
    Build a Double Diagonal (wing>=1) or Double Calendar (wing=0).

    lc_long_dte: if set, Long Call uses this DTE instead of long_dte.
                 Creates asymmetric structure (Crawling Crab Light).
    smile_call_lc: smile for LC's expiration (required if lc_long_dte set).
    """
    T_s = short_dte / 365.0
    T_l = long_dte / 365.0
    T_lc = (lc_long_dte or long_dte) / 365.0
    actual_lc_dte = lc_long_dte or long_dte

    try:
        sp_k = _round(bs.solve_strike_for_delta(put_delta, spot, T_s, r, iv_atm, q, "put"), step)
        sc_k = _round(bs.solve_strike_for_delta(call_delta, spot, T_s, r, iv_atm, q, "call"), step)
    except Exception:
        return None
    if sp_k >= sc_k:
        return None

    lp_k = sp_k - step * wing
    lc_k = sc_k + step * wing

    iv_sp = _iv(smile_put_s, sp_k, iv_atm)
    iv_lp = _iv(smile_put_l, lp_k, iv_atm)
    iv_sc = _iv(smile_call_s, sc_k, iv_atm)
    iv_lc = _iv(smile_call_lc if smile_call_lc is not None else smile_call_l, lc_k, iv_atm)

    sp = bs.calculate_all(spot, sp_k, T_s, r, iv_sp, q, "put")
    lp = bs.calculate_all(spot, lp_k, T_l, r, iv_lp, q, "put")
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, "call")
    lc = bs.calculate_all(spot, lc_k, T_lc, r, iv_lc, q, "call")

    signs = [-1, 1, -1, 1]
    greeks = [sp, lp, sc, lc]
    delta = sum(s * g["delta"] for s, g in zip(signs, greeks))
    theta = sum(s * g["theta_daily"] for s, g in zip(signs, greeks))
    vega  = sum(s * g["vega_pct"] for s, g in zip(signs, greeks))
    gamma = sum(s * g["gamma"] for s, g in zip(signs, greeks))
    cost  = sum(s * g["price"] for s, g in zip(signs, greeks))

    put_edge  = abs(sp["theta_daily"]) / max(abs(lp["theta_daily"]), 1e-4)
    call_edge = abs(sc["theta_daily"]) / max(abs(lc["theta_daily"]), 1e-4)

    # ── P&L at short expiry (current IV) ──
    remaining_lp = (long_dte - short_dte) / 365.0
    remaining_lc = (actual_lc_dte - short_dte) / 365.0
    remaining = remaining_lp  # for backward compat in _pnl_at
    if fast:
        test_spots = [spot * m for m in [0.95, 0.97, 1.0, 1.03, 1.05]]
        pnls_quick = []
        for S_t in test_spots:
            pnls_quick.append(_pnl_at(S_t, spot, sp_k, lp_k, sc_k, lc_k,
                                       remaining_lp, r, q, iv_lp, iv_lc, cost,
                                       remaining_lc))
        max_profit = max(pnls_quick)
        max_loss = min(pnls_quick)
        zone_spots = [s for s, p in zip(test_spots, pnls_quick) if p > 0]
        zone_low = min(zone_spots) if zone_spots else spot
        zone_high = max(zone_spots) if zone_spots else spot
        curve = []
    else:
        curve = []
        for S_t in np.linspace(spot * 0.90, spot * 1.10, 100):
            pnl = _pnl_at(S_t, spot, sp_k, lp_k, sc_k, lc_k,
                           remaining_lp, r, q, iv_lp, iv_lc, cost,
                           remaining_lc)
            curve.append({"spot": S_t, "pnl": pnl})
        pnls = [p["pnl"] for p in curve]
        max_profit = max(pnls)
        max_loss = min(pnls)
        prof = [p["spot"] for p in curve if p["pnl"] > 0]
        zone_low = min(prof) if prof else spot
        zone_high = max(prof) if prof else spot

    # ── Days to 50% profit (theta-based estimate) ──
    target_50 = abs(cost) * 0.50 if cost > 0 else abs(min(pnls)) * 0.50
    days_to_50 = target_50 / theta if theta > 0 else 999

    # ── P&L at IV floor (spot flat, IV drops to floor) ──
    mid_dte = max(short_dte * 0.6, 1)
    T_s_mid = mid_dte / 365.0
    T_l_mid = (long_dte - (short_dte - mid_dte)) / 365.0
    T_lc_mid = (actual_lc_dte - (short_dte - mid_dte)) / 365.0
    iv_f = max(iv_floor, 0.05)
    try:
        sp_f = bs.calculate_all(spot, sp_k, T_s_mid, r, iv_f, q, "put")["price"]
        lp_f = bs.calculate_all(spot, lp_k, T_l_mid, r, iv_f, q, "put")["price"]
        sc_f = bs.calculate_all(spot, sc_k, T_s_mid, r, iv_f, q, "call")["price"]
        lc_f = bs.calculate_all(spot, lc_k, T_lc_mid, r, iv_f, q, "call")["price"]
        pnl_at_floor = (-sp_f + lp_f - sc_f + lc_f) - cost
    except Exception:
        pnl_at_floor = -cost

    return {
        "sp_k": sp_k, "lp_k": lp_k, "sc_k": sc_k, "lc_k": lc_k,
        "sp": sp, "lp": lp, "sc": sc, "lc": lc,
        "iv_sp": iv_sp, "iv_lp": iv_lp, "iv_sc": iv_sc, "iv_lc": iv_lc,
        "delta": delta, "theta": theta, "vega": vega, "gamma": gamma,
        "cost": cost, "put_edge": put_edge, "call_edge": call_edge,
        "max_profit": max_profit, "max_loss": max_loss,
        "zone_low": zone_low, "zone_high": zone_high,
        "pnl_curve": curve,
        "short_dte": short_dte, "long_dte": long_dte,
        "lc_long_dte": actual_lc_dte,
        "days_to_50": days_to_50, "target_50": target_50,
        "pnl_at_floor": pnl_at_floor, "iv_floor": iv_floor,
    }


def _pnl_at(S_t, spot, sp_k, lp_k, sc_k, lc_k, remaining, r, q,
            iv_lp, iv_lc, cost, remaining_lc=None):
    """P&L at short expiry for a given spot."""
    sp_v = -max(sp_k - S_t, 0)
    sc_v = -max(S_t - sc_k, 0)
    iv_p = iv_lp * (1 + 0.4 * max(0, (spot - S_t) / spot))
    iv_c = iv_lc * (1 + 0.4 * max(0, (S_t - spot) / spot))
    T_lp = remaining
    T_lc = remaining_lc if remaining_lc is not None else remaining
    try:
        lp_v = bs.calculate_all(S_t, lp_k, T_lp, r, iv_p, q, "put")["price"]
    except Exception:
        lp_v = max(lp_k - S_t, 0)
    try:
        lc_v = bs.calculate_all(S_t, lc_k, T_lc, r, iv_c, q, "call")["price"]
    except Exception:
        lc_v = max(S_t - lc_k, 0)
    return sp_v + lp_v + sc_v + lc_v - cost


def score(res, spot):
    """
    Score optimizing for:
    1. Speed to 50% profit (days_to_50 → lower = better)
    2. Profit zone width (wider = safer)
    3. IV-floor resilience (pnl_at_floor must be > 0, or penalized)
    4. Delta neutrality
    """
    if res is None:
        return -9999

    d50 = res.get("days_to_50", 999)
    zw = res["zone_high"] - res["zone_low"]
    pnl_floor = res.get("pnl_at_floor", -999)
    delta = abs(res["delta"])
    cost = res["cost"]
    short_dte = res["short_dte"]

    # Speed: inverse of days_to_50, capped. Best if < 5 days.
    if d50 <= 0 or d50 > short_dte:
        speed = 0.01
    else:
        speed = 1.0 / max(d50, 1)

    # Zone width as % of spot
    zone_pct = zw / spot * 100

    # IV-floor resilience: bonus if profitable, harsh penalty if not
    if pnl_floor > 0:
        iv_factor = 1.0 + min(pnl_floor / max(cost, 1), 1.0)
    else:
        iv_factor = max(0.1, 1.0 + pnl_floor / max(cost, 1))

    # Delta penalty
    delta_factor = max(0, 1.0 - delta * 5)

    return speed * zone_pct * iv_factor * delta_factor * 10000


# ── Compute ───────────────────────────────────────────────────────────────

def compute(symbol, step, mode, short_dte_in, long_dte_in,
            put_delta, call_delta, bias_pct, wing, iv_floor=0.15,
            asym_lc=False):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(short_dte_in if short_dte_in > 0 else 10)
    vix = get_vix()
    iv = vix / 100.0 if not np.isnan(vix) else 0.20
    bias_spot = spot * (1 + bias_pct / 100.0)
    auto = mode != "Manual DTEs"
    today = datetime.date.today()

    if auto:
        try:
            exps, ct = get_available_expirations(symbol)
        except Exception as e:
            raise ValueError(f"Could not fetch expirations: {e}")
        if not exps:
            raise ValueError(f"No expirations for {symbol}. Try Manual DTEs.")

        # Map expiration → DTE, filter range (wider when asym_lc enabled)
        max_dte_range = 90 if asym_lc else 60
        exp_dte_map = {}
        for e in exps:
            try:
                d = (datetime.date.fromisoformat(e) - today).days
                if 5 <= d <= max_dte_range:
                    exp_dte_map[e] = d
            except Exception:
                continue
        if not exp_dte_map:
            raise ValueError(f"No expirations in 5-{max_dte_range} DTE range.")

        # Load each expiration chain ONCE
        chain_by_exp = {}
        for exp_str in exp_dte_map:
            try:
                ch, _, _ = resolve_options_chain(symbol, exp_dte_map[exp_str])
                # Cache by actual expiration to avoid dupes
                actual_exp = ch["expiration"]
                if actual_exp not in chain_by_exp:
                    chain_by_exp[actual_exp] = ch
            except Exception:
                continue

        if not chain_by_exp:
            raise ValueError("Could not load any option chains.")

        # Build unique (short_exp, long_exp) combos
        best = None; best_sc = -9999; combos = []
        all_exps = sorted(chain_by_exp.keys())

        for s_exp in all_exps:
            cs = chain_by_exp[s_exp]
            s_dte = cs["dte_actual"]
            if s_dte < 5 or s_dte > 14:
                continue
            sm_ps = build_smile_curve(cs["puts"], spot)
            sm_cs = build_smile_curve(cs["calls"], spot)

            for l_exp in all_exps:
                cl = chain_by_exp[l_exp]
                l_dte = cl["dte_actual"]
                if l_dte <= s_dte:
                    continue
                ratio = l_dte / s_dte
                if ratio < 1.5 or ratio > 4.0:
                    continue

                sm_pl = build_smile_curve(cl["puts"], spot)
                sm_cl = build_smile_curve(cl["calls"], spot)

                # Asymmetric LC DTE: find chain at ~1.5× long DTE
                lc_dte_arg = None
                sm_cl_lc = None
                exp_lc = l_exp
                if asym_lc:
                    lc_target = round(l_dte * 1.5)
                    cands = [(e, chain_by_exp[e]["dte_actual"])
                             for e in all_exps
                             if chain_by_exp[e]["dte_actual"] >= lc_target]
                    if cands:
                        best_lc_exp = min(cands, key=lambda x: x[1])
                        lc_chain = chain_by_exp[best_lc_exp[0]]
                        lc_dte_arg = lc_chain["dte_actual"]
                        sm_cl_lc = build_smile_curve(lc_chain["calls"], spot)
                        exp_lc = best_lc_exp[0]

                res = build(bias_spot, r, q, iv, step,
                            put_delta, call_delta,
                            s_dte, l_dte,
                            sm_ps, sm_pl, sm_cs, sm_cl, wing,
                            iv_floor, fast=True,
                            lc_long_dte=lc_dte_arg,
                            smile_call_lc=sm_cl_lc)
                if res is None:
                    continue

                sc_val = score(res, spot)
                res["exp_short"] = s_exp
                res["exp_long"]  = l_exp
                res["exp_lc"]    = exp_lc
                res["short_dte"] = s_dte
                res["long_dte"]  = l_dte
                res["symbol"] = symbol; res["spot"] = spot
                res["vix"] = vix; res["r"] = r; res["q"] = q

                combos.append({
                    "short": s_dte, "long": l_dte,
                    "ratio": f"{l_dte/s_dte:.1f}x",
                    "theta": res["theta"], "delta": res["delta"],
                    "vega": res["vega"], "cost": res["cost"],
                    "zone": res["zone_high"] - res["zone_low"],
                    "d50": res["days_to_50"],
                    "floor_pnl": res["pnl_at_floor"],
                    "score": sc_val,
                    "_s_exp": s_exp, "_l_exp": l_exp,
                })
                if sc_val > best_sc:
                    best_sc = sc_val
                    best = dict(res)

        if best is None:
            raise ValueError("No valid configuration found. Try Manual DTEs.")
        combos.sort(key=lambda x: x["score"], reverse=True)

        # Rebuild top 10 with full P&L curves (chains already loaded)
        for combo in combos[:10]:
            cs_c = chain_by_exp[combo["_s_exp"]]
            cl_c = chain_by_exp[combo["_l_exp"]]
            sm_ps_c = build_smile_curve(cs_c["puts"], spot)
            sm_pl_c = build_smile_curve(cl_c["puts"], spot)
            sm_cs_c = build_smile_curve(cs_c["calls"], spot)
            sm_cl_c = build_smile_curve(cl_c["calls"], spot)

            # Asym LC for full rebuild
            lc_dte_c = None; sm_cl_lc_c = None; exp_lc_c = combo["_l_exp"]
            if asym_lc:
                lc_target = round(cl_c["dte_actual"] * 1.5)
                cands = [(e, chain_by_exp[e]["dte_actual"])
                         for e in all_exps
                         if chain_by_exp[e]["dte_actual"] >= lc_target]
                if cands:
                    be = min(cands, key=lambda x: x[1])
                    lc_dte_c = chain_by_exp[be[0]]["dte_actual"]
                    sm_cl_lc_c = build_smile_curve(chain_by_exp[be[0]]["calls"], spot)
                    exp_lc_c = be[0]

            full = build(bias_spot, r, q, iv, step,
                         put_delta, call_delta,
                         cs_c["dte_actual"], cl_c["dte_actual"],
                         sm_ps_c, sm_pl_c, sm_cs_c, sm_cl_c, wing,
                         iv_floor, fast=False,
                         lc_long_dte=lc_dte_c,
                         smile_call_lc=sm_cl_lc_c)
            if full:
                full["exp_short"] = combo["_s_exp"]
                full["exp_long"] = combo["_l_exp"]
                full["exp_lc"] = exp_lc_c
                full["short_dte"] = cs_c["dte_actual"]
                full["long_dte"] = cl_c["dte_actual"]
                full["symbol"] = symbol; full["spot"] = spot
                full["vix"] = vix; full["r"] = r; full["q"] = q
                combo["_full"] = full
            else:
                combo["_full"] = best

        best = combos[0]["_full"]
        best["dte_scan"] = combos[:10]

    else:
        cs, _, _ = resolve_options_chain(symbol, short_dte_in)
        cl, _, _ = resolve_options_chain(symbol, long_dte_in)
        sm_ps = build_smile_curve(cs["puts"], spot)
        sm_pl = build_smile_curve(cl["puts"], spot)
        sm_cs = build_smile_curve(cs["calls"], spot)
        sm_cl = build_smile_curve(cl["calls"], spot)

        lc_dte_m = None; sm_cl_lc_m = None; exp_lc_m = cl["expiration"]
        if asym_lc:
            lc_target = round(cl["dte_actual"] * 1.5)
            try:
                cl_lc, _, _ = resolve_options_chain(symbol, lc_target)
                lc_dte_m = cl_lc["dte_actual"]
                sm_cl_lc_m = build_smile_curve(cl_lc["calls"], spot)
                exp_lc_m = cl_lc["expiration"]
            except Exception:
                pass  # fallback to same DTE

        best = build(bias_spot, r, q, iv, step,
                     put_delta, call_delta,
                     cs["dte_actual"], cl["dte_actual"],
                     sm_ps, sm_pl, sm_cs, sm_cl, wing,
                     iv_floor,
                     lc_long_dte=lc_dte_m,
                     smile_call_lc=sm_cl_lc_m)
        if best is None:
            raise ValueError("Could not construct position.")
        best["exp_short"] = cs["expiration"]
        best["exp_long"]  = cl["expiration"]
        best["exp_lc"]    = exp_lc_m
        best["short_dte"] = cs["dte_actual"]
        best["long_dte"]  = cl["dte_actual"]
        best["dte_scan"]  = None

    best["symbol"] = symbol; best["spot"] = spot
    best["vix"] = vix; best["r"] = r; best["q"] = q
    return best


# ── Display ───────────────────────────────────────────────────────────────

def display(res):
    spot = res["spot"]
    # Header only if not already shown by main (no dte_scan = manual mode)
    if res.get("dte_scan") is not None or "exp_short" not in res:
        pass  # header already shown
    else:
        st.markdown("---")
        st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")

    # Structure
    st.markdown("### Structure")
    st.dataframe(pd.DataFrame([
        {"Leg": "Short Put",  "Strike": f"{res['sp_k']:,.0f}",
         "Delta": f"{res['sp']['delta']:.3f}", "IV": f"{res['iv_sp']*100:.1f}%",
         "Price": f"${res['sp']['price']:.2f}", "Qty": "-1",
         "DTE": res["short_dte"], "Exp": res["exp_short"]},
        {"Leg": "Long Put",   "Strike": f"{res['lp_k']:,.0f}",
         "Delta": f"{res['lp']['delta']:.3f}", "IV": f"{res['iv_lp']*100:.1f}%",
         "Price": f"${res['lp']['price']:.2f}", "Qty": "+1",
         "DTE": res["long_dte"], "Exp": res["exp_long"]},
        {"Leg": "Short Call", "Strike": f"{res['sc_k']:,.0f}",
         "Delta": f"{res['sc']['delta']:.3f}", "IV": f"{res['iv_sc']*100:.1f}%",
         "Price": f"${res['sc']['price']:.2f}", "Qty": "-1",
         "DTE": res["short_dte"], "Exp": res["exp_short"]},
        {"Leg": "Long Call",  "Strike": f"{res['lc_k']:,.0f}",
         "Delta": f"{res['lc']['delta']:.3f}", "IV": f"{res['iv_lc']*100:.1f}%",
         "Price": f"${res['lc']['price']:.2f}", "Qty": "+1",
         "DTE": res.get("lc_long_dte", res["long_dte"]),
         "Exp": res.get("exp_lc", res["exp_long"])},
    ]), use_container_width=True, hide_index=True)

    # Greeks
    st.markdown("### Combined Position")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Entry Cost", f"${res['cost']:.2f}")
    m2.metric("Delta",      f"{res['delta']:.3f}")
    m3.metric("Theta/day",  f"${res['theta']:.2f}")
    m4.metric("Vega/1%",    f"${res['vega']:.2f}")
    d50 = res.get("days_to_50", 999)
    m5.metric("Days to 50%", f"{d50:.0f}d" if d50 < 900 else "n/a")
    pnl_f = res.get("pnl_at_floor", 0)
    iv_f = res.get("iv_floor", 0.15)
    m6.metric(f"P&L @ IV {iv_f*100:.0f}%",
              f"${pnl_f:.1f}",
              delta="safe" if pnl_f > 0 else "at risk",
              delta_color="normal" if pnl_f > 0 else "inverse")
    t50 = res.get("target_50", 0)
    st.caption(
        f"Per SPX contract (x100): Cost ${res['cost']*100:,.0f}, "
        f"Theta ${res['theta']*100:,.0f}/day, "
        f"50% target ${t50*100:,.0f}, "
        f"Max Profit ${res['max_profit']*100:,.0f}, "
        f"Max Loss ${res['max_loss']*100:,.0f}")

    # Chart
    st.markdown("### P&L at Short Expiry")
    import plotly.graph_objects as go
    spots = [p["spot"] for p in res["pnl_curve"]]
    pnls  = [p["pnl"]  for p in res["pnl_curve"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=[max(0,p) for p in pnls],
        fill="tozeroy", fillcolor="rgba(50,180,80,0.15)",
        line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=spots, y=pnls, mode="lines",
        line=dict(color="#1f77b4", width=2), showlegend=False))
    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot {spot:,.0f}")
    fig.add_vline(x=res["sp_k"], line_dash="dot", line_color="red",
                  annotation_text=f"SP {res['sp_k']:,.0f}",
                  annotation_position="bottom right")
    fig.add_vline(x=res["sc_k"], line_dash="dot", line_color="red",
                  annotation_text=f"SC {res['sc_k']:,.0f}",
                  annotation_position="bottom left")
    if res["lp_k"] != res["sp_k"]:
        fig.add_vline(x=res["lp_k"], line_dash="dot", line_color="green",
                      annotation_text=f"LP {res['lp_k']:,.0f}",
                      annotation_position="top right")
    if res["lc_k"] != res["sc_k"]:
        fig.add_vline(x=res["lc_k"], line_dash="dot", line_color="green",
                      annotation_text=f"LC {res['lc_k']:,.0f}",
                      annotation_position="top left")
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    fig.update_layout(template="plotly_white", height=400,
        xaxis_title="Spot at Short Expiry", yaxis_title="P&L (per point)",
        margin=dict(l=50,r=20,t=30,b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Profit zone: {res['zone_low']:,.0f} - {res['zone_high']:,.0f} "
        f"({(res['zone_high']-res['zone_low'])/spot*100:.1f}% width)")

    # Export
    legs = [
        {"strike": res["sp_k"], "option_type": "put",
         "expiration": res["exp_short"], "long": False, "qty": 1,
         "price": res["sp"]["price"]},
        {"strike": res["sc_k"], "option_type": "call",
         "expiration": res["exp_short"], "long": False, "qty": 1,
         "price": res["sc"]["price"]},
        {"strike": res["lp_k"], "option_type": "put",
         "expiration": res["exp_long"], "long": True, "qty": 1,
         "price": res["lp"]["price"]},
        {"strike": res["lc_k"], "option_type": "call",
         "expiration": res.get("exp_lc", res["exp_long"]), "long": True, "qty": 1,
         "price": res["lc"]["price"]},
    ]
    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], legs)
        if url:
            st.markdown(f"[OptionStrat]({url})")
    with e2:
        st.download_button("IBKR Basket CSV",
            bs.ibkr_basket_csv(res["symbol"], legs, tag="DoubleDiag"),
            f"ddiag_{res['symbol']}_{res['exp_short']}.csv", "text/csv")

    # Management
    with st.expander("Management Guidelines"):
        t50 = res.get("target_50", 0)
        d50 = res.get("days_to_50", 999)
        pnl_f = res.get("pnl_at_floor", 0)
        iv_f = res.get("iv_floor", 0.15)
        st.markdown(
            f"- **Profit target**: 50% of debit = ${t50*100:,.0f} per contract\n"
            f"- **Expected days**: ~{d50:.0f} days (theta-based)\n"
            f"- **Close**: At 50% profit, or 3-4 days before short expiry\n"
            f"- **IV Floor** (IV={iv_f*100:.0f}%): "
            f"{'Profitable' if pnl_f > 0 else 'At risk'} "
            f"(${pnl_f*100:+,.0f}/contract)\n"
            f"- **Put side**: Self-correcting on drops\n"
            f"- **Call side**: Roll short call if spot > {res['sc_k']:,.0f}\n"
            f"- **IV environment**: Best when VIX > {iv_f*100:.0f} "
            f"(currently {res['vix']:.1f})")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Double Diagonal Builder")
    st.caption("Asymmetric Put + Call Diagonal. "
               "Wide DTE spread for theta edge and crash protection.")

    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="SPX").upper()
    with c2:
        style = st.selectbox("Style", ["Diagonal", "Calendar"],
            help="Diagonal: long strikes offset further OTM. "
                 "Calendar: same strikes (higher vega risk).")
    with c3:
        default_step = 25 if "SPX" in symbol else 5
        step = st.number_input("Strike Step", value=default_step,
                                min_value=1, max_value=50)
    with c4:
        mode = st.radio("Mode", ["Auto", "Manual DTEs"],
                         horizontal=True)
    with c5:
        asym_lc = st.toggle("Asym LC DTE", value=False,
            help="Give the Long Call a longer DTE (1.5× the Long Put DTE). "
                 "Adds extra vega on the call side to compensate for "
                 "IV drops during rallies. Turns DD into a 'Crawling Crab Light'. "
                 "The LC can be reused for multiple short cycles.")

    if mode == "Manual DTEs":
        mc1, mc2 = st.columns(2)
        with mc1:
            short_dte = st.number_input("Short DTE", value=9,
                min_value=5, max_value=21, help="DTE for both short legs.")
        with mc2:
            long_dte = st.number_input("Long DTE", value=30,
                min_value=14, max_value=60, help="DTE for both long legs.")
    else:
        short_dte = long_dte = 0

    st.markdown("#### Positioning")
    wing = 0
    if style == "Diagonal":
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        with d6:
            wing = st.number_input("Wing (steps)", value=1,
                min_value=1, max_value=5,
                help="Strike steps to offset longs from shorts.")
    else:
        d1, d2, d3, d4, d5 = st.columns(5)

    with d1:
        put_delta = st.number_input("Put Delta", value=-0.30,
            min_value=-0.50, max_value=-0.10, step=0.05, format="%.2f")
    with d2:
        call_delta = st.number_input("Call Delta", value=0.30,
            min_value=0.10, max_value=0.50, step=0.05, format="%.2f")
    with d3:
        bias = st.number_input("Bias %", value=0.0,
            min_value=-5.0, max_value=5.0, step=0.5, format="%.1f",
            help="Negative = bearish, positive = bullish.")
    with d4:
        iv_floor = st.number_input("IV Floor %", value=15, min_value=5,
            max_value=40, step=1,
            help="Min IV for resilience test. SPX ~15, QQQ ~18.")
    with d5:
        profit_target = st.number_input("Target %", value=50, min_value=10,
            max_value=100, step=10,
            help="Profit target as % of debit. Standard: 50%.")

    if st.button("Build Double Diagonal", type="primary",
                  use_container_width=True):
        with st.spinner("Computing..."):
            try:
                r = compute(symbol, step, mode, short_dte, long_dte,
                            put_delta, call_delta, bias, wing,
                            iv_floor / 100.0, asym_lc)
                st.session_state["dd_result"] = r
                st.session_state["dd_selected"] = 0
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "dd_result" not in st.session_state:
        st.info("Set parameters and click Build.")
        return

    result = st.session_state["dd_result"]
    scan = result.get("dte_scan")

    # Show DTE scan table and selector before display
    if scan and len(scan) > 1:
        st.markdown("---")
        st.markdown(f"### {result['symbol']} @ {result['spot']:,.2f}  |  "
                    f"VIX {result['vix']:.1f}")
        st.markdown("### DTE Optimization")
        st.caption("Scored by: speed to 50%, IV-floor resilience, "
                   "profit zone width, delta neutrality.")
        rows = []
        for i, s in enumerate(scan):
            rows.append({
                "#": i + 1, "Short": s["short"], "Long": s["long"],
                "Ratio": s["ratio"],
                "Theta/d": f"${s['theta']:.2f}",
                "Delta": f"{s['delta']:.3f}",
                "Cost": f"${s['cost']:.2f}",
                "Zone": f"{s['zone']:.0f} pts",
                "Days50": f"{s['d50']:.0f}",
                "IV Floor": f"${s['floor_pnl']:.1f}",
                "Score": f"{s['score']:.0f}",
            })
        st.dataframe(pd.DataFrame(rows),
                      use_container_width=True, hide_index=True)

        options = [f"#{i+1}: {s['short']}/{s['long']}d "
                   f"(Score {s['score']:.0f}, Days50 {s['d50']:.0f})"
                   for i, s in enumerate(scan)]
        idx = st.selectbox("Result", range(len(options)),
                            format_func=lambda i: options[i],
                            key="dd_sel")
        active = scan[idx]["_full"]
    else:
        active = result

    display(active)


main()
