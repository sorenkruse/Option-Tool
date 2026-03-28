"""
Crawling Crab – Stock Replacement + Credit Spread income.

Bull Crab: Long Call LEAPS + Short Call monthly + Bull Put Spread (short DTE)
Bear Crab: Long Put LEAPS + Short Put monthly + Bear Call Spread (short DTE)

Same as Stock Replacement but with an additional credit spread that
increases per-cycle income and accelerates LEAPS financing.
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

def scan(spot, r, q, iv_atm, step, direction, target_pct, target_dte,
         min_long_dte,
         expirations_data, sc_dte_actual,
         sc_smile_call, sc_smile_put,
         sc_delta, sp_delta_short, sp_delta_long):
    """Scan LEAPS + short diagonal + credit spread configurations."""
    results = []
    is_bull = direction == "Bull"
    target_price = spot * (1 + target_pct / 100) if is_bull else spot * (1 - abs(target_pct) / 100)
    T_s = sc_dte_actual / 365.0

    # ── Short-DTE legs (fixed across all LEAPS scans) ──

    # Diagonal short (same type as LEAPS)
    diag_type = "call" if is_bull else "put"
    try:
        if is_bull:
            sc_k = _round(bs.solve_strike_for_delta(sc_delta, spot, T_s, r, iv_atm, q, "call"), step)
        else:
            sc_k = _round(bs.solve_strike_for_delta(-sc_delta, spot, T_s, r, iv_atm, q, "put"), step)
    except Exception:
        return results, target_price
    iv_sc = _iv(sc_smile_call if is_bull else sc_smile_put, sc_k, iv_atm)
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, diag_type)

    # Credit spread (opposite type: puts for bull, calls for bear)
    # SP and LP use same DTE; diagonal offset applied post-scan if enabled
    spread_type = "put" if is_bull else "call"
    spread_smile = sc_smile_put if is_bull else sc_smile_call
    try:
        if is_bull:
            sp_k = _round(bs.solve_strike_for_delta(-sp_delta_short, spot, T_s, r, iv_atm, q, "put"), step)
            lp_k = _round(bs.solve_strike_for_delta(-sp_delta_long, spot, T_s, r, iv_atm, q, "put"), step)
            if sp_k <= lp_k:
                return results, target_price
        else:
            sp_k = _round(bs.solve_strike_for_delta(sp_delta_short, spot, T_s, r, iv_atm, q, "call"), step)
            lp_k = _round(bs.solve_strike_for_delta(sp_delta_long, spot, T_s, r, iv_atm, q, "call"), step)
            if lp_k <= sp_k:
                return results, target_price
    except Exception:
        return results, target_price

    iv_sp = _iv(spread_smile, sp_k, iv_atm)
    iv_lp = _iv(spread_smile, lp_k, iv_atm)
    sp = bs.calculate_all(spot, sp_k, T_s, r, iv_sp, q, spread_type)
    lp = bs.calculate_all(spot, lp_k, T_s, r, iv_lp, q, spread_type)
    spread_credit = sp["price"] - lp["price"]

    if spread_credit <= 0:
        return results, target_price

    # Cycle credit from all 3 short-DTE legs: SC + (SP - LP)
    cycle_credit = sc["price"] + spread_credit
    cycle_income = cycle_credit * 0.50

    # Theta from short-DTE legs only
    theta_short = (-sc["theta_daily"] + sp["theta_daily"] - lp["theta_daily"])
    days_to_50 = cycle_income / theta_short if theta_short > 0 else 30

    # ── Scan LEAPS ──

    for exp_str, chain_data in expirations_data.items():
        dte = chain_data["dte_actual"]
        if dte < max(60, min_long_dte) or dte > 548:
            continue
        if dte < target_dte + 30:
            continue
        T_l = dte / 365.0
        smile = build_smile_curve(
            chain_data["calls"] if is_bull else chain_data["puts"], spot)

        for delta_target in [0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.75, 0.70]:
            try:
                if is_bull:
                    lc_k = _round(bs.solve_strike_for_delta(
                        delta_target, spot, T_l, r, iv_atm, q, "call"), step)
                    if sc_k <= lc_k:
                        continue
                else:
                    lc_k = _round(bs.solve_strike_for_delta(
                        -delta_target, spot, T_l, r, iv_atm, q, "put"), step)
                    if sc_k >= lc_k:
                        continue
            except Exception:
                continue

            iv_lc = _iv(smile, lc_k, iv_atm)
            lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, diag_type)
            if lc["price"] < 0.50:
                continue

            intrinsic = max(spot - lc_k, 0) if is_bull else max(lc_k - spot, 0)
            extrinsic = lc["price"] - intrinsic
            extrinsic_pct = extrinsic / lc["price"] * 100 if lc["price"] > 0 else 0

            cycles = max(1, int((dte - 45) / sc_dte_actual))
            total_income = cycle_income * cycles
            net = lc["price"] - total_income

            if is_bull:
                be = lc_k + net
                pnl_target = max(target_price - lc_k, 0) - lc["price"] + total_income
            else:
                be = lc_k - net
                pnl_target = max(lc_k - target_price, 0) - lc["price"] + total_income

            if pnl_target <= 0:
                continue

            roc = pnl_target / lc["price"] * 100 if lc["price"] > 0 else 0
            leverage = abs(lc["delta"]) * spot / lc["price"] if lc["price"] > 0 else 0

            # Annualized theta cost of LEAPS
            hold_days = max(dte - 45, 30)
            rolls_per_year = 365 / hold_days
            try:
                lc_exit = bs.calculate_all(spot, lc_k, 45/365, r, iv_lc, q, diag_type)
                roll_loss = lc["price"] - lc_exit["price"]
            except Exception:
                roll_loss = lc["price"] * 0.5
            annual_cost = roll_loss * rolls_per_year

            # Cycles to fund & DTE after funding
            cycles_to_fund = lc["price"] / cycle_income if cycle_income > 0 else 99
            days_needed = cycles_to_fund * days_to_50
            remaining_after = dte - days_needed

            # Score
            be_margin = (spot - be) / spot * 100 if is_bull else (be - spot) / spot * 100
            cap_eff = min(extrinsic_pct / 50, 2.0)
            theta_per_delta = annual_cost / max(abs(lc["delta"]), 0.1)
            theta_eff = max(0.2, 1.0 - theta_per_delta / (spot * 0.5))
            if dte < 120:
                dte_eff = 0.3 + 0.7 * (dte / 120)
            elif dte <= 400:
                dte_eff = 1.0
            else:
                dte_eff = max(0.5, 1.0 - (dte - 400) / 400)
            # Bonus for faster funding
            fund_eff = max(0.3, 1.0 - cycles_to_fund / 20)
            score = roc * max(0.1, 1 + be_margin / 10) * cap_eff * theta_eff * dte_eff * fund_eff

            results.append({
                "lc_k": lc_k, "sc_k": sc_k, "sp_k": sp_k, "lp_k": lp_k,
                "leaps_dte": dte, "sc_dte": sc_dte_actual,
                "delta": lc["delta"],
                "leaps_cost": lc["price"],
                "extrinsic_pct": extrinsic_pct,
                "leverage": leverage,
                "annual_cost": annual_cost,
                "sc_premium": sc["price"],
                "spread_credit": spread_credit,
                "cycle_credit": cycle_credit,
                "cycle_income": cycle_income,
                "cycles": cycles,
                "total_income": total_income,
                "net_cost": net,
                "breakeven": be,
                "be_margin": be_margin,
                "pnl_target": pnl_target,
                "roc": roc, "score": score,
                "cycles_to_fund": cycles_to_fund,
                "days_to_50": days_to_50,
                "remaining_after": remaining_after,
                "exp": exp_str, "iv_lc": iv_lc, "opt_type": diag_type,
                # Store short leg details for display
                "sc_res": sc, "sp_res": sp, "lp_res": lp,
                "iv_sc": iv_sc, "iv_sp": iv_sp, "iv_lp": iv_lp,
                "spread_type": spread_type,
            })

    return results, target_price


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, direction, target_pct, target_dte,
            min_long_dte, sc_delta, sp_d_short, sp_d_long, sc_dte_target,
            auto_mode=False):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(90)
    vix = get_vix()
    iv = vix / 100.0 if not np.isnan(vix) else 0.30
    today = datetime.date.today()

    try:
        exps, _ = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")
    if not exps:
        raise ValueError(f"No expirations for {symbol}.")

    chain_by_exp = {}
    for exp_str in exps:
        try:
            d = (datetime.date.fromisoformat(exp_str) - today).days
            if d >= 7:
                ch, _, _ = resolve_options_chain(symbol, d)
                actual_exp = ch["expiration"]
                if actual_exp not in chain_by_exp:
                    chain_by_exp[actual_exp] = ch
        except Exception:
            continue
    if not chain_by_exp:
        raise ValueError("Could not load option chains.")

    leaps_data = {e: c for e, c in chain_by_exp.items() if c["dte_actual"] >= 60}

    if auto_mode:
        # Scan across multiple short DTEs
        all_results = []
        short_dtes_seen = set()
        for s_exp, s_chain in chain_by_exp.items():
            s_dte = s_chain["dte_actual"]
            if s_dte < 10 or s_dte > 50 or s_dte in short_dtes_seen:
                continue
            short_dtes_seen.add(s_dte)

            sm_call = build_smile_curve(s_chain["calls"], spot)
            sm_put = build_smile_curve(s_chain["puts"], spot)

            results, target_price = scan(
                spot, r, q, iv, step, direction, target_pct, target_dte,
                min_long_dte, leaps_data, s_dte,
                sm_call, sm_put,
                sc_delta, sp_d_short, sp_d_long)
            all_results.extend(results)

        if not all_results:
            raise ValueError("No valid configurations found.")
        all_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "symbol": symbol, "spot": spot, "vix": vix, "iv": iv,
            "r": r, "q": q, "direction": direction,
            "target_price": target_price, "target_pct": target_pct,
            "target_dte": target_dte,
            "sc_dte_actual": all_results[0]["sc_dte"],
            "sc_exp": all_results[0]["exp"],
            "all_results": all_results,
            "chain_by_exp": chain_by_exp,
        }
    else:
        sc_exp = min(chain_by_exp.keys(),
                      key=lambda e: abs(chain_by_exp[e]["dte_actual"] - sc_dte_target))
        sc_chain = chain_by_exp[sc_exp]
        sc_smile_call = build_smile_curve(sc_chain["calls"], spot)
        sc_smile_put = build_smile_curve(sc_chain["puts"], spot)

        results, target_price = scan(
            spot, r, q, iv, step, direction, target_pct, target_dte,
            min_long_dte, leaps_data, sc_chain["dte_actual"],
            sc_smile_call, sc_smile_put,
            sc_delta, sp_d_short, sp_d_long)

        if not results:
            raise ValueError("No valid configurations found.")
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "symbol": symbol, "spot": spot, "vix": vix, "iv": iv,
            "r": r, "q": q, "direction": direction,
            "target_price": target_price, "target_pct": target_pct,
            "target_dte": target_dte,
            "sc_dte_actual": sc_chain["dte_actual"], "sc_exp": sc_exp,
            "all_results": results,
            "chain_by_exp": chain_by_exp,
        }


# ── Display ──────────────────────────────────────────────────────────────

SCAN_CFG = {
    "LEAPS": st.column_config.TextColumn("LEAPS", help="LEAPS strike"),
    "SC": st.column_config.TextColumn("SC", help="Short diagonal strike"),
    "Spread": st.column_config.TextColumn("Spread", help="Credit spread strikes (short/long)"),
    "L DTE": st.column_config.NumberColumn("L DTE", help="LEAPS days to expiration"),
    "S DTE": st.column_config.NumberColumn("S DTE", help="Short-DTE legs expiration"),
    "Delta": st.column_config.TextColumn("Delta", help="LEAPS delta"),
    "Cost": st.column_config.TextColumn("Cost", help="LEAPS premium (×100)"),
    "Cyc Cred": st.column_config.TextColumn("Cyc Cred",
        help="Per-cycle credit: SC premium + spread credit, at 50% (×100)"),
    "Cyc": st.column_config.NumberColumn("Cyc", help="Cycles possible"),
    "CycFund": st.column_config.TextColumn("CycFund",
        help="Cycles needed to fully finance the LEAPS"),
    "WksFund": st.column_config.TextColumn("WksFund",
        help="Weeks to fully fund = cycles × days to 50% / 7"),
    "BE": st.column_config.TextColumn("BE", help="Break-even at expiry"),
    "P&L": st.column_config.TextColumn("P&L", help="Profit at target (×100)"),
    "ROC": st.column_config.TextColumn("ROC", help="Return on Capital at target"),
}


def display(res):
    spot = res["spot"]
    target = res["target_price"]
    direction = res["direction"]
    is_bull = direction == "Bull"
    name = "Bull Crab" if is_bull else "Bear Crab"
    opt_label = "Call" if is_bull else "Put"
    spread_label = "Bull Put" if is_bull else "Bear Call"

    st.markdown("---")
    move_str = f"+{res['target_pct']:.0f}%" if is_bull else f"-{abs(res['target_pct']):.0f}%"
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  "
                f"Target ${target:,.0f} ({move_str}) in {res['target_dte']}d  |  {name}")

    df = pd.DataFrame(res["all_results"])

    # Top picks
    st.markdown("### Best Configurations")
    top = df.head(15).copy()
    disp = pd.DataFrame({
        "LEAPS": top["lc_k"].map(lambda x: f"{x:,.0f}"),
        "SC": top["sc_k"].map(lambda x: f"{x:,.0f}"),
        "Spread": top.apply(lambda r: f"{r['sp_k']:,.0f}/{r['lp_k']:,.0f}", axis=1),
        "L DTE": top["leaps_dte"],
        "S DTE": top["sc_dte"],
        "Delta": top["delta"].map(lambda x: f"{abs(x):.2f}"),
        "Cost": top["leaps_cost"].map(lambda x: f"${x*100:,.0f}"),
        "Cyc Cred": top["cycle_income"].map(lambda x: f"${x*100:,.0f}"),
        "Cyc": top["cycles"],
        "CycFund": top["cycles_to_fund"].map(lambda x: f"{x:.1f}"),
        "WksFund": top.apply(lambda r: f"{r['cycles_to_fund']*r['days_to_50']/7:.1f}", axis=1),
        "BE": top["breakeven"].map(lambda x: f"${x:,.1f}"),
        "P&L": top["pnl_target"].map(lambda x: f"${x*100:+,.0f}"),
        "ROC": top["roc"].map(lambda x: f"{x:+.0f}%"),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config=SCAN_CFG)

    # Select
    options = [f"LEAPS {abs(r['lc_k']):.0f}{opt_label[0]} ({r['leaps_dte']}d, "
               f"Δ{abs(r['delta']):.2f}) + SC {r['sc_k']:.0f} + "
               f"{spread_label} {r['sp_k']:.0f}/{r['lp_k']:.0f} → ROC {r['roc']:+.0f}%"
               for _, r in df.head(10).iterrows()]
    if not options:
        return
    idx = st.selectbox("Select configuration", range(len(options)),
                        format_func=lambda i: options[i], key="crab_sel")
    pick = df.iloc[idx]

    # Position table
    st.markdown("### Position")
    lc_res = bs.calculate_all(spot, pick["lc_k"],
                               pick["leaps_dte"]/365, res["r"], pick["iv_lc"], res["q"],
                               pick["opt_type"])
    leg_rows = [
        {"Leg": f"Long {opt_label} (LEAPS)", "Strike": f"${pick['lc_k']:,.0f}",
         "Delta": f"{lc_res['delta']:.3f}", "IV": f"{pick['iv_lc']*100:.1f}%",
         "Price": f"${pick['leaps_cost']*100:,.0f}", "DTE": pick["leaps_dte"],
         "Role": f"Diagonal {opt_label}"},
        {"Leg": f"Short {opt_label}", "Strike": f"${pick['sc_k']:,.0f}",
         "Delta": f"{pick['sc_res']['delta']:.3f}", "IV": f"{pick['iv_sc']*100:.1f}%",
         "Price": f"${pick['sc_res']['price']*100:,.0f}", "DTE": pick["sc_dte"],
         "Role": f"Diagonal {opt_label}"},
    ]
    sp_type = "Put" if is_bull else "Call"

    # Diagonal spread offset: LP gets +20% DTE if enabled
    diag_on = res.get("diag_spread", False)
    sc_dte_val = int(pick["sc_dte"])
    if diag_on:
        lp_dte_min = round(sc_dte_val * 1.20)
        # Find next available expiration >= +20%
        avail_dtes = sorted(c["dte_actual"] for c in res["chain_by_exp"].values())
        lp_dte_val = sc_dte_val  # fallback
        for d in avail_dtes:
            if d >= lp_dte_min and d > sc_dte_val:
                lp_dte_val = d
                break
        lp_role = f"{spread_label} Spread (Diag +{lp_dte_val - sc_dte_val}d)"
    else:
        lp_dte_val = sc_dte_val
        lp_role = f"{spread_label} Spread"

    leg_rows.extend([
        {"Leg": f"Short {sp_type}", "Strike": f"${pick['sp_k']:,.0f}",
         "Delta": f"{pick['sp_res']['delta']:.3f}", "IV": f"{pick['iv_sp']*100:.1f}%",
         "Price": f"${pick['sp_res']['price']*100:,.0f}", "DTE": sc_dte_val,
         "Role": f"{spread_label} Spread"},
        {"Leg": f"Long {sp_type}", "Strike": f"${pick['lp_k']:,.0f}",
         "Delta": f"{pick['lp_res']['delta']:.3f}", "IV": f"{pick['iv_lp']*100:.1f}%",
         "Price": f"${pick['lp_res']['price']*100:,.0f}",
         "DTE": lp_dte_val,
         "Role": lp_role},
    ])
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # Economics
    st.markdown("### Economics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LEAPS Cost", f"${pick['leaps_cost']*100:,.0f}",
              help=f"Capital required (vs shares: ${spot*100:,.0f})")
    m2.metric("Cycle Credit", f"${pick['cycle_income']*100:,.0f}",
              help="Per-cycle income at 50%: SC premium + spread credit")
    m3.metric("Cycles to Fund", f"{pick['cycles_to_fund']:.1f}",
              help=f"Cycles to fully pay for the LEAPS")
    m4.metric("Days to 50%", f"{pick['days_to_50']:.0f}d",
              help="Days per cycle until short-DTE legs reach 50% profit")
    m5.metric("Weeks to Fund", f"{pick['cycles_to_fund'] * pick['days_to_50'] / 7:.1f}",
              help="Total weeks = cycles × days per cycle / 7")
    remaining = pick["remaining_after"]
    m6.metric("DTE after Funding", f"{remaining:.0f}d",
              delta="safe" if remaining > 45 else "at risk",
              delta_color="normal" if remaining > 45 else "inverse",
              help="LEAPS DTE remaining after all funding cycles")

    if remaining < 30:
        st.warning(f"LEAPS has only ~{remaining:.0f}d after funding. "
                   "Consider longer LEAPS DTE.")

    # Comparison
    st.markdown(f"### {name} vs 100 Shares")
    shares_label = "100 Shares Long" if is_bull else "100 Shares Short"
    pnl_stock = abs(target - spot)
    comp = [
        {"": "Capital", name: f"${pick['leaps_cost']*100:,.0f}",
         shares_label: f"${spot*100:,.0f}",
         "": f"${(spot-pick['leaps_cost'])*100:,.0f} saved"},
        {"": "Monthly Income", name: f"${pick['cycle_income']*100:,.0f}",
         shares_label: "$0", "": ""},
        {"": f"P&L @ ${target:,.0f}", name: f"${pick['pnl_target']*100:+,.0f}",
         shares_label: f"${pnl_stock*100:+,.0f}",
         "": f"ROC {pick['roc']:+.0f}% vs {pnl_stock/spot*100:+.0f}%"},
        {"": "Break-Even", name: f"${pick['breakeven']:,.1f}",
         shares_label: f"${spot:,.2f}", "": ""},
    ]
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    # Scenario chart
    st.markdown("### Scenario Analysis")
    import plotly.graph_objects as go
    scenarios = []
    for pct in [-30, -20, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50]:
        s_end = spot * (1 + pct / 100)
        if is_bull:
            leaps_val = max(s_end - pick["lc_k"], 0)
        else:
            leaps_val = max(pick["lc_k"] - s_end, 0)
        crab_pnl = (leaps_val - pick["leaps_cost"] + pick["total_income"]) * 100
        stock_pnl = (s_end - spot) * 100 if is_bull else (spot - s_end) * 100
        scenarios.append({"spot": s_end, "crab": crab_pnl, "stock": stock_pnl})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[s["spot"] for s in scenarios], y=[s["crab"] for s in scenarios],
        mode="lines+markers", name=name, line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(
        x=[s["spot"] for s in scenarios], y=[s["stock"] for s in scenarios],
        mode="lines+markers", name=shares_label,
        line=dict(color="#aaa", width=1, dash="dash")))
    fig.add_vline(x=spot, line_dash="dot", line_color="gray",
                  annotation_text=f"Spot ${spot:,.0f}")
    fig.add_vline(x=target, line_dash="dot", line_color="green",
                  annotation_text=f"Target ${target:,.0f}")
    fig.add_vline(x=pick["breakeven"], line_dash="dot", line_color="orange",
                  annotation_text=f"BE ${pick['breakeven']:,.1f}")
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    fig.update_layout(template="plotly_white", height=400,
        xaxis_title="Stock Price at LEAPS Expiry",
        yaxis_title="P&L ($, per contract)",
        margin=dict(l=60, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("### Export")
    # Find expirations for export
    sc_exp_pick = min(res["chain_by_exp"].keys(),
                       key=lambda e: abs(res["chain_by_exp"][e]["dte_actual"] - sc_dte_val))
    lp_exp_pick = min(res["chain_by_exp"].keys(),
                       key=lambda e: abs(res["chain_by_exp"][e]["dte_actual"] - lp_dte_val))
    os_legs = [
        {"strike": int(pick["lc_k"]), "option_type": pick["opt_type"],
         "expiration": str(pick["exp"]), "long": True, "qty": 1},
        {"strike": int(pick["sc_k"]), "option_type": pick["opt_type"],
         "expiration": str(sc_exp_pick), "long": False, "qty": 1},
        {"strike": int(pick["sp_k"]), "option_type": pick["spread_type"],
         "expiration": str(sc_exp_pick), "long": False, "qty": 1},
        {"strike": int(pick["lp_k"]), "option_type": pick["spread_type"],
         "expiration": str(lp_exp_pick), "long": True, "qty": 1},
    ]
    os_sorted = sorted(os_legs, key=lambda l: (l["long"], l["option_type"] != "put"))
    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], os_sorted)
        if url:
            st.markdown(f"[Full Position: OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], os_sorted, tag="CrawlCrab")
        st.download_button("IBKR Basket CSV", csv,
                            f"crab_{res['symbol'].replace('^','')}.csv", "text/csv")

    # Separate links
    diag_legs = os_legs[:2]
    spread_legs = os_legs[2:]
    c1, c2 = st.columns(2)
    with c1:
        url_d = bs.optionstrat_url(res["symbol"], diag_legs)
        if url_d:
            st.markdown(f"[Diagonal {opt_label}]({url_d})")
    with c2:
        url_s = bs.optionstrat_url(res["symbol"], spread_legs)
        if url_s:
            st.markdown(f"[{spread_label} Spread]({url_s})")

    with st.expander("Strategy Guide"):
        st.markdown(
            f"**{name} = Diagonal {opt_label} + {spread_label} Spread**\n\n"
            f"**4 Legs:**\n"
            f"- Long {opt_label} {pick['lc_k']:.0f} ({pick['leaps_dte']}d): "
            f"${pick['leaps_cost']*100:,.0f} — the core trend bet\n"
            f"- Short {opt_label} {pick['sc_k']:.0f} ({pick['sc_dte']}d): "
            f"${pick['sc_res']['price']*100:,.0f} — diagonal income\n"
            f"- Short {sp_type} {pick['sp_k']:.0f} ({pick['sc_dte']}d): "
            f"${pick['sp_res']['price']*100:,.0f} — spread income\n"
            f"- Long {sp_type} {pick['lp_k']:.0f} ({pick['sc_dte']}d): "
            f"${pick['lp_res']['price']*100:,.0f} — spread protection\n\n"
            f"**Cycle (~{pick['days_to_50']:.0f}d):**\n"
            f"1. Close all 3 short-DTE legs at 50% profit → ${pick['cycle_income']*100:,.0f}\n"
            f"2. The LEAPS stays open\n"
            f"3. Open 3 new short-DTE legs at current deltas\n"
            f"4. After ~{pick['cycles_to_fund']:.0f} cycles, LEAPS is fully funded\n\n"
            f"**LEAPS Management:**\n"
            f"- OTM: keep cycling shorts against it\n"
            f"- ATM/ITM: let it run in profit, buy new LEAPS further along trend\n"
            f"- Goal: accumulate ITM LEAPS over weeks/months"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Crawling Crab")
    st.caption("Stock Replacement + Credit Spread income. "
               "The credit spread accelerates LEAPS financing.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="GOOGL",
            help="Underlying to trade.").upper()
    with c2:
        direction = st.selectbox("Direction", ["Bull", "Bear"],
            help="Bull: Long Call LEAPS + Short Call + Bull Put Spread. "
                 "Bear: Long Put LEAPS + Short Put + Bear Call Spread.")
    with c3:
        target_pct = st.number_input("Target %", value=20.0,
            min_value=5.0, max_value=100.0, step=5.0, format="%.0f",
            help="Expected % move. Bull +20% = stock rises 20%.")
    with c4:
        target_dte = st.number_input("Target DTE", value=270,
            min_value=30, max_value=730, step=30,
            help="Days until target. LEAPS must outlast this + 30d buffer.")

    c5, c6, c7, c8, c9, c10, c11 = st.columns([1, 1, 1, 1, 1, 1, 1])
    with c5:
        mode = st.radio("Mode", ["Auto", "Manual"], horizontal=True,
            help="Auto scans all short DTEs (10-50d). Manual uses fixed short DTE.")
    with c6:
        diag_spread = st.toggle("Diagonal Spread", value=True,
            help="Long leg of credit spread gets +20% DTE. "
                 "Better IV-spike protection, minimal cost.")
    with c7:
        min_long_dte = st.number_input("Min Long DTE", value=60,
            min_value=30, max_value=180,
            help="Minimum remaining DTE after funding cycles complete.")
    with c8:
        sc_delta = st.number_input("Diag Short Δ", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for the short leg of the diagonal.")
    with c9:
        sp_d_short = st.number_input("Spread Short Δ", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta of the short leg in the credit spread.")
    with c10:
        sp_d_long = st.number_input("Spread Long Δ", value=0.20,
            min_value=0.05, max_value=0.35, step=0.05, format="%.2f",
            help="Delta of the long (protective) leg in the spread.")
    with c11:
        sc_dte = st.number_input("Short DTE", value=30,
            min_value=14, max_value=60,
            help="DTE for short-DTE legs (Manual mode only).")

    step = 5 if spot_guess(symbol) > 50 else 1

    if st.button(f"Build {'Bull' if direction == 'Bull' else 'Bear'} Crab",
                  type="primary", use_container_width=True):
        with st.spinner("Scanning configurations..."):
            try:
                result = compute(symbol, step, direction,
                                  target_pct, target_dte, min_long_dte,
                                  sc_delta, sp_d_short, sp_d_long, sc_dte,
                                  auto_mode=(mode == "Auto"))
                result["diag_spread"] = diag_spread
                st.session_state["crab_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "crab_result" not in st.session_state:
        st.info("Set parameters and click Build.")
        return

    display(st.session_state["crab_result"])


def spot_guess(symbol):
    s = symbol.upper().replace("^", "")
    if s in ("SPX", "GSPC", "NDX"):
        return 5000
    if s in ("SPY", "QQQ", "IWM"):
        return 400
    return 200


main()
