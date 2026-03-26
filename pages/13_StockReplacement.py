"""
Stock Replacement – PMCC/PMCP Optimizer

Bull: Long Call LEAPS + Short Call monthly (Poor Man's Covered Call)
Bear: Long Put LEAPS + Short Put monthly (Poor Man's Covered Put)

Finds optimal LEAPS strike/DTE based on target move %, time horizon,
annualized theta cost, and capital efficiency.
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
         expirations_data, sc_dte_actual, sc_smile, sc_delta):
    """Scan LEAPS expirations and deltas for optimal PMCC/PMCP."""
    results = []
    is_bull = direction == "Bull"
    opt_type = "call" if is_bull else "put"
    target_price = spot * (1 + target_pct / 100) if is_bull else spot * (1 - abs(target_pct) / 100)
    T_s = sc_dte_actual / 365.0

    # Short option (fixed)
    try:
        if is_bull:
            sc_k = _round(bs.solve_strike_for_delta(sc_delta, spot, T_s, r, iv_atm, q, "call"), step)
        else:
            sc_k = _round(bs.solve_strike_for_delta(-sc_delta, spot, T_s, r, iv_atm, q, "put"), step)
    except Exception:
        return results, target_price

    iv_sc = _iv(sc_smile, sc_k, iv_atm)
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, opt_type)

    for exp_str, chain_data in expirations_data.items():
        dte = chain_data["dte_actual"]
        if dte < 60 or dte > 548:
            continue
        # LEAPS must outlast the target horizon
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
            lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, opt_type)

            if lc["price"] < 0.50:
                continue

            intrinsic = max(spot - lc_k, 0) if is_bull else max(lc_k - spot, 0)
            extrinsic = lc["price"] - intrinsic
            extrinsic_pct = extrinsic / lc["price"] * 100 if lc["price"] > 0 else 0

            cycles = max(1, int((dte - 45) / sc_dte_actual))
            income = sc["price"] * 0.50 * cycles
            net = lc["price"] - income
            if is_bull:
                be = lc_k + net
            else:
                be = lc_k - net

            if is_bull:
                pnl_target = max(target_price - lc_k, 0) - lc["price"] + income
            else:
                pnl_target = max(lc_k - target_price, 0) - lc["price"] + income

            if pnl_target <= 0:
                continue

            pnl_stock = abs(target_price - spot)
            roc = pnl_target / lc["price"] * 100 if lc["price"] > 0 else 0
            leverage = abs(lc["delta"]) * spot / lc["price"] if lc["price"] > 0 else 0

            # Annualized theta cost
            hold_days = max(dte - 45, 30)
            rolls_per_year = 365 / hold_days
            try:
                lc_at_exit = bs.calculate_all(spot, lc_k, 45 / 365, r, iv_lc, q, opt_type)
                roll_loss = lc["price"] - lc_at_exit["price"]
            except Exception:
                roll_loss = lc["price"] * 0.5
            annual_cost = roll_loss * rolls_per_year

            # Score
            if is_bull:
                be_margin = (spot - be) / spot * 100
            else:
                be_margin = (be - spot) / spot * 100
            cap_eff = min(extrinsic_pct / 50, 2.0)
            theta_per_delta = annual_cost / max(abs(lc["delta"]), 0.1)
            theta_eff = max(0.2, 1.0 - theta_per_delta / (spot * 0.5))
            if dte < 120:
                dte_eff = 0.3 + 0.7 * (dte / 120)
            elif dte <= 400:
                dte_eff = 1.0
            else:
                dte_eff = max(0.5, 1.0 - (dte - 400) / 400)
            score = roc * max(0.1, 1 + be_margin / 10) * cap_eff * theta_eff * dte_eff

            results.append({
                "lc_k": lc_k, "sc_k": sc_k,
                "leaps_dte": dte, "sc_dte": sc_dte_actual,
                "delta": lc["delta"],
                "leaps_cost": lc["price"],
                "extrinsic_pct": extrinsic_pct,
                "leverage": leverage,
                "annual_cost": annual_cost,
                "sc_premium": sc["price"],
                "cycles": cycles,
                "total_income": income,
                "net_cost": net,
                "breakeven": be,
                "be_margin": be_margin,
                "pnl_target": pnl_target,
                "pnl_stock": pnl_stock,
                "roc": roc,
                "score": score,
                "exp": exp_str,
                "iv_lc": iv_lc,
                "opt_type": opt_type,
            })

    return results, target_price


# ── Build full PMCC/PMCP ─────────────────────────────────────────────────

def build_position(spot, r, q, iv_atm, step, direction, target_price,
                    leaps_dte, leaps_delta, sc_dte, sc_delta,
                    smile_l, smile_s):
    is_bull = direction == "Bull"
    opt_type = "call" if is_bull else "put"
    T_l = leaps_dte / 365.0
    T_s = sc_dte / 365.0

    try:
        if is_bull:
            lc_k = _round(bs.solve_strike_for_delta(leaps_delta, spot, T_l, r, iv_atm, q, "call"), step)
            sc_k = _round(bs.solve_strike_for_delta(sc_delta, spot, T_s, r, iv_atm, q, "call"), step)
        else:
            lc_k = _round(bs.solve_strike_for_delta(-leaps_delta, spot, T_l, r, iv_atm, q, "put"), step)
            sc_k = _round(bs.solve_strike_for_delta(-sc_delta, spot, T_s, r, iv_atm, q, "put"), step)
    except Exception:
        return None

    iv_lc = _iv(smile_l, lc_k, iv_atm)
    iv_sc = _iv(smile_s, sc_k, iv_atm)
    lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, opt_type)
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, opt_type)

    cycles = max(1, int((leaps_dte - 45) / sc_dte))
    income_per_cycle = sc["price"] * 0.50
    total_income = income_per_cycle * cycles
    net_cost = lc["price"] - total_income

    if is_bull:
        be = lc_k + net_cost
        pnl_at_target = max(target_price - lc_k, 0) - lc["price"] + total_income
    else:
        be = lc_k - net_cost
        pnl_at_target = max(lc_k - target_price, 0) - lc["price"] + total_income

    pnl_stock = abs(target_price - spot)
    capital_eff = pnl_at_target / lc["price"] * 100 if lc["price"] > 0 else 0

    # Scenarios
    scenarios = []
    for pct in [-30, -20, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50]:
        s_end = spot * (1 + pct / 100)
        if is_bull:
            leaps_val = max(s_end - lc_k, 0)
        else:
            leaps_val = max(lc_k - s_end, 0)
        pmcc_pnl = (leaps_val - lc["price"] + total_income) * 100
        stock_pnl = (s_end - spot) * 100 if is_bull else (spot - s_end) * 100
        scenarios.append({"spot": s_end, "pct": pct,
                          "pmcc_pnl": pmcc_pnl, "stock_pnl": stock_pnl})

    return {
        "lc_k": lc_k, "sc_k": sc_k, "lc": lc, "sc": sc,
        "iv_lc": iv_lc, "iv_sc": iv_sc, "opt_type": opt_type,
        "leaps_cost": lc["price"], "sc_premium": sc["price"],
        "income_per_cycle": income_per_cycle, "cycles_possible": cycles,
        "total_income": total_income, "net_cost": net_cost,
        "breakeven": be, "pnl_at_target": pnl_at_target,
        "pnl_stock": pnl_stock, "capital_efficiency": capital_eff,
        "leaps_dte": leaps_dte, "sc_dte": sc_dte,
        "scenarios": scenarios, "direction": direction,
    }


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, direction, target_pct, target_dte, sc_delta, sc_dte_target):
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

    sc_exp = min(chain_by_exp.keys(),
                  key=lambda e: abs(chain_by_exp[e]["dte_actual"] - sc_dte_target))
    sc_chain = chain_by_exp[sc_exp]
    opt_type = "call" if direction == "Bull" else "put"
    sc_smile = build_smile_curve(
        sc_chain["calls"] if direction == "Bull" else sc_chain["puts"], spot)

    leaps_data = {e: c for e, c in chain_by_exp.items() if c["dte_actual"] >= 60}

    results, target_price = scan(
        spot, r, q, iv, step, direction, target_pct, target_dte,
        leaps_data, sc_chain["dte_actual"], sc_smile, sc_delta)

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
        "chain_by_exp": chain_by_exp, "sc_smile": sc_smile,
    }


# ── Display ──────────────────────────────────────────────────────────────

SCAN_CFG = {
    "LEAPS": st.column_config.TextColumn("LEAPS",
        help="LEAPS strike. ATM/OTM = max leverage. For bear: put strike."),
    "Short": st.column_config.TextColumn("Short",
        help="Short option strike (rolled monthly for income)"),
    "DTE": st.column_config.NumberColumn("DTE", help="LEAPS days to expiration"),
    "Delta": st.column_config.TextColumn("Delta",
        help="LEAPS delta. 0.50-0.60 = best capital efficiency."),
    "Cost": st.column_config.TextColumn("Cost", help="LEAPS premium per contract (×100)"),
    "Ext%": st.column_config.TextColumn("Ext%",
        help="% of cost that is extrinsic (optionality). Higher = better."),
    "Lever": st.column_config.TextColumn("Lever",
        help="Effective leverage vs holding shares"),
    "Ann$": st.column_config.TextColumn("Ann$",
        help="Annualized theta cost including rolls (×100). Lower = better."),
    "SC/mo": st.column_config.TextColumn("SC/mo",
        help="Short option 50% income per cycle (×100)"),
    "Cyc": st.column_config.NumberColumn("Cyc", help="Income cycles possible"),
    "Net$": st.column_config.TextColumn("Net$",
        help="LEAPS cost minus total income (×100)"),
    "BE": st.column_config.TextColumn("BE", help="Break-even price at expiry"),
    "P&L": st.column_config.TextColumn("P&L", help="Profit at target (×100)"),
    "ROC": st.column_config.TextColumn("ROC", help="Return on Capital at target"),
}


def display(res):
    spot = res["spot"]
    target = res["target_price"]
    direction = res["direction"]
    is_bull = direction == "Bull"
    label = "PMCC" if is_bull else "PMCP"
    opt_label = "Call" if is_bull else "Put"

    st.markdown("---")
    move_str = f"+{res['target_pct']:.0f}%" if is_bull else f"-{abs(res['target_pct']):.0f}%"
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  "
                f"Target ${target:,.0f} ({move_str}) in {res['target_dte']}d")

    df = pd.DataFrame(res["all_results"])

    st.markdown("### Best Configurations")
    top = df.head(15).copy()
    disp = pd.DataFrame({
        "LEAPS": top["lc_k"].map(lambda x: f"{x:,.0f}"),
        "Short": top["sc_k"].map(lambda x: f"{x:,.0f}"),
        "DTE": top["leaps_dte"],
        "Delta": top["delta"].map(lambda x: f"{abs(x):.2f}"),
        "Cost": top["leaps_cost"].map(lambda x: f"${x*100:,.0f}"),
        "Ext%": top["extrinsic_pct"].map(lambda x: f"{x:.0f}%"),
        "Lever": top["leverage"].map(lambda x: f"{x:.1f}x"),
        "Ann$": top["annual_cost"].map(lambda x: f"${x*100:,.0f}"),
        "SC/mo": top["sc_premium"].map(lambda x: f"${x*50:,.0f}"),
        "Cyc": top["cycles"],
        "Net$": top["net_cost"].map(lambda x: f"${x*100:,.0f}"),
        "BE": top["breakeven"].map(lambda x: f"${x:,.1f}"),
        "P&L": top["pnl_target"].map(lambda x: f"${x*100:+,.0f}"),
        "ROC": top["roc"].map(lambda x: f"{x:+.0f}%"),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config=SCAN_CFG)

    # Select config
    options = [f"LEAPS {abs(r['lc_k']):.0f}{opt_label[0]} ({r['leaps_dte']}d, "
               f"Δ{abs(r['delta']):.2f}) + Short {r['sc_k']:.0f}{opt_label[0]} "
               f"→ ROC {r['roc']:+.0f}%"
               for _, r in df.head(10).iterrows()]
    if not options:
        return
    idx = st.selectbox("Select configuration", range(len(options)),
                        format_func=lambda i: options[i], key="sr_sel")
    pick = df.iloc[idx]

    # Build full position
    chain_l = res["chain_by_exp"].get(pick["exp"])
    if chain_l is None:
        st.error("Chain not found.")
        return
    smile_l = build_smile_curve(
        chain_l["calls"] if is_bull else chain_l["puts"], spot)

    pmcc = build_position(spot, res["r"], res["q"], res["iv"],
                           5 if spot > 100 else 1,
                           direction, target,
                           int(pick["leaps_dte"]), abs(float(pick["delta"])),
                           int(res["sc_dte_actual"]), 0.30,
                           smile_l, res["sc_smile"])
    if pmcc is None:
        st.error("Could not build position.")
        return

    # Position
    st.markdown("### Position")
    leg_rows = [
        {"Leg": f"Long {opt_label} (LEAPS)", "Strike": f"${pmcc['lc_k']:,.0f}",
         "Delta": f"{pmcc['lc']['delta']:.3f}", "IV": f"{pmcc['iv_lc']*100:.1f}%",
         "Price": f"${pmcc['leaps_cost']*100:,.0f}", "DTE": pmcc["leaps_dte"]},
        {"Leg": f"Short {opt_label} (monthly)", "Strike": f"${pmcc['sc_k']:,.0f}",
         "Delta": f"{pmcc['sc']['delta']:.3f}", "IV": f"{pmcc['iv_sc']*100:.1f}%",
         "Price": f"${pmcc['sc_premium']*100:,.0f}", "DTE": pmcc["sc_dte"]},
    ]
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # Economics
    st.markdown("### Economics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LEAPS Cost", f"${pmcc['leaps_cost']*100:,.0f}",
              help=f"Capital required (vs shares: ${spot*100:,.0f})")
    m2.metric("Income/Cycle", f"${pmcc['income_per_cycle']*100:,.0f}",
              help="Short option 50% profit per cycle")
    m3.metric("Cycles", f"{pmcc['cycles_possible']}",
              help="Cycles before LEAPS hits 45 DTE")
    m4.metric("Total Income", f"${pmcc['total_income']*100:,.0f}")
    m5.metric("Net Cost", f"${pmcc['net_cost']*100:,.0f}",
              help="LEAPS cost minus total income")
    m6.metric("Break-Even", f"${pmcc['breakeven']:,.1f}",
              delta=f"{(pmcc['breakeven']/spot-1)*100:+.1f}% from spot" if is_bull
              else f"{(1-pmcc['breakeven']/spot)*100:+.1f}% from spot",
              delta_color="normal" if (is_bull and pmcc['breakeven'] <= spot) or
                          (not is_bull and pmcc['breakeven'] >= spot) else "inverse")

    # Comparison
    shares_label = "100 Shares Long" if is_bull else "100 Shares Short"
    st.markdown(f"### {label} vs {shares_label}")
    comp = [
        {"": "Capital", label: f"${pmcc['leaps_cost']*100:,.0f}",
         shares_label: f"${spot*100:,.0f}",
         "Savings": f"${(spot-pmcc['leaps_cost'])*100:,.0f} ({(1-pmcc['leaps_cost']/spot)*100:.0f}%)"},
        {"": "Delta", label: f"{abs(pmcc['lc']['delta']):.2f}",
         shares_label: "1.00", "Savings": ""},
        {"": "Monthly Income", label: f"${pmcc['income_per_cycle']*100:,.0f}",
         shares_label: "$0", "Savings": f"+${pmcc['income_per_cycle']*100:,.0f}/mo"},
        {"": f"P&L @ ${target:,.0f}", label: f"${pmcc['pnl_at_target']*100:+,.0f}",
         shares_label: f"${pmcc['pnl_stock']*100:+,.0f}",
         "Savings": f"ROC {pmcc['capital_efficiency']:+.0f}% vs "
                    f"{pmcc['pnl_stock']/spot*100:+.0f}%"},
        {"": "Break-Even", label: f"${pmcc['breakeven']:,.1f}",
         shares_label: f"${spot:,.2f}", "Savings": ""},
        {"": "Max Loss", label: f"${pmcc['net_cost']*100:,.0f}",
         shares_label: f"${spot*100:,.0f}", "Savings": ""},
    ]
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    # Scenario chart
    st.markdown("### Scenario Analysis")
    import plotly.graph_objects as go
    scen = pmcc["scenarios"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[s["spot"] for s in scen], y=[s["pmcc_pnl"] for s in scen],
        mode="lines+markers", name=label, line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(
        x=[s["spot"] for s in scen], y=[s["stock_pnl"] for s in scen],
        mode="lines+markers", name=shares_label,
        line=dict(color="#aaa", width=1, dash="dash")))
    fig.add_vline(x=spot, line_dash="dot", line_color="gray",
                  annotation_text=f"Spot ${spot:,.0f}")
    fig.add_vline(x=target, line_dash="dot", line_color="green",
                  annotation_text=f"Target ${target:,.0f}")
    fig.add_vline(x=pmcc["breakeven"], line_dash="dot", line_color="orange",
                  annotation_text=f"BE ${pmcc['breakeven']:,.1f}")
    fig.add_hline(y=0, line_color="gray", line_width=0.5)
    fig.update_layout(template="plotly_white", height=400,
        xaxis_title="Stock Price at LEAPS Expiry",
        yaxis_title="P&L ($, per contract)",
        margin=dict(l=60, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("### Export")
    os_legs = [
        {"strike": int(pmcc["lc_k"]), "option_type": pmcc["opt_type"],
         "expiration": str(pick["exp"]), "long": True, "qty": 1},
        {"strike": int(pmcc["sc_k"]), "option_type": pmcc["opt_type"],
         "expiration": str(res["sc_exp"]), "long": False, "qty": 1},
    ]
    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], os_legs)
        if url:
            st.markdown(f"[OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], os_legs, tag="StockRepl")
        st.download_button("IBKR Basket CSV", csv,
                            f"stockrepl_{res['symbol'].replace('^','')}.csv", "text/csv")

    with st.expander("Strategy Guide"):
        st.markdown(
            f"**Setup ({label}):**\n"
            f"- Buy LEAPS {pmcc['lc_k']:.0f}{opt_label[0]} ({pmcc['leaps_dte']}d): "
            f"${pmcc['leaps_cost']*100:,.0f}\n"
            f"- Sell {pmcc['sc_k']:.0f}{opt_label[0]} ({pmcc['sc_dte']}d): "
            f"${pmcc['sc_premium']*100:,.0f} income\n\n"
            f"**Monthly Cycle:**\n"
            f"1. When short {opt_label.lower()} reaches 50% profit → close it\n"
            f"2. Sell new OTM {opt_label.lower()} at ~{abs(pmcc['sc']['delta']):.2f} delta\n"
            f"3. Repeat up to {pmcc['cycles_possible']} times\n\n"
            f"**If stock moves {'past short call' if is_bull else 'below short put'}:**\n"
            f"- Roll the short {opt_label.lower()} {'up' if is_bull else 'down'} and out\n"
            f"- Or close short at a loss, let LEAPS run\n\n"
            f"**Exit:**\n"
            f"- At target ${target:,.0f}: close both, profit ${pmcc['pnl_at_target']*100:+,.0f}\n"
            f"- Or keep rolling shorts as synthetic covered {opt_label.lower()}"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Stock Replacement")
    st.caption("Replace stock positions with PMCC (bull) or PMCP (bear). "
               "Same directional exposure at a fraction of the capital, plus monthly income.")

    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="GOOGL",
            help="Stock to replace with options.").upper()
    with c2:
        direction = st.selectbox("Direction", ["Bull", "Bear"],
            help="Bull: Long Call LEAPS + Short Call monthly (PMCC). "
                 "Bear: Long Put LEAPS + Short Put monthly (PMCP).")
    with c3:
        target_pct = st.number_input("Target %", value=20.0,
            min_value=5.0, max_value=100.0, step=5.0, format="%.0f",
            help="Expected % move in your direction. "
                 "Bull +20% means stock rises 20%. Bear +20% means stock drops 20%.")
    with c4:
        target_dte = st.number_input("Target DTE", value=270,
            min_value=30, max_value=730, step=30,
            help="Days until you expect the target to be reached. "
                 "LEAPS must outlast this + 30d buffer.")
    with c5:
        sc_delta = st.number_input("SC Delta", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for the monthly short option. 0.25-0.35 typical.")
    with c6:
        sc_dte = st.number_input("SC DTE", value=30,
            min_value=14, max_value=60,
            help="DTE for the short option. 30-45d optimal.")

    step = 5 if spot_guess(symbol) > 50 else 1

    if st.button("Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning configurations..."):
            try:
                result = compute(symbol, step, direction,
                                  target_pct, target_dte, sc_delta, sc_dte)
                st.session_state["sr_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "sr_result" not in st.session_state:
        st.info("Enter your thesis and click Scan.")
        return

    display(st.session_state["sr_result"])


def spot_guess(symbol):
    """Quick guess for strike step before data loads."""
    s = symbol.upper().replace("^", "")
    if s in ("SPX", "GSPC", "NDX"):
        return 5000
    if s in ("SPY", "QQQ", "IWM"):
        return 400
    return 200


main()
