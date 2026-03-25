"""
Stock Replacement – PMCC Optimizer

Find the optimal Poor Man's Covered Call to replace a stock position:
- Deep ITM LEAPS Call as stock substitute (high delta, fraction of cost)
- Rolling short OTM calls for monthly income (reduce cost basis)

Input: Symbol, price target, target date
Output: Optimal LEAPS strike, short call parameters, scenario analysis
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


# ── Build PMCC ───────────────────────────────────────────────────────────

def build_pmcc(spot, r, q, iv_atm, step, target_price,
               leaps_dte, leaps_delta, sc_dte, sc_delta,
               smile_call_l, smile_call_s):
    """Build a PMCC position and compute economics."""
    T_l = leaps_dte / 365.0
    T_s = sc_dte / 365.0

    # LEAPS Call strike from delta
    try:
        lc_k = _round(bs.solve_strike_for_delta(leaps_delta, spot, T_l, r, iv_atm, q, "call"), step)
    except Exception:
        return None

    # Short Call strike from delta
    try:
        sc_k = _round(bs.solve_strike_for_delta(sc_delta, spot, T_s, r, iv_atm, q, "call"), step)
    except Exception:
        return None

    if sc_k <= lc_k:
        sc_k = lc_k + step

    # Price both legs
    iv_lc = _iv(smile_call_l, lc_k, iv_atm)
    iv_sc = _iv(smile_call_s, sc_k, iv_atm)

    lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, "call")
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, "call")

    # Monthly income cycles
    cycles_possible = max(1, int((leaps_dte - 45) / sc_dte))
    income_per_cycle = sc["price"] * 0.50  # close at 50%
    total_income = income_per_cycle * cycles_possible
    net_cost = lc["price"] - total_income
    breakeven = lc_k + net_cost

    # P&L at target
    pnl_at_target = max(target_price - lc_k, 0) - lc["price"] + total_income
    pnl_stock = target_price - spot
    capital_efficiency = pnl_at_target / lc["price"] * 100 if lc["price"] > 0 else 0

    # Scenario P&L
    scenarios = []
    for pct in [-20, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50]:
        s_end = spot * (1 + pct / 100)
        leaps_val = max(s_end - lc_k, 0)
        pmcc_pnl = (leaps_val - lc["price"] + total_income) * 100
        stock_pnl = (s_end - spot) * 100
        scenarios.append({
            "spot": s_end,
            "pct": pct,
            "pmcc_pnl": pmcc_pnl,
            "stock_pnl": stock_pnl,
            "pmcc_roc": pmcc_pnl / (lc["price"] * 100) * 100 if lc["price"] > 0 else 0,
            "stock_roc": stock_pnl / (spot * 100) * 100,
        })

    return {
        "lc_k": lc_k, "sc_k": sc_k,
        "lc": lc, "sc": sc,
        "iv_lc": iv_lc, "iv_sc": iv_sc,
        "leaps_cost": lc["price"],
        "sc_premium": sc["price"],
        "income_per_cycle": income_per_cycle,
        "cycles_possible": cycles_possible,
        "total_income": total_income,
        "net_cost": net_cost,
        "breakeven": breakeven,
        "pnl_at_target": pnl_at_target,
        "pnl_stock": pnl_stock,
        "capital_efficiency": capital_efficiency,
        "leaps_dte": leaps_dte, "sc_dte": sc_dte,
        "scenarios": scenarios,
    }


# ── Scan ─────────────────────────────────────────────────────────────────

def scan_pmcc(spot, r, q, iv_atm, step, target_price,
              expirations_data, sc_dte_actual, sc_smile,
              sc_delta):
    """Scan LEAPS expirations and deltas to find optimal PMCC."""
    results = []
    T_s = sc_dte_actual / 365.0

    # Short call (fixed for all scans)
    try:
        sc_k = _round(bs.solve_strike_for_delta(sc_delta, spot, T_s, r, iv_atm, q, "call"), step)
    except Exception:
        return results

    iv_sc = _iv(sc_smile, sc_k, iv_atm)
    sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, "call")

    for exp_str, chain_data in expirations_data.items():
        dte = chain_data["dte_actual"]
        if dte < 60:
            continue
        T_l = dte / 365.0
        call_smile = build_smile_curve(chain_data["calls"], spot)

        for delta_target in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]:
            try:
                lc_k = _round(bs.solve_strike_for_delta(delta_target, spot, T_l, r, iv_atm, q, "call"), step)
            except Exception:
                continue

            if sc_k <= lc_k:
                continue

            iv_lc = _iv(call_smile, lc_k, iv_atm)
            lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, "call")

            if lc["price"] < 1.0:
                continue

            cycles = max(1, int((dte - 45) / sc_dte_actual))
            income = sc["price"] * 0.50 * cycles
            net = lc["price"] - income
            be = lc_k + net

            pnl_target = max(target_price - lc_k, 0) - lc["price"] + income
            pnl_flat = max(spot - lc_k, 0) - lc["price"] + income
            roc = pnl_target / lc["price"] * 100 if lc["price"] > 0 else 0

            # Score: balance ROC with break-even safety
            be_margin = (spot - be) / spot * 100  # how far BE is below spot
            score = roc * max(0, 1 + be_margin / 10)

            results.append({
                "lc_k": lc_k, "sc_k": sc_k,
                "leaps_dte": dte, "sc_dte": sc_dte_actual,
                "delta": lc["delta"],
                "leaps_cost": lc["price"],
                "sc_premium": sc["price"],
                "cycles": cycles,
                "total_income": income,
                "net_cost": net,
                "breakeven": be,
                "be_margin": be_margin,
                "pnl_target": pnl_target,
                "pnl_flat": pnl_flat,
                "roc": roc,
                "score": score,
                "exp": exp_str,
                "iv_lc": iv_lc,
            })

    return results


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, target_price, sc_delta, sc_dte_target):
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

    # Load chains
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

    # Short call chain (closest to target DTE)
    sc_exp = min(chain_by_exp.keys(),
                  key=lambda e: abs(chain_by_exp[e]["dte_actual"] - sc_dte_target))
    sc_chain = chain_by_exp[sc_exp]
    sc_smile = build_smile_curve(sc_chain["calls"], spot)

    # LEAPS chains (60+ DTE)
    leaps_data = {e: c for e, c in chain_by_exp.items() if c["dte_actual"] >= 60}

    results = scan_pmcc(spot, r, q, iv, step, target_price,
                         leaps_data, sc_chain["dte_actual"], sc_smile,
                         sc_delta)

    if not results:
        raise ValueError("No valid PMCC configurations found.")

    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "symbol": symbol, "spot": spot, "vix": vix, "iv": iv,
        "r": r, "q": q,
        "target_price": target_price,
        "sc_dte_actual": sc_chain["dte_actual"],
        "sc_exp": sc_exp,
        "all_results": results,
        "chain_by_exp": chain_by_exp,
        "sc_smile": sc_smile,
    }


# ── Display ──────────────────────────────────────────────────────────────

SCAN_CFG = {
    "LEAPS": st.column_config.TextColumn("LEAPS",
        help="LEAPS call strike (deep ITM for stock-like delta)"),
    "SC": st.column_config.TextColumn("SC",
        help="Short call strike (OTM, rolled monthly for income)"),
    "DTE": st.column_config.NumberColumn("DTE",
        help="LEAPS days to expiration"),
    "Delta": st.column_config.TextColumn("Delta",
        help="LEAPS delta (how closely it tracks the stock, 0.80+ = stock-like)"),
    "Cost": st.column_config.TextColumn("Cost",
        help="LEAPS premium (×100 per contract)"),
    "SC/mo": st.column_config.TextColumn("SC/mo",
        help="Short call 50% profit income per cycle (×100)"),
    "Cycles": st.column_config.NumberColumn("Cycles",
        help="Number of short call cycles possible before LEAPS expires"),
    "Income": st.column_config.TextColumn("Income",
        help="Total projected income from all short call cycles (×100)"),
    "Net Cost": st.column_config.TextColumn("Net Cost",
        help="LEAPS cost minus total projected short call income (×100)"),
    "BE": st.column_config.TextColumn("BE",
        help="Break-even price at LEAPS expiry (LEAPS strike + net cost)"),
    "P&L Target": st.column_config.TextColumn("P&L Target",
        help="Profit if stock reaches your target price (×100)"),
    "ROC": st.column_config.TextColumn("ROC",
        help="Return on Capital = P&L at target / LEAPS cost"),
    "Score": st.column_config.TextColumn("Score",
        help="Combined score: ROC weighted by break-even safety margin"),
}


def display(res):
    spot = res["spot"]
    target = res["target_price"]

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  "
                f"Target ${target:,.0f} ({(target/spot-1)*100:+.0f}%)")

    results = res["all_results"]
    df = pd.DataFrame(results)

    # Top picks table
    st.markdown("### Best Configurations")
    st.caption("Sorted by score (ROC × break-even safety). "
               "Higher delta = more stock-like. Higher DTE = more cycles.")

    top = df.head(15).copy()
    disp = pd.DataFrame({
        "LEAPS": top["lc_k"].map(lambda x: f"{x:,.0f}"),
        "SC": top["sc_k"].map(lambda x: f"{x:,.0f}"),
        "DTE": top["leaps_dte"],
        "Delta": top["delta"].map(lambda x: f"{x:.2f}"),
        "Cost": top["leaps_cost"].map(lambda x: f"${x*100:,.0f}"),
        "SC/mo": top["sc_premium"].map(lambda x: f"${x*50:,.0f}"),
        "Cycles": top["cycles"],
        "Income": top["total_income"].map(lambda x: f"${x*100:,.0f}"),
        "Net Cost": top["net_cost"].map(lambda x: f"${x*100:,.0f}"),
        "BE": top["breakeven"].map(lambda x: f"${x:,.1f}"),
        "P&L Target": top["pnl_target"].map(lambda x: f"${x*100:+,.0f}"),
        "ROC": top["roc"].map(lambda x: f"{x:+.0f}%"),
        "Score": top["score"].map(lambda x: f"{x:.0f}"),
    })
    st.dataframe(disp, use_container_width=True, hide_index=True,
                  column_config=SCAN_CFG)

    # Let user select a configuration
    options = [f"LEAPS {r['lc_k']:.0f}C ({r['leaps_dte']}d, Δ{r['delta']:.2f}) "
               f"+ SC {r['sc_k']:.0f}C → ROC {r['roc']:+.0f}%"
               for _, r in df.head(10).iterrows()]
    if not options:
        return
    idx = st.selectbox("Select configuration", range(len(options)),
                        format_func=lambda i: options[i], key="sr_sel")

    pick = df.iloc[idx]

    # Build full PMCC for selected config
    chain_l = res["chain_by_exp"].get(pick["exp"])
    if chain_l is None:
        st.error("Chain not found for selected expiration.")
        return

    smile_l = build_smile_curve(chain_l["calls"], spot)
    pmcc = build_pmcc(spot, res["r"], res["q"], res["iv"],
                       5 if spot < 500 else (1 if spot < 50 else 25),
                       target, int(pick["leaps_dte"]),
                       float(pick["delta"]),
                       int(res["sc_dte_actual"]),
                       0.30,  # sc_delta
                       smile_l, res["sc_smile"])

    if pmcc is None:
        st.error("Could not build PMCC for this configuration.")
        return

    # Position details
    st.markdown("### Position")
    leg_rows = [
        {"Leg": "Long Call (LEAPS)", "Strike": f"${pmcc['lc_k']:,.0f}",
         "Delta": f"{pmcc['lc']['delta']:.3f}",
         "IV": f"{pmcc['iv_lc']*100:.1f}%",
         "Price": f"${pmcc['leaps_cost']*100:,.0f}",
         "DTE": pmcc["leaps_dte"]},
        {"Leg": "Short Call (monthly)", "Strike": f"${pmcc['sc_k']:,.0f}",
         "Delta": f"{pmcc['sc']['delta']:.3f}",
         "IV": f"{pmcc['iv_sc']*100:.1f}%",
         "Price": f"${pmcc['sc_premium']*100:,.0f}",
         "DTE": pmcc["sc_dte"]},
    ]
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # Metrics
    st.markdown("### Economics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("LEAPS Cost", f"${pmcc['leaps_cost']*100:,.0f}",
              help="Capital required (vs stock: " +
                   f"${spot*100:,.0f}). Saves ${(spot-pmcc['leaps_cost'])*100:,.0f}.")
    m2.metric("Income/Cycle", f"${pmcc['income_per_cycle']*100:,.0f}",
              help="Short call 50% profit per cycle")
    m3.metric("Cycles", f"{pmcc['cycles_possible']}",
              help="Short call cycles before LEAPS hits 45 DTE")
    m4.metric("Total Income", f"${pmcc['total_income']*100:,.0f}",
              help="Projected total income from all cycles")
    m5.metric("Net Cost", f"${pmcc['net_cost']*100:,.0f}",
              help="LEAPS cost minus total income. Your real cost basis.")
    m6.metric("Break-Even", f"${pmcc['breakeven']:,.1f}",
              delta=f"{(pmcc['breakeven']/spot-1)*100:+.1f}% from spot",
              delta_color="normal" if pmcc['breakeven'] <= spot else "inverse",
              help="Stock price needed at LEAPS expiry to break even")

    # Comparison: PMCC vs Stock
    st.markdown("### PMCC vs 100 Shares")
    comp = [
        {"": "Capital", "PMCC": f"${pmcc['leaps_cost']*100:,.0f}",
         "100 Shares": f"${spot*100:,.0f}",
         "Savings": f"${(spot-pmcc['leaps_cost'])*100:,.0f} "
                    f"({(1-pmcc['leaps_cost']/spot)*100:.0f}%)"},
        {"": "Delta", "PMCC": f"{pmcc['lc']['delta']:.2f}",
         "100 Shares": "1.00", "Savings": ""},
        {"": "Monthly Income", "PMCC": f"${pmcc['income_per_cycle']*100:,.0f}",
         "100 Shares": "$0", "Savings": f"+${pmcc['income_per_cycle']*100:,.0f}/mo"},
        {"": f"P&L @ ${target:,.0f}", "PMCC": f"${pmcc['pnl_at_target']*100:+,.0f}",
         "100 Shares": f"${pmcc['pnl_stock']*100:+,.0f}",
         "Savings": f"ROC {pmcc['capital_efficiency']:+.0f}% vs "
                    f"{pmcc['pnl_stock']/spot*100:+.0f}%"},
        {"": "Break-Even", "PMCC": f"${pmcc['breakeven']:,.1f}",
         "100 Shares": f"${spot:,.2f}", "Savings": ""},
        {"": "Max Loss", "PMCC": f"${pmcc['net_cost']*100:,.0f}",
         "100 Shares": f"${spot*100:,.0f}", "Savings": ""},
    ]
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    # Scenario chart
    st.markdown("### Scenario Analysis")
    import plotly.graph_objects as go

    scen = pmcc["scenarios"]
    spots = [s["spot"] for s in scen]
    pmcc_pnls = [s["pmcc_pnl"] for s in scen]
    stock_pnls = [s["stock_pnl"] for s in scen]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=pmcc_pnls, mode="lines+markers",
        name="PMCC", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=spots, y=stock_pnls, mode="lines+markers",
        name="100 Shares", line=dict(color="#aaaaaa", width=1, dash="dash")))
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
        {"strike": int(pmcc["lc_k"]), "option_type": "call",
         "expiration": str(pick["exp"]), "long": True, "qty": 1,
         "price": float(pmcc["leaps_cost"])},
        {"strike": int(pmcc["sc_k"]), "option_type": "call",
         "expiration": str(res["sc_exp"]), "long": False, "qty": 1,
         "price": float(pmcc["sc_premium"])},
    ]

    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], os_legs)
        if url:
            st.markdown(f"[OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], os_legs, tag="StockRepl")
        clean = res["symbol"].replace("^", "")
        st.download_button("IBKR Basket CSV", csv,
                            f"stockrepl_{clean}.csv", "text/csv")

    with st.expander("Strategy Guide"):
        st.markdown(
            f"**Setup:**\n"
            f"- Buy LEAPS {pmcc['lc_k']:.0f}C ({pmcc['leaps_dte']}d): "
            f"${pmcc['leaps_cost']*100:,.0f}\n"
            f"- Sell {pmcc['sc_k']:.0f}C ({pmcc['sc_dte']}d): "
            f"${pmcc['sc_premium']*100:,.0f} income\n\n"
            f"**Monthly Cycle:**\n"
            f"1. When short call reaches 50% profit → close it\n"
            f"2. Sell new OTM call at ~30 delta with fresh DTE\n"
            f"3. Repeat up to {pmcc['cycles_possible']} times\n\n"
            f"**If stock rallies past short call:**\n"
            f"- Roll the short call up and out (higher strike, further DTE)\n"
            f"- Or close the short call at a loss, let LEAPS run\n\n"
            f"**If stock drops significantly:**\n"
            f"- Short call expires worthless → keep premium\n"
            f"- Sell another short call at lower strike for continued income\n"
            f"- LEAPS delta decreases, reducing downside exposure\n\n"
            f"**Exit:**\n"
            f"- At target ${target:,.0f}: close both legs, profit "
            f"${pmcc['pnl_at_target']*100:+,.0f}\n"
            f"- Or keep rolling shorts indefinitely as a synthetic covered call"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Stock Replacement")
    st.caption("Replace stock positions with PMCC (Poor Man's Covered Call). "
               "Same upside exposure at a fraction of the capital, plus monthly income.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="GOOGL",
            help="Stock to replace with options.").upper()
    with c2:
        target_price = st.number_input("Price Target $", value=207.0,
            min_value=1.0, max_value=10000.0, step=5.0, format="%.1f",
            help="Your price target for the stock. "
                 "Used to calculate P&L and ROC for each configuration.")
    with c3:
        sc_delta = st.number_input("SC Delta", value=0.30,
            min_value=0.15, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for the monthly short call. "
                 "Higher = more income but caps upside sooner. "
                 "0.25-0.35 is typical.")
    with c4:
        sc_dte = st.number_input("SC DTE", value=30,
            min_value=14, max_value=60,
            help="DTE for the short call. 30-45 days is optimal for theta.")

    default_step = 5 if target_price > 100 else 1
    step = default_step

    if st.button("Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning PMCC configurations..."):
            try:
                result = compute(symbol, step, target_price, sc_delta, sc_dte)
                st.session_state["sr_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "sr_result" not in st.session_state:
        st.info("Enter your stock and target, then click Scan.")
        return

    display(st.session_state["sr_result"])


main()
