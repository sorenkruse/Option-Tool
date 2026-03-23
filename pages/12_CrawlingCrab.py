"""
Crawling Crab – Trend-following strategy with rolling short premium.

Bull Crab (uptrend):
  Lower Leg: Bull Put Spread (SP ~30Δ + LP ~20Δ, short DTE)
  Upper Leg: Diagonal Call Spread (LC ~45Δ 90 DTE + SC ~30Δ short DTE)

Bear Crab (downtrend):
  Upper Leg: Bear Call Spread (SC ~30Δ + LC ~20Δ, short DTE)
  Lower Leg: Diagonal Put Spread (LP ~45Δ 90 DTE + SP ~30Δ short DTE)

Shorts are closed at 50% profit, then reopened. The long leg stays
and accumulates value as the trend continues.
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


# ── Build ────────────────────────────────────────────────────────────────

def build_crab(spot, r, q, iv_atm, step, direction,
               short_dte, long_dte,
               spread_delta_short, spread_delta_long,
               diag_delta_long, diag_delta_short,
               smile_put_s, smile_call_s, smile_put_l, smile_call_l):
    """
    Build a Crawling Crab position.

    direction: "Bull" or "Bear"
    Returns result dict or None.
    """
    T_s = short_dte / 365.0
    T_l = long_dte / 365.0

    if direction == "Bull":
        # Lower: Bull Put Spread (SP higher Δ, LP lower Δ, both short DTE)
        try:
            sp_k = _round(bs.solve_strike_for_delta(
                -spread_delta_short, spot, T_s, r, iv_atm, q, "put"), step)
            lp_k = _round(bs.solve_strike_for_delta(
                -spread_delta_long, spot, T_s, r, iv_atm, q, "put"), step)
        except Exception:
            return None
        if sp_k <= lp_k:
            return None

        # Upper: Diagonal Call (LC long DTE, SC short DTE)
        try:
            lc_k = _round(bs.solve_strike_for_delta(
                diag_delta_long, spot, T_l, r, iv_atm, q, "call"), step)
            sc_k = _round(bs.solve_strike_for_delta(
                diag_delta_short, spot, T_s, r, iv_atm, q, "call"), step)
        except Exception:
            return None

        # Price all legs
        iv_sp = _iv(smile_put_s, sp_k, iv_atm)
        iv_lp = _iv(smile_put_s, lp_k, iv_atm)
        iv_lc = _iv(smile_call_l, lc_k, iv_atm)
        iv_sc = _iv(smile_call_s, sc_k, iv_atm)

        sp = bs.calculate_all(spot, sp_k, T_s, r, iv_sp, q, "put")
        lp = bs.calculate_all(spot, lp_k, T_s, r, iv_lp, q, "put")
        lc = bs.calculate_all(spot, lc_k, T_l, r, iv_lc, q, "call")
        sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, "call")

        legs = [
            {"label": "Short Put", "type": "put", "strike": sp_k, "dte": short_dte,
             "long": False, "res": sp, "iv": iv_sp},
            {"label": "Long Put", "type": "put", "strike": lp_k, "dte": short_dte,
             "long": True, "res": lp, "iv": iv_lp},
            {"label": "Long Call", "type": "call", "strike": lc_k, "dte": long_dte,
             "long": True, "res": lc, "iv": iv_lc},
            {"label": "Short Call", "type": "call", "strike": sc_k, "dte": short_dte,
             "long": False, "res": sc, "iv": iv_sc},
        ]
        spread_label = "Bull Put"
        diag_label = "Diagonal Call"
        core_leg = lc
        core_label = "Long Call"

    else:  # Bear
        # Upper: Bear Call Spread (SC higher Δ, LC lower Δ, both short DTE)
        try:
            sc_k = _round(bs.solve_strike_for_delta(
                spread_delta_short, spot, T_s, r, iv_atm, q, "call"), step)
            lc_k = _round(bs.solve_strike_for_delta(
                spread_delta_long, spot, T_s, r, iv_atm, q, "call"), step)
        except Exception:
            return None
        if lc_k <= sc_k:
            return None

        # Lower: Diagonal Put (LP long DTE, SP short DTE)
        try:
            lp_k = _round(bs.solve_strike_for_delta(
                -diag_delta_long, spot, T_l, r, iv_atm, q, "put"), step)
            sp_k = _round(bs.solve_strike_for_delta(
                -diag_delta_short, spot, T_s, r, iv_atm, q, "put"), step)
        except Exception:
            return None

        iv_sc = _iv(smile_call_s, sc_k, iv_atm)
        iv_lc = _iv(smile_call_s, lc_k, iv_atm)
        iv_lp = _iv(smile_put_l, lp_k, iv_atm)
        iv_sp = _iv(smile_put_s, sp_k, iv_atm)

        sc = bs.calculate_all(spot, sc_k, T_s, r, iv_sc, q, "call")
        lc = bs.calculate_all(spot, lc_k, T_s, r, iv_lc, q, "call")
        lp = bs.calculate_all(spot, lp_k, T_l, r, iv_lp, q, "put")
        sp = bs.calculate_all(spot, sp_k, T_s, r, iv_sp, q, "put")

        legs = [
            {"label": "Short Call", "type": "call", "strike": sc_k, "dte": short_dte,
             "long": False, "res": sc, "iv": iv_sc},
            {"label": "Long Call", "type": "call", "strike": lc_k, "dte": short_dte,
             "long": True, "res": lc, "iv": iv_lc},
            {"label": "Long Put", "type": "put", "strike": lp_k, "dte": long_dte,
             "long": True, "res": lp, "iv": iv_lp},
            {"label": "Short Put", "type": "put", "strike": sp_k, "dte": short_dte,
             "long": False, "res": sp, "iv": iv_sp},
        ]
        spread_label = "Bear Call"
        diag_label = "Diagonal Put"
        core_leg = lp
        core_label = "Long Put"

    # Aggregates
    delta = sum((-1 if not l["long"] else 1) * l["res"]["delta"] for l in legs)
    theta = sum((-1 if not l["long"] else 1) * l["res"]["theta_daily"] for l in legs)
    vega = sum((-1 if not l["long"] else 1) * l["res"]["vega_pct"] for l in legs)

    long_cost = sum(l["res"]["price"] for l in legs if l["long"])
    short_income = sum(l["res"]["price"] for l in legs if not l["long"])
    net_debit = long_cost - short_income
    core_price = core_leg["price"]

    # The 3 short-DTE legs (2 shorts + 1 long in spread) that get cycled
    short_dte_legs = [l for l in legs if l["dte"] == short_dte]
    # Net credit from these 3 legs: sold premium minus bought premium
    cycle_credit = sum(
        l["res"]["price"] * (1 if not l["long"] else -1)
        for l in short_dte_legs
    )

    # Theta from short-DTE legs only (what drives the 50% target)
    theta_short_dte = sum(
        (-1 if not l["long"] else 1) * l["res"]["theta_daily"]
        for l in short_dte_legs
    )

    # Cycle economics: close all 3 short-DTE legs at 50% profit
    profit_per_cycle = cycle_credit * 0.50
    cycles_to_fund = core_price / profit_per_cycle if profit_per_cycle > 0 else 99
    # Days to 50%: based on theta of short-DTE legs only
    days_to_50 = profit_per_cycle / theta_short_dte if theta_short_dte > 0 else 99

    return {
        "legs": legs,
        "direction": direction,
        "spread_label": spread_label,
        "diag_label": diag_label,
        "core_label": core_label,
        "core_price": core_price,
        "delta": delta, "theta": theta, "vega": vega,
        "short_income": short_income,
        "long_cost": long_cost,
        "net_debit": net_debit,
        "cycle_credit": cycle_credit,
        "profit_per_cycle": profit_per_cycle,
        "cycles_to_fund": cycles_to_fund,
        "days_to_50": days_to_50,
        "short_dte": short_dte, "long_dte": long_dte,
        "n_short_dte_legs": len(short_dte_legs),
    }


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, step, direction, short_dte, long_dte,
            spread_d_short, spread_d_long, diag_d_long, diag_d_short,
            auto_mode=False, min_long_dte=60):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(short_dte if short_dte > 0 else 14)
    vix = get_vix()
    iv = vix / 100.0 if not np.isnan(vix) else 0.20
    today = datetime.date.today()

    if auto_mode:
        try:
            exps, _ = get_available_expirations(symbol)
        except Exception as e:
            raise ValueError(f"Could not fetch expirations: {e}")
        if not exps:
            raise ValueError(f"No expirations for {symbol}.")

        # Load chains by unique expiration
        chain_by_exp = {}
        for exp_str in exps:
            try:
                d = (datetime.date.fromisoformat(exp_str) - today).days
                if 5 <= d <= 180:
                    ch, _, _ = resolve_options_chain(symbol, d)
                    actual_exp = ch["expiration"]
                    if actual_exp not in chain_by_exp:
                        chain_by_exp[actual_exp] = ch
            except Exception:
                continue

        if not chain_by_exp:
            raise ValueError("Could not load option chains.")

        all_exps = sorted(chain_by_exp.keys())
        combos = []

        for s_exp in all_exps:
            cs = chain_by_exp[s_exp]
            s_dte = cs["dte_actual"]
            if s_dte < 7 or s_dte > 30:
                continue

            sm_ps = build_smile_curve(cs["puts"], spot)
            sm_cs = build_smile_curve(cs["calls"], spot)

            for l_exp in all_exps:
                cl = chain_by_exp[l_exp]
                l_dte = cl["dte_actual"]
                if l_dte < max(45, min_long_dte) or l_dte <= s_dte * 2:
                    continue

                sm_pl = build_smile_curve(cl["puts"], spot)
                sm_cl = build_smile_curve(cl["calls"], spot)

                res = build_crab(spot, r, q, iv, step, direction,
                                  s_dte, l_dte,
                                  spread_d_short, spread_d_long,
                                  diag_d_long, diag_d_short,
                                  sm_ps, sm_cs, sm_pl, sm_cl)
                if res is None:
                    continue
                if res["cycles_to_fund"] <= 0 or res["days_to_50"] <= 0:
                    continue

                res["symbol"] = symbol
                res["spot"] = spot
                res["vix"] = vix
                res["r"] = r
                res["q"] = q
                res["exp_short"] = cs["expiration"]
                res["exp_long"] = cl["expiration"]

                weeks = res["cycles_to_fund"] * res["days_to_50"] / 7
                days_needed = res["cycles_to_fund"] * res["days_to_50"]
                remaining_after_cycles = l_dte - days_needed
                # Skip if long would be in heavy theta decay zone after funding
                if remaining_after_cycles < min_long_dte * 0.5:
                    continue

                combos.append({
                    "short_dte": s_dte,
                    "long_dte": l_dte,
                    "cycle_credit": res["cycle_credit"],
                    "days_to_50": res["days_to_50"],
                    "cycles": res["cycles_to_fund"],
                    "weeks": weeks,
                    "days_needed": days_needed,
                    "remaining": remaining_after_cycles,
                    "net_debit": res["net_debit"],
                    "theta": res["theta"],
                    "delta": res["delta"],
                    "_full": dict(res),
                })

        if not combos:
            raise ValueError("No valid configurations found.")

        combos.sort(key=lambda x: x["weeks"])
        best = combos[0]["_full"]
        best["dte_scan"] = combos[:10]
        return best

    else:
        ch_s, _, _ = resolve_options_chain(symbol, short_dte)
        ch_l, _, _ = resolve_options_chain(symbol, long_dte)

        sm_ps = build_smile_curve(ch_s["puts"], spot)
        sm_cs = build_smile_curve(ch_s["calls"], spot)
        sm_pl = build_smile_curve(ch_l["puts"], spot)
        sm_cl = build_smile_curve(ch_l["calls"], spot)

        result = build_crab(spot, r, q, iv, step, direction,
                             ch_s["dte_actual"], ch_l["dte_actual"],
                             spread_d_short, spread_d_long,
                             diag_d_long, diag_d_short,
                             sm_ps, sm_cs, sm_pl, sm_cl)

        if result is None:
            raise ValueError("Could not construct Crawling Crab.")

        result["symbol"] = symbol
        result["spot"] = spot
        result["vix"] = vix
        result["r"] = r
        result["q"] = q
        result["exp_short"] = ch_s["expiration"]
        result["exp_long"] = ch_l["expiration"]
        result["dte_scan"] = None
        return result


# ── Display ──────────────────────────────────────────────────────────────

LEG_CFG = {
    "Leg": st.column_config.TextColumn("Leg", help="Position type: Short/Long Put/Call"),
    "Strike": st.column_config.TextColumn("Strike", help="Option strike price"),
    "Delta": st.column_config.TextColumn("Delta", help="Option delta at entry"),
    "IV": st.column_config.TextColumn("IV", help="Implied volatility from smile curve"),
    "Price": st.column_config.TextColumn("Price", help="Option premium per point (×100 for contract value)"),
    "DTE": st.column_config.NumberColumn("DTE", help="Days to expiration"),
    "Exp": st.column_config.TextColumn("Exp", help="Expiration date"),
    "Role": st.column_config.TextColumn("Role", help="Which part of the strategy this leg serves"),
}


def display(res):
    spot = res["spot"]
    direction = res["direction"]
    name = f"{'Bull' if direction == 'Bull' else 'Bear'} Crab"

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}  |  {name}")

    # Structure table
    st.markdown("### Structure")
    leg_rows = []
    for l in res["legs"]:
        role = res["spread_label"] if l["dte"] == res["short_dte"] and (
            (direction == "Bull" and l["type"] == "put") or
            (direction == "Bear" and l["type"] == "call")
        ) else res["diag_label"]
        leg_rows.append({
            "Leg": l["label"],
            "Strike": f"{l['strike']:,.0f}",
            "Delta": f"{l['res']['delta']:.3f}",
            "IV": f"{l['iv']*100:.1f}%",
            "Price": f"${l['res']['price']:.2f}",
            "DTE": l["dte"],
            "Exp": res["exp_short"] if l["dte"] == res["short_dte"] else res["exp_long"],
            "Role": role,
        })
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True,
                  hide_index=True, column_config=LEG_CFG)

    # Key metrics
    st.markdown("### Position Metrics")
    days_needed = res["cycles_to_fund"] * res["days_to_50"]
    remaining_dte = res["long_dte"] - days_needed

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Net Debit", f"${res['net_debit']*100:,.0f}",
              help="Total cost to open: long premiums minus short premiums (×100)")
    m2.metric("Delta", f"{res['delta']:.3f}",
              help="Net position delta. Bull Crab: positive. Bear Crab: negative.")
    m3.metric("Theta/day", f"${res['theta']*100:+,.0f}",
              help="Daily theta income (×100). Positive = earning from short decay.")
    m4.metric("Days to 50%", f"{res['days_to_50']:.0f}d",
              help="Estimated days until short-DTE legs reach 50% of entry credit.")
    m5.metric(f"{res['core_label']} Cost", f"${res['core_price']*100:,.0f}",
              help=f"Cost of the core long-dated {res['core_label']} that stays open.")
    m6.metric("DTE after Funding", f"{remaining_dte:.0f}d",
              delta="safe" if remaining_dte > 45 else "at risk",
              delta_color="normal" if remaining_dte > 45 else "inverse",
              help="Remaining DTE on the core long after all funding cycles complete. "
                   "Below 45d → theta accelerates, reducing the long's value.")

    if remaining_dte < 30:
        st.warning(
            f"The core {res['core_label']} will only have ~{remaining_dte:.0f} days "
            f"remaining after {res['cycles_to_fund']:.0f} funding cycles. "
            f"Consider a longer Long DTE or fewer cycles needed."
        )

    # Cycle economics
    st.markdown("### Cycle Economics")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cycle Credit", f"${res['cycle_credit']*100:,.0f}",
              help="Net credit from the 3 short-DTE legs (2 shorts + 1 long in spread). "
                   "This is the total premium you collect per cycle.")
    e2.metric("50% Profit/Cycle", f"${res['profit_per_cycle']*100:,.0f}",
              help="Close all 3 short-DTE legs when they've decayed to 50% of entry credit.")
    e3.metric("Cycles to Fund", f"{res['cycles_to_fund']:.1f}",
              help=f"Number of cycles to fully pay for the {res['core_label']}")
    e4.metric("Weeks to Fund", f"{res['cycles_to_fund'] * res['days_to_50'] / 7:.1f}",
              help="Estimated weeks until the core long is fully financed")

    n = res.get("n_short_dte_legs", 3)
    st.caption(
        f"Each cycle: open {n} short-DTE legs (2 shorts + 1 long in spread) → "
        f"wait ~{res['days_to_50']:.0f} days → "
        f"close all {n} at 50% profit (${res['profit_per_cycle']*100:,.0f}) → repeat. "
        f"After ~{res['cycles_to_fund']:.0f} cycles, the {res['core_label']} is fully paid for."
    )

    # OptionStrat + IBKR
    st.markdown("### Export")
    os_legs = []
    for l in res["legs"]:
        os_legs.append({
            "strike": int(l["strike"]),
            "option_type": l["type"],
            "expiration": res["exp_short"] if l["dte"] == res["short_dte"] else res["exp_long"],
            "long": l["long"],
            "qty": 1,
            "price": float(l["res"]["price"]),
        })

    os_sorted = sorted(os_legs, key=lambda x: (x["long"], x["option_type"] != "put"))

    e1, e2 = st.columns(2)
    with e1:
        url = bs.optionstrat_url(res["symbol"], os_sorted)
        if url:
            st.markdown(f"[Full Position: OptionStrat]({url})")
    with e2:
        csv = bs.ibkr_basket_csv(res["symbol"], os_sorted, tag="CrawlingCrab")
        clean_sym = res["symbol"].replace("^", "")
        st.download_button("IBKR Basket CSV", csv,
                            f"crab_{clean_sym}.csv", "text/csv")

    # Separate links: spread + diagonal
    spread_legs = [l for l in os_legs
                   if (direction == "Bull" and l["option_type"] == "put") or
                      (direction == "Bear" and l["option_type"] == "call")]
    diag_legs = [l for l in os_legs
                 if (direction == "Bull" and l["option_type"] == "call") or
                    (direction == "Bear" and l["option_type"] == "put")]

    c1, c2 = st.columns(2)
    with c1:
        url_sp = bs.optionstrat_url(res["symbol"], spread_legs)
        if url_sp:
            st.markdown(f"[{res['spread_label']} Spread]({url_sp})")
    with c2:
        url_diag = bs.optionstrat_url(res["symbol"], diag_legs)
        if url_diag:
            st.markdown(f"[{res['diag_label']}]({url_diag})")

    # Management guide
    with st.expander("Management Playbook"):
        st.markdown(
            f"**Entry:**\n"
            f"- Open all 4 legs simultaneously\n"
            f"- Net debit: ${res['net_debit']*100:,.0f}\n\n"
            f"**Cycle Management (repeat every ~{res['days_to_50']:.0f} days):**\n"
            f"1. Monitor the 3 short-DTE legs ({res['spread_label']} spread + "
            f"{res['diag_label']} short leg)\n"
            f"2. When the combined credit decays to 50% → close all 3 short-DTE legs\n"
            f"3. The {res['core_label']} ({res['long_dte']} DTE) stays open\n"
            f"4. Open 3 new short-DTE legs at current delta targets\n\n"
            f"**{res['core_label']} Management:**\n"
            f"- If still OTM after several cycles: keep rolling short-DTE legs against it\n"
            f"- If approaching ATM/ITM: let it run in profit, buy a new {res['core_label']} "
            f"further along the trend\n"
            f"- Goal: accumulate multiple ITM {res['core_label']}s over weeks/months\n\n"
            f"**Risk:**\n"
            f"- Trend reversal: {res['core_label']} loses value, short-DTE legs may get tested\n"
            f"- Max loss on spread: "
            f"${abs(res['legs'][0]['strike'] - res['legs'][1]['strike'])*100:,.0f} per contract\n"
            f"- {res['core_label']} has defined risk (premium paid: "
            f"${res['core_price']*100:,.0f})"
        )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Crawling Crab")
    st.caption("Trend-following strategy: long-dated directional bet "
               "financed by rolling short premium cycles.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
            help="Underlying to trade.").upper()
    with c2:
        direction = st.selectbox("Direction", ["Bull", "Bear"],
            help="Bull Crab: uptrend (Bull Put Spread + Diagonal Call). "
                 "Bear Crab: downtrend (Bear Call Spread + Diagonal Put).")
    with c3:
        default_step = 25 if "SPX" in symbol else 5
        step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50,
            help="Strike increment for rounding.")
    with c4:
        mode = st.radio("Mode", ["Auto", "Manual DTEs"], horizontal=True,
            help="Auto scans all DTE combinations and finds the one "
                 "that minimizes weeks to fund the core long.")

    if mode == "Manual DTEs":
        st.markdown("#### DTEs")
        dt1, dt2, dt3 = st.columns(3)
        with dt1:
            short_dte = st.number_input("Short DTE", value=14,
                min_value=7, max_value=30,
                help="DTE for the 3 short-DTE legs. 10-20 days is ideal for theta.")
        with dt2:
            long_dte = st.number_input("Long DTE", value=90,
                min_value=45, max_value=180,
                help="DTE for the core long leg. 90 days gives time for the trend.")
        with dt3:
            min_long_dte = st.number_input("Min Long DTE", value=60,
                min_value=30, max_value=120,
                help="Minimum acceptable DTE for the core long. "
                     "The long must have enough time to survive all funding cycles "
                     "without entering heavy theta decay (typically below 45 DTE).")
    else:
        short_dte = long_dte = 0
        min_long_dte = st.number_input("Min Long DTE", value=60,
            min_value=30, max_value=120,
            help="Minimum acceptable DTE for the core long. "
                 "Auto mode filters out combinations where the long DTE is too short "
                 "to survive the funding cycles. "
                 "Rule: Long DTE must be > cycles × days per cycle + this buffer.")

    st.markdown("#### Delta Targets")
    if direction == "Bull":
        st.caption("Bull Crab: Bull Put Spread (lower) + Diagonal Call (upper)")
    else:
        st.caption("Bear Crab: Bear Call Spread (upper) + Diagonal Put (lower)")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        spread_d_short = st.number_input(
            "Spread Short Δ", value=0.30,
            min_value=0.15, max_value=0.50, step=0.05, format="%.2f",
            help="Delta of the short leg in the credit spread. "
                 "Higher = more premium but closer to ATM (more risk).")
    with d2:
        spread_d_long = st.number_input(
            "Spread Long Δ", value=0.20,
            min_value=0.05, max_value=0.40, step=0.05, format="%.2f",
            help="Delta of the long (protective) leg in the credit spread. "
                 "Lower = cheaper protection, wider spread.")
    with d3:
        diag_d_long = st.number_input(
            "Diagonal Long Δ", value=0.45,
            min_value=0.25, max_value=0.60, step=0.05, format="%.2f",
            help="Delta of the core long-dated leg (the trend bet). "
                 "Higher = more expensive but higher delta exposure. "
                 "This leg stays open across multiple cycles.")
    with d4:
        diag_d_short = st.number_input(
            "Diagonal Short Δ", value=0.30,
            min_value=0.15, max_value=0.50, step=0.05, format="%.2f",
            help="Delta of the short leg in the diagonal spread. "
                 "Premium received partially finances the long leg.")

    if st.button(f"Build {'Bull' if direction == 'Bull' else 'Bear'} Crab",
                  type="primary", use_container_width=True):
        with st.spinner("Building position..."):
            try:
                result = compute(symbol, step, direction,
                                  short_dte, long_dte,
                                  spread_d_short, spread_d_long,
                                  diag_d_long, diag_d_short,
                                  auto_mode=(mode != "Manual DTEs"),
                                  min_long_dte=min_long_dte)
                st.session_state["crab_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "crab_result" not in st.session_state:
        st.info("Set parameters and click Build.")
        return

    result = st.session_state["crab_result"]
    scan = result.get("dte_scan")

    if scan and len(scan) > 1:
        st.markdown("---")
        st.markdown(f"### {result['symbol']} @ {result['spot']:,.2f}  |  "
                    f"VIX {result['vix']:.1f}")
        st.markdown("### DTE Optimization")
        st.caption("Sorted by fewest weeks to fund the core long position.")

        DTE_CFG = {
            "#": st.column_config.NumberColumn("#", help="Rank"),
            "Short": st.column_config.NumberColumn("Short", help="Short-DTE legs"),
            "Long": st.column_config.NumberColumn("Long", help="Core long DTE"),
            "Credit/Cyc": st.column_config.TextColumn("Credit/Cyc",
                help="Net credit from 3 short-DTE legs per cycle (×100)"),
            "Days50": st.column_config.TextColumn("Days50",
                help="Estimated days until short-DTE legs reach 50% profit"),
            "Cycles": st.column_config.TextColumn("Cycles",
                help="Number of cycles to fully fund the core long"),
            "Weeks": st.column_config.TextColumn("Weeks",
                help="Total weeks to fund = cycles × days50 / 7"),
            "Remaining": st.column_config.TextColumn("Remaining",
                help="DTE remaining on core long after all funding cycles complete. "
                     "Should be > 30d to avoid theta acceleration."),
            "Net Debit": st.column_config.TextColumn("Net Debit",
                help="Initial net cost of the full 4-leg position (×100)"),
            "Delta": st.column_config.TextColumn("Delta",
                help="Net position delta"),
        }

        rows = []
        for i, s in enumerate(scan):
            rows.append({
                "#": i + 1,
                "Short": s["short_dte"],
                "Long": s["long_dte"],
                "Credit/Cyc": f"${s['cycle_credit']*100:,.0f}",
                "Days50": f"{s['days_to_50']:.0f}",
                "Cycles": f"{s['cycles']:.1f}",
                "Weeks": f"{s['weeks']:.1f}",
                "Remaining": f"{s['remaining']:.0f}d",
                "Net Debit": f"${s['net_debit']*100:,.0f}",
                "Delta": f"{s['delta']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                      hide_index=True, column_config=DTE_CFG)

        options = [f"#{i+1}: {s['short_dte']}/{s['long_dte']}d "
                   f"({s['weeks']:.1f} weeks, {s['cycles']:.1f} cycles)"
                   for i, s in enumerate(scan)]
        idx = st.selectbox("Result", range(len(options)),
                            format_func=lambda i: options[i],
                            key="crab_sel")
        active = scan[idx]["_full"]
    else:
        active = result

    display(active)


main()
