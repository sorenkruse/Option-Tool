"""
Verticals – Forecast-driven Vertical Spread Scanner

Scans all 4 vertical spread types across strikes and DTEs:
- Bull Put Spread (credit): SP higher + LP lower → flat/rally
- Bear Call Spread (credit): SC lower + LC higher → flat/drop
- Bull Call Spread (debit): LC lower + SC higher → rally
- Bear Put Spread (debit): LP higher + SP lower → drop

Uses same forecast model as Prognose (3 scenarios, conviction weighting).
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
    pct_move = (spot_target - spot_now) / spot_now
    if pct_move < 0:
        iv_shift = -pct_move * 0.40
    else:
        iv_shift = -pct_move * 0.25
    return max(iv_now + iv_shift, 0.05)


def calc_pop_spread(spot, k_short, k_long, T, r, iv, q, spread_type):
    """
    Probability of Profit for a vertical spread.

    For credit spreads: P(short leg expires OTM)
    For debit spreads: P(spot moves past break-even)
    """
    from scipy.stats import norm
    if T <= 0 or iv <= 0:
        return 0.0

    if spread_type == "Bull Put":
        # Credit: profit if spot stays above short put
        d2 = (np.log(spot / k_short) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        return norm.cdf(d2)
    elif spread_type == "Bear Call":
        # Credit: profit if spot stays below short call
        d2 = (np.log(spot / k_short) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        return norm.cdf(-d2)
    elif spread_type == "Bull Call":
        # Debit: profit if spot rises above long call + debit
        d2 = (np.log(spot / k_long) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        return norm.cdf(d2) * 0.8  # rough adjustment for break-even
    elif spread_type == "Bear Put":
        # Debit: profit if spot drops below long put - debit
        d2 = (np.log(spot / k_long) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        return norm.cdf(-d2) * 0.8
    return 0.5


# ── Scan ─────────────────────────────────────────────────────────────────

def scan_verticals(spot, r, q, iv_atm, target_spot, target_iv,
                   forecast_dte, strike_step, expirations_data,
                   scan_dtes, conviction, widths, lp_dte_offset_pct=20):
    """Scan vertical spreads across all strikes, DTEs, and widths.
    Long legs get +lp_dte_offset_pct% more DTE for better vega protection."""
    w_full = 0.15 + 0.45 * conviction
    w_half = 0.25 + 0.10 * conviction
    w_flat = 1.0 - w_full - w_half

    half_spot = spot + (target_spot - spot) * 0.5
    half_iv = project_iv(iv_atm, spot, half_spot)
    flat_iv = max(iv_atm - 0.005, 0.05)

    strike_low = _round(spot * 0.92, strike_step)
    strike_high = _round(spot * 1.08, strike_step)
    strikes = np.arange(strike_low, strike_high + strike_step, strike_step)

    # Map DTEs to expirations for finding long-leg chain
    dte_to_exp = {}
    for exp_str, ch in expirations_data.items():
        dte_to_exp[ch["dte_actual"]] = (exp_str, ch)

    results = []

    for exp_str, chain_data in expirations_data.items():
        dte = chain_data["dte_actual"]
        if dte not in scan_dtes:
            continue

        call_smile = build_smile_curve(chain_data["calls"], spot)
        put_smile = build_smile_curve(chain_data["puts"], spot)

        remaining_dte = max(dte - forecast_dte, 1)
        T_entry = dte / 365.0
        T_exit = remaining_dte / 365.0

        # Long leg: find next available chain with >= +20% more DTE
        long_dte_min = round(dte * (1 + lp_dte_offset_pct / 100))
        # Find the closest DTE that is strictly > dte (next expiration out)
        candidates = [d for d in dte_to_exp.keys() if d >= long_dte_min and d > dte]
        if candidates:
            closest_long_dte = min(candidates)
            long_exp_str, long_chain = dte_to_exp[closest_long_dte]
            long_call_smile = build_smile_curve(long_chain["calls"], spot)
            long_put_smile = build_smile_curve(long_chain["puts"], spot)
            long_dte_actual = closest_long_dte
            T_entry_long = long_dte_actual / 365.0
            remaining_long = max(long_dte_actual - forecast_dte, 1)
            T_exit_long = remaining_long / 365.0
        else:
            # No offset chain available, use same DTE
            long_call_smile = call_smile
            long_put_smile = put_smile
            long_dte_actual = dte
            long_exp_str = exp_str
            T_entry_long = T_entry
            T_exit_long = T_exit

        for width in widths:
            offset = strike_step * width

            for k_anchor in strikes:
                # ── 4 spread types ──

                spreads = []

                # Bull Put: SP at k_anchor (higher), LP at k_anchor - offset
                k_sp, k_lp = k_anchor, k_anchor - offset
                if k_sp < spot and k_sp <= spot * 0.98 and k_lp > spot * 0.85:
                    spreads.append(("Bull Put", "put", k_sp, k_lp, False))

                # Bear Call: SC at k_anchor (lower), LC at k_anchor + offset
                k_sc, k_lc = k_anchor, k_anchor + offset
                if k_sc > spot and k_sc >= spot * 1.02 and k_lc < spot * 1.15:
                    spreads.append(("Bear Call", "call", k_sc, k_lc, False))

                # Bull Call: LC at k_anchor (lower), SC at k_anchor + offset
                k_lc2, k_sc2 = k_anchor, k_anchor + offset
                if k_lc2 > spot and k_sc2 < spot * 1.15:
                    spreads.append(("Bull Call", "call", k_sc2, k_lc2, True))

                # Bear Put: LP at k_anchor (higher), SP at k_anchor - offset
                k_lp2, k_sp2 = k_anchor, k_anchor - offset
                if k_lp2 < spot and k_sp2 > spot * 0.85:
                    spreads.append(("Bear Put", "put", k_sp2, k_lp2, True))

                for spread_name, opt_type, k_short, k_long, is_debit in spreads:
                    smile = call_smile if opt_type == "call" else put_smile
                    long_smile = long_call_smile if opt_type == "call" else long_put_smile
                    iv_short = _iv(smile, k_short, iv_atm)
                    iv_long = _iv(long_smile, k_long, iv_atm)

                    try:
                        short_entry = bs.calculate_all(spot, k_short, T_entry, r, iv_short, q, opt_type)
                        long_entry = bs.calculate_all(spot, k_long, T_entry_long, r, iv_long, q, opt_type)
                    except Exception:
                        continue

                    if short_entry["price"] < 0.05 or long_entry["price"] < 0.05:
                        continue

                    # Net premium
                    credit = short_entry["price"] - long_entry["price"]
                    if is_debit:
                        net_cost = -credit  # debit = we pay
                        if net_cost <= 0:
                            continue
                    else:
                        net_cost = credit  # credit = we receive
                        if net_cost <= 0:
                            continue

                    max_loss = offset - abs(credit)
                    if max_loss <= 0:
                        continue

                    # Net delta
                    net_delta = -short_entry["delta"] + long_entry["delta"]

                    # Exit prices for 3 scenarios
                    iv_change_full = target_iv - iv_atm
                    iv_change_half = half_iv - iv_atm
                    iv_change_flat = flat_iv - iv_atm

                    def _exit(s_target, iv_shift_val, iv_s, iv_l):
                        iv_s_e = max(iv_s + iv_shift_val, 0.05)
                        iv_l_e = max(iv_l + iv_shift_val, 0.05)
                        try:
                            se = bs.calculate_all(s_target, k_short, T_exit, r, iv_s_e, q, opt_type)["price"]
                            le = bs.calculate_all(s_target, k_long, T_exit_long, r, iv_l_e, q, opt_type)["price"]
                            return se, le
                        except Exception:
                            return None, None

                    exits = {}
                    for sc_name, s_tgt, iv_ch in [("full", target_spot, iv_change_full),
                                                    ("half", half_spot, iv_change_half),
                                                    ("flat", spot, iv_change_flat)]:
                        se, le = _exit(s_tgt, iv_ch, iv_short, iv_long)
                        if se is None:
                            break
                        exit_credit = se - le
                        if is_debit:
                            pnl = exit_credit - (-credit)  # close spread - initial debit
                        else:
                            pnl = credit - exit_credit  # initial credit - close cost
                        exits[sc_name] = pnl

                    if len(exits) < 3:
                        continue

                    pnl_w = (w_full * exits["full"] + w_half * exits["half"]
                             + w_flat * exits["flat"])

                    pop = calc_pop_spread(spot, k_short, k_long, T_entry, r,
                                          iv_short, q, spread_name)
                    ev = pnl_w * pop

                    capital = net_cost if is_debit else max_loss
                    roc = pnl_w / capital if capital > 0 else 0

                    results.append({
                        "type": spread_name,
                        "k_short": k_short,
                        "k_long": k_long,
                        "width": offset,
                        "dte": dte,
                        "dte_long": long_dte_actual,
                        "exp": exp_str,
                        "exp_long": long_exp_str,
                        "opt_type": opt_type,
                        "credit": credit,
                        "net_cost": net_cost,
                        "max_loss": max_loss,
                        "pnl_full": exits["full"],
                        "pnl_half": exits["half"],
                        "pnl_flat": exits["flat"],
                        "pnl_w": pnl_w,
                        "pnl_pct": pnl_w / capital * 100 if capital > 0 else 0,
                        "pop": pop,
                        "ev": ev,
                        "roc": roc,
                        "delta": net_delta,
                        "is_debit": is_debit,
                    })

    return results


# ── Display ──────────────────────────────────────────────────────────────

# Column configs with tooltips
VERT_COL_CFG = {
    "Short K": st.column_config.TextColumn("Short K", help="Strike of the sold (short) leg"),
    "Long K": st.column_config.TextColumn("Long K", help="Strike of the bought (long) leg, defines risk boundary"),
    "Width": st.column_config.TextColumn("Width", help="Distance between strikes in points"),
    "S DTE": st.column_config.NumberColumn("S DTE", help="Short leg days to expiration"),
    "L DTE": st.column_config.NumberColumn("L DTE", help="Long leg DTE (+20% offset for vega protection)"),
    "Premium": st.column_config.TextColumn("Premium", help="Net credit received or debit paid (×100 per contract)"),
    "Full": st.column_config.TextColumn("Full", help="P&L if your full forecast move happens"),
    "Half": st.column_config.TextColumn("Half", help="P&L if only 50% of the move happens"),
    "Flat": st.column_config.TextColumn("Flat", help="P&L if spot stays flat (theta only)"),
    "Weighted": st.column_config.TextColumn("Weighted", help="Conviction-weighted avg: Full×w1 + Half×w2 + Flat×w3"),
    "W%": st.column_config.TextColumn("W%", help="Weighted P&L as % of capital at risk"),
    "Max Loss": st.column_config.TextColumn("Max Loss", help="Maximum possible loss per contract"),
    "PoP": st.column_config.TextColumn("PoP", help="Probability of Profit (BS-based)"),
    "EV": st.column_config.TextColumn("EV", help="Expected Value = Weighted P&L × PoP (risk-adjusted)"),
    "Delta": st.column_config.TextColumn("Delta", help="Net delta. Negative = bearish, positive = bullish"),
}

VERT_TOP_CFG = {
    "Type": st.column_config.TextColumn("Type", help="Spread type and credit/debit classification"),
    "Short": st.column_config.TextColumn("Short", help="Short leg strike"),
    "Long": st.column_config.TextColumn("Long", help="Long leg strike"),
    "Width": st.column_config.TextColumn("Width", help="Distance between strikes"),
    "S DTE": st.column_config.NumberColumn("S DTE", help="Short leg DTE"),
    "L DTE": st.column_config.NumberColumn("L DTE", help="Long leg DTE (+20% offset)"),
    "Premium": st.column_config.TextColumn("Premium", help="Net credit or debit per contract (×100)"),
    "Full": st.column_config.TextColumn("Full", help="P&L at full forecast move"),
    "Half": st.column_config.TextColumn("Half", help="P&L at half move"),
    "Flat": st.column_config.TextColumn("Flat", help="P&L if spot unchanged"),
    "Weighted": st.column_config.TextColumn("Weighted", help="Conviction-weighted P&L"),
    "PoP": st.column_config.TextColumn("PoP", help="Probability of Profit"),
    "EV": st.column_config.TextColumn("EV", help="Expected Value = Weighted × PoP"),
}


def _fmt_table(df_sub):
    d = df_sub[["k_short", "k_long", "width", "dte", "dte_long", "net_cost",
                 "pnl_full", "pnl_half", "pnl_flat", "pnl_w", "pnl_pct",
                 "max_loss", "pop", "ev", "delta"]].copy()
    d.columns = ["Short K", "Long K", "Width", "S DTE", "L DTE", "Premium",
                  "Full", "Half", "Flat", "Weighted", "W%",
                  "Max Loss", "PoP", "EV", "Delta"]
    d["Short K"]  = d["Short K"].map(lambda x: f"{x:,.0f}")
    d["Long K"]   = d["Long K"].map(lambda x: f"{x:,.0f}")
    d["Width"]    = d["Width"].map(lambda x: f"{x:,.0f}")
    d["Premium"]  = d["Premium"].map(lambda x: f"${x*100:,.0f}")
    d["Full"]     = d["Full"].map(lambda x: f"${x*100:+,.0f}")
    d["Half"]     = d["Half"].map(lambda x: f"${x*100:+,.0f}")
    d["Flat"]     = d["Flat"].map(lambda x: f"${x*100:+,.0f}")
    d["Weighted"] = d["Weighted"].map(lambda x: f"${x*100:+,.0f}")
    d["W%"]       = d["W%"].map(lambda x: f"{x:+.0f}%")
    d["Max Loss"] = d["Max Loss"].map(lambda x: f"${x*100:,.0f}")
    d["PoP"]      = d["PoP"].map(lambda x: f"{x*100:.0f}%")
    d["EV"]       = d["EV"].map(lambda x: f"${x*100:+,.0f}")
    d["Delta"]    = d["Delta"].map(lambda x: f"{x:.3f}")
    return d


def display(res):
    spot = res["spot"]
    target = res["target_spot"]
    move_pct = res["move_pct"]

    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Target", f"{target:,.0f}", f"{move_pct:+.1f}%",
              help="Projected spot = current × (1 + move%)")
    m2.metric("IV Now", f"{res['iv_now']*100:.1f}%",
              help="Current implied volatility from VIX")
    m3.metric("IV Projected", f"{res['iv_target']*100:.1f}%",
              f"{(res['iv_target']-res['iv_now'])*100:+.1f}%",
              help="Projected IV at target. Drops → IV rises, rallies → IV falls.")
    conv = res.get("conviction", 0.5)
    w_full = 0.15 + 0.45 * conv
    w_half = 0.25 + 0.10 * conv
    w_flat = 1.0 - w_full - w_half
    m4.metric("Conviction", f"{conv*100:.0f}%",
              help="Scenario weights: Full/Half/Flat")
    m5.metric("Spreads Scanned", f"{len(res['all_results']):,}")

    df = pd.DataFrame(res["all_results"])
    if df.empty:
        st.warning("No spreads found.")
        return

    # ── Top picks (1 per type) ──
    st.markdown("### Best Picks (Top 1 per Type)")
    top_picks = []
    spread_types = ["Bull Put", "Bear Call", "Bull Call", "Bear Put"]
    for stype in spread_types:
        sub = df[df["type"] == stype].sort_values("ev", ascending=False)
        if not sub.empty:
            row = sub.iloc[0]
            kind = "Credit" if not row["is_debit"] else "Debit"
            top_picks.append({
                "Type": f"{stype} ({kind})",
                "Short": f"{row['k_short']:,.0f}",
                "Long": f"{row['k_long']:,.0f}",
                "Width": f"{row['width']:,.0f}",
                "S DTE": row["dte"],
                "L DTE": row.get("dte_long", row["dte"]),
                "Premium": f"${row['net_cost']*100:,.0f}",
                "Full": f"${row['pnl_full']*100:+,.0f}",
                "Half": f"${row['pnl_half']*100:+,.0f}",
                "Flat": f"${row['pnl_flat']*100:+,.0f}",
                "Weighted": f"${row['pnl_w']*100:+,.0f}",
                "PoP": f"{row['pop']*100:.0f}%",
                "EV": f"${row['ev']*100:+,.0f}",
            })
    if top_picks:
        st.dataframe(pd.DataFrame(top_picks), use_container_width=True,
                      hide_index=True, column_config=VERT_TOP_CFG)

    # OptionStrat links
    if top_picks:
        all_legs = []
        credit_legs = []
        debit_legs = []
        per_type_data = []
        for stype in spread_types:
            sub = df[df["type"] == stype].sort_values("ev", ascending=False)
            if sub.empty:
                continue
            top = sub.iloc[0]
            legs = [
                {"strike": int(float(top["k_short"])),
                 "option_type": str(top["opt_type"]),
                 "expiration": str(top["exp"]),
                 "long": False, "qty": 1},
                {"strike": int(float(top["k_long"])),
                 "option_type": str(top["opt_type"]),
                 "expiration": str(top.get("exp_long", top["exp"])),
                 "long": True, "qty": 1},
            ]
            all_legs.extend(legs)
            per_type_data.append((stype, legs))
            if stype in ("Bull Put", "Bear Call"):
                credit_legs.extend(legs)
            else:
                debit_legs.extend(legs)

        clean_sym = res["symbol"].replace("^", "")

        # Row 1: All + Credits + Debits
        r1, r2, r3 = st.columns(3)
        with r1:
            all_sorted = sorted(all_legs, key=lambda l: (l["long"], l["option_type"] != "put"))
            url_all = bs.optionstrat_url(res["symbol"], all_sorted)
            if url_all:
                st.markdown(f"[All Picks]({url_all})")
            csv_all = bs.ibkr_basket_csv(res["symbol"], all_sorted, tag="Verticals")
            st.download_button("All IBKR CSV", csv_all,
                                f"vert_all_{clean_sym}.csv", "text/csv", key="csv_v_all")
        with r2:
            if credit_legs:
                cr_sorted = sorted(credit_legs, key=lambda l: (l["long"], l["option_type"] != "put"))
                url_cr = bs.optionstrat_url(res["symbol"], cr_sorted)
                if url_cr:
                    st.markdown(f"[Credits: Bull Put + Bear Call]({url_cr})")
                csv_cr = bs.ibkr_basket_csv(res["symbol"], cr_sorted, tag="Credits")
                st.download_button("Credits IBKR CSV", csv_cr,
                                    f"vert_credits_{clean_sym}.csv", "text/csv", key="csv_v_cr")
        with r3:
            if debit_legs:
                db_sorted = sorted(debit_legs, key=lambda l: (l["long"], l["option_type"] != "put"))
                url_db = bs.optionstrat_url(res["symbol"], db_sorted)
                if url_db:
                    st.markdown(f"[Debits: Bull Call + Bear Put]({url_db})")
                csv_db = bs.ibkr_basket_csv(res["symbol"], db_sorted, tag="Debits")
                st.download_button("Debits IBKR CSV", csv_db,
                                    f"vert_debits_{clean_sym}.csv", "text/csv", key="csv_v_db")

        # Row 2: Per-type links
        cols = st.columns(len(per_type_data))
        for i, (stype, legs) in enumerate(per_type_data):
            with cols[i]:
                url = bs.optionstrat_url(res["symbol"], legs)
                if url:
                    st.markdown(f"[{stype}]({url})")
                csv = bs.ibkr_basket_csv(res["symbol"], legs,
                                          tag=stype.replace(" ", ""))
                st.download_button(f"{stype} CSV", csv,
                                    f"vert_{stype.replace(' ','')}_{clean_sym}.csv",
                                    "text/csv", key=f"csv_v_{i}")

    # ── Column legend ──
    with st.expander("Column Legend"):
        st.markdown(
            "- **Short K**: Strike of the sold (short) leg. For credits: closer to ATM. For debits: further OTM.\n"
            "- **Long K**: Strike of the bought (long) leg. Defines the risk boundary.\n"
            "- **Width**: Distance between strikes in points. "
            "For credit spreads: width = max loss + premium received. "
            "For debit spreads: width = max profit + premium paid.\n"
            "- **Premium**: Net credit received (credit spreads) or debit paid (debit spreads), ×100 per contract.\n"
            "- **Full**: P&L if your full forecast move happens. "
            "Calculated: BS price at target spot, projected IV, reduced DTE.\n"
            "- **Half**: P&L if only 50% of the move happens.\n"
            "- **Flat**: P&L if spot stays unchanged. "
            "Credit spreads gain from theta; debit spreads lose.\n"
            "- **Weighted**: Conviction-weighted average: "
            "Full×w1 + Half×w2 + Flat×w3. Best single metric for comparison.\n"
            "- **W%**: Weighted P&L as % of capital at risk.\n"
            "- **Max Loss**: Maximum possible loss = width - premium (credits) or premium paid (debits).\n"
            "- **PoP**: Probability of Profit. "
            "Credits: P(short leg expires OTM). Debits: P(spot moves past break-even).\n"
            "- **EV**: Expected Value = Weighted × PoP. "
            "The risk-adjusted metric. Higher = better risk/reward.\n"
            "- **Delta**: Net delta. Negative = bearish exposure, positive = bullish.\n\n"
            "**Spread Types:**\n"
            "- **Bull Put** (credit): Sell higher put, buy lower put. "
            "Max profit = premium. Max loss = width - premium. Best for: flat to bullish.\n"
            "- **Bear Call** (credit): Sell lower call, buy higher call. "
            "Max profit = premium. Max loss = width - premium. Best for: flat to bearish.\n"
            "- **Bull Call** (debit): Buy lower call, sell higher call. "
            "Max profit = width - premium. Max loss = premium. Best for: bullish.\n"
            "- **Bear Put** (debit): Buy higher put, sell lower put. "
            "Max profit = width - premium. Max loss = premium. Best for: bearish.\n\n"
            "**Typical combos:**\n"
            "- Bearish view: Bear Put (debit, directional) + Bull Put (credit, income) = Iron Condor downside\n"
            "- Bullish view: Bull Call (debit, directional) + Bear Call (credit, income) = Iron Condor upside"
        )

    # ── Sort + per-type tables ──
    sort_by = st.selectbox("Sort all tables by",
                            ["ev", "pnl_w", "pop", "roc"],
                            format_func=lambda x: {"ev": "Expected Value",
                                "pnl_w": "Weighted P&L", "pop": "Prob. of Profit",
                                "roc": "Return on Capital"}[x],
                            help="EV (recommended): balances P&L with probability. "
                                 "Weighted P&L: conviction-weighted profit, ignores probability. "
                                 "PoP: highest win rate, ignores profit size. "
                                 "ROC: return on capital at risk (premium for debits, max loss for credits).",
                            key="v_sort")

    for stype in spread_types:
        sub = df[df["type"] == stype].sort_values(sort_by, ascending=False).head(15)
        if sub.empty:
            continue
        kind = "Credit" if stype in ["Bull Put", "Bear Call"] else "Debit"
        st.markdown(f"### {stype} ({kind})")
        st.dataframe(_fmt_table(sub), use_container_width=True, hide_index=True,
                      column_config=VERT_COL_CFG)


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, move_pct, forecast_dte, strike_step, scan_dte_targets,
            conviction, widths, lp_dte_offset_pct=20):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(forecast_dte)
    vix = get_vix()
    iv_now = vix / 100.0 if not np.isnan(vix) else 0.20

    target_spot = spot * (1 + move_pct / 100.0)
    iv_target = project_iv(iv_now, spot, target_spot)

    today = datetime.date.today()
    try:
        exps, chain_ticker = get_available_expirations(symbol)
    except Exception as e:
        raise ValueError(f"Could not fetch expirations: {e}")

    if not exps:
        raise ValueError(f"No expirations for {symbol}.")

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

    scan_dtes = set()
    expirations_data = {}
    errors = []

    # Load chains for target DTEs first
    for target_dte in scan_dte_targets:
        best_exp = min(exp_dte.keys(), key=lambda e: abs(exp_dte[e] - target_dte))
        if best_exp in expirations_data:
            continue
        try:
            ch, _, _ = resolve_options_chain(symbol, exp_dte[best_exp])
            expirations_data[ch["expiration"]] = ch
            scan_dtes.add(ch["dte_actual"])
        except Exception as e:
            errors.append(str(e))
            continue

    # Load additional chains for offset (next expiration out)
    if lp_dte_offset_pct > 0:
        loaded_dtes = {c["dte_actual"] for c in expirations_data.values()}
        for base_dte in list(loaded_dtes):
            min_offset_dte = round(base_dte * (1 + lp_dte_offset_pct / 100))
            offset_candidates = [(e, d) for e, d in exp_dte.items()
                                  if d >= min_offset_dte and d not in loaded_dtes
                                  and e not in expirations_data]
            if offset_candidates:
                best = min(offset_candidates, key=lambda x: x[1])
                try:
                    ch, _, _ = resolve_options_chain(symbol, best[1])
                    expirations_data[ch["expiration"]] = ch
                    loaded_dtes.add(ch["dte_actual"])
                except Exception:
                    pass

    if not expirations_data:
        raise ValueError(f"Could not load chains. Errors: {'; '.join(errors)}")

    results = scan_verticals(spot, r, q, iv_now, target_spot, iv_target,
                              forecast_dte, strike_step, expirations_data,
                              scan_dtes, conviction, widths, lp_dte_offset_pct)

    return {
        "symbol": symbol, "spot": spot, "vix": vix,
        "iv_now": iv_now, "iv_target": iv_target,
        "target_spot": target_spot, "move_pct": move_pct,
        "forecast_dte": forecast_dte, "conviction": conviction,
        "all_results": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Verticals")
    st.caption("Forecast-driven vertical spread scanner. "
               "Finds the best credit and debit spreads for your directional view.")

    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
            help="Underlying to scan.").upper()
    with c2:
        move_pct = st.number_input("Move %", value=-2.0,
            min_value=-20.0, max_value=20.0, step=0.5, format="%.1f",
            help="Expected % move. Negative = bearish, positive = bullish.")
    with c3:
        forecast_dte = st.number_input("Forecast DTE", value=5,
            min_value=1, max_value=30, step=1,
            help="Days until expected move. Scans 3 DTEs: this, ~14d, ~30d.")
    with c4:
        conviction = st.number_input("Conviction %", value=50,
            min_value=10, max_value=90, step=10,
            help="Scenario weighting. Low → flat favors credits. High → favors debits.")
    with c5:
        default_step = 25 if "SPX" in symbol else 5
        strike_step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50,
            help="Strike increment. Spreads at 1× and 2× this value.")
    with c6:
        diag_spread = st.toggle("Diagonal Spread", value=True,
            help="Long leg gets +20% more DTE than short leg. "
                 "Better vega protection in IV spikes.")

    scan_targets = sorted(set([forecast_dte,
                                max(forecast_dte, 14),
                                max(forecast_dte, 30)]))
    widths = [1, 2]  # 1 and 2 strike steps

    if st.button("Scan Verticals", type="primary", use_container_width=True):
        with st.spinner("Scanning vertical spreads..."):
            try:
                result = compute(symbol, move_pct, forecast_dte,
                                  strike_step, scan_targets,
                                  conviction / 100.0, widths,
                                  lp_dte_offset_pct=20 if diag_spread else 0)
                result["diag_spread"] = diag_spread
                st.session_state["vert_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "vert_result" not in st.session_state:
        st.info("Enter your forecast and click Scan Verticals.")
        return

    display(st.session_state["vert_result"])


main()
