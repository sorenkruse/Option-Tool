"""
Spreads – Vertical & Diagonal Spread Scanner

Scans all 4 spread types across DTEs using user-defined deltas:
- Bull Put (credit): SP + LP below spot
- Bear Call (credit): SC + LC above spot
- Bull Call (debit): LC + SC above spot
- Bear Put (debit): LP + SP below spot

Diagonal Spread toggle gives credit spread long legs +20% more DTE.
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

def scan_spreads(spot, r, q, iv_atm, strike_step,
                 sp_delta, lp_delta, sc_delta, lc_delta,
                 expirations_data, scan_dtes, lp_dte_offset_pct=20):
    """Scan all 4 spread types across DTEs."""

    # Map DTEs to chains for diagonal offset
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
        T = dte / 365.0

        # Diagonal offset: find chain with >= +20% DTE for credit long legs
        long_dte_min = round(dte * (1 + lp_dte_offset_pct / 100))
        candidates = [d for d in dte_to_exp.keys() if d >= long_dte_min and d > dte]
        if candidates and lp_dte_offset_pct > 0:
            long_dte_actual = min(candidates)
            long_exp_str, long_chain = dte_to_exp[long_dte_actual]
            long_call_smile = build_smile_curve(long_chain["calls"], spot)
            long_put_smile = build_smile_curve(long_chain["puts"], spot)
            T_long = long_dte_actual / 365.0
        else:
            long_dte_actual = dte
            long_exp_str = exp_str
            long_call_smile = call_smile
            long_put_smile = put_smile
            T_long = T

        # Resolve strikes from deltas
        try:
            k_sp = _round(bs.solve_strike_for_delta(-sp_delta, spot, T, r, iv_atm, q, "put"), strike_step)
            k_lp = _round(bs.solve_strike_for_delta(-lp_delta, spot, T, r, iv_atm, q, "put"), strike_step)
            k_sc = _round(bs.solve_strike_for_delta(sc_delta, spot, T, r, iv_atm, q, "call"), strike_step)
            k_lc = _round(bs.solve_strike_for_delta(lc_delta, spot, T, r, iv_atm, q, "call"), strike_step)
        except Exception:
            continue

        if k_sp <= k_lp:
            k_lp = k_sp - strike_step
        if k_lc <= k_sc:
            k_lc = k_sc + strike_step

        spreads = []

        # ── Bull Put (Credit): SP + LP below spot ──
        if k_sp < spot:
            iv_sp = _iv(put_smile, k_sp, iv_atm)
            iv_lp_c = _iv(long_put_smile, k_lp, iv_atm)
            try:
                sp_e = bs.calculate_all(spot, k_sp, T, r, iv_sp, q, "put")
                lp_e = bs.calculate_all(spot, k_lp, T_long, r, iv_lp_c, q, "put")
                credit = sp_e["price"] - lp_e["price"]
                if credit > 0:
                    width = k_sp - k_lp
                    spreads.append({
                        "type": "Bull Put", "opt_type": "put", "is_debit": False,
                        "k_short": k_sp, "k_long": k_lp, "width": width,
                        "dte": dte, "dte_long": long_dte_actual,
                        "exp": exp_str, "exp_long": long_exp_str,
                        "net_cost": credit, "max_loss": width - credit,
                        "short_entry": sp_e, "long_entry": lp_e,
                        "iv_short": iv_sp, "iv_long": iv_lp_c,
                    })
            except Exception:
                pass

        # ── Bear Call (Credit): SC + LC above spot ──
        if k_sc > spot:
            iv_sc = _iv(call_smile, k_sc, iv_atm)
            iv_lc_c = _iv(long_call_smile, k_lc, iv_atm)
            try:
                sc_e = bs.calculate_all(spot, k_sc, T, r, iv_sc, q, "call")
                lc_e = bs.calculate_all(spot, k_lc, T_long, r, iv_lc_c, q, "call")
                credit = sc_e["price"] - lc_e["price"]
                if credit > 0:
                    width = k_lc - k_sc
                    spreads.append({
                        "type": "Bear Call", "opt_type": "call", "is_debit": False,
                        "k_short": k_sc, "k_long": k_lc, "width": width,
                        "dte": dte, "dte_long": long_dte_actual,
                        "exp": exp_str, "exp_long": long_exp_str,
                        "net_cost": credit, "max_loss": width - credit,
                        "short_entry": sc_e, "long_entry": lc_e,
                        "iv_short": iv_sc, "iv_long": iv_lc_c,
                    })
            except Exception:
                pass

        # ── Bull Call (Debit): LC near ATM + SC further OTM ──
        # Use SC delta for short, LC delta would be too far OTM
        # For debit: long is closer to money, short is further out
        k_bc_long = _round(bs.solve_strike_for_delta(
            sc_delta, spot, T, r, iv_atm, q, "call"), strike_step)
        k_bc_short = k_bc_long + strike_step
        if k_bc_long > spot * 0.99:
            iv_bcl = _iv(call_smile, k_bc_long, iv_atm)
            iv_bcs = _iv(call_smile, k_bc_short, iv_atm)
            try:
                # Debit spreads: BOTH legs same DTE (no diagonal offset)
                bcl = bs.calculate_all(spot, k_bc_long, T, r, iv_bcl, q, "call")
                bcs = bs.calculate_all(spot, k_bc_short, T, r, iv_bcs, q, "call")
                debit = bcl["price"] - bcs["price"]
                if debit > 0:
                    width = k_bc_short - k_bc_long
                    spreads.append({
                        "type": "Bull Call", "opt_type": "call", "is_debit": True,
                        "k_short": k_bc_short, "k_long": k_bc_long, "width": width,
                        "dte": dte, "dte_long": dte,
                        "exp": exp_str, "exp_long": exp_str,
                        "net_cost": debit, "max_loss": debit,
                        "short_entry": bcs, "long_entry": bcl,
                        "iv_short": iv_bcs, "iv_long": iv_bcl,
                    })
            except Exception:
                pass

        # ── Bear Put (Debit): LP near ATM + SP further OTM ──
        k_bp_long = _round(bs.solve_strike_for_delta(
            -sp_delta, spot, T, r, iv_atm, q, "put"), strike_step)
        k_bp_short = k_bp_long - strike_step
        if k_bp_long < spot * 1.01:
            iv_bpl = _iv(put_smile, k_bp_long, iv_atm)
            iv_bps = _iv(put_smile, k_bp_short, iv_atm)
            try:
                bpl = bs.calculate_all(spot, k_bp_long, T, r, iv_bpl, q, "put")
                bps = bs.calculate_all(spot, k_bp_short, T, r, iv_bps, q, "put")
                debit = bpl["price"] - bps["price"]
                if debit > 0:
                    width = k_bp_long - k_bp_short
                    spreads.append({
                        "type": "Bear Put", "opt_type": "put", "is_debit": True,
                        "k_short": k_bp_short, "k_long": k_bp_long, "width": width,
                        "dte": dte, "dte_long": dte,
                        "exp": exp_str, "exp_long": exp_str,
                        "net_cost": debit, "max_loss": debit,
                        "short_entry": bps, "long_entry": bpl,
                        "iv_short": iv_bps, "iv_long": iv_bpl,
                    })
            except Exception:
                pass

        # Add Greeks and scoring
        for sp in spreads:
            se = sp["short_entry"]
            le = sp["long_entry"]
            sp["delta"] = -se["delta"] + le["delta"]
            sp["theta"] = -se["theta_daily"] + le["theta_daily"]
            sp["vega"] = -se["vega_pct"] + le["vega_pct"]
            sp["pop"] = _calc_pop(spot, sp, T, r, iv_atm, q)
            capital = sp["net_cost"] if sp["is_debit"] else sp["max_loss"]
            sp["roc"] = sp["net_cost"] / capital if capital > 0 else 0

        results.extend(spreads)

    return results


def _calc_pop(spot, sp, T, r, iv, q):
    """Simple PoP estimate for a spread."""
    try:
        if sp["is_debit"]:
            return 0.5  # placeholder for debit
        # Credit spread: probability spot stays between breakeven
        if sp["opt_type"] == "put":
            be = sp["k_short"] - sp["net_cost"]
            d2 = (np.log(spot / be) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            from scipy.stats import norm
            return norm.cdf(d2)
        else:
            be = sp["k_short"] + sp["net_cost"]
            d2 = (np.log(spot / be) + (r - q - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            from scipy.stats import norm
            return norm.cdf(d2)
    except Exception:
        return 0.5


# ── Compute ──────────────────────────────────────────────────────────────

def compute(symbol, strike_step, sp_delta, lp_delta, sc_delta, lc_delta,
            scan_dte_targets, lp_dte_offset_pct=20):
    spot, _ = resolve_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(14)
    vix = get_vix()
    iv_now = vix / 100.0 if not np.isnan(vix) else 0.20

    today = datetime.date.today()
    try:
        exps, _ = get_available_expirations(symbol)
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

    # Load chains for target DTEs
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
            errors.append(str(e))

    # Load offset chains
    if lp_dte_offset_pct > 0:
        loaded = {c["dte_actual"] for c in expirations_data.values()}
        for base in list(loaded):
            min_off = round(base * (1 + lp_dte_offset_pct / 100))
            cands = [(e, d) for e, d in exp_dte.items()
                     if d >= min_off and d not in loaded and e not in expirations_data]
            if cands:
                best = min(cands, key=lambda x: x[1])
                try:
                    ch, _, _ = resolve_options_chain(symbol, best[1])
                    expirations_data[ch["expiration"]] = ch
                    loaded.add(ch["dte_actual"])
                except Exception:
                    pass

    if not expirations_data:
        raise ValueError(f"Could not load chains. {'; '.join(errors)}")

    results = scan_spreads(spot, r, q, iv_now, strike_step,
                            sp_delta, lp_delta, sc_delta, lc_delta,
                            expirations_data, scan_dtes, lp_dte_offset_pct)
    if not results:
        raise ValueError("No valid spreads found.")

    return {
        "symbol": symbol, "spot": spot, "vix": vix,
        "iv_now": iv_now, "r": r, "q": q,
        "sp_delta": sp_delta, "lp_delta": lp_delta,
        "sc_delta": sc_delta, "lc_delta": lc_delta,
        "all_results": results,
    }


# ── Display ──────────────────────────────────────────────────────────────

TABLE_CFG = {
    "Short K": st.column_config.TextColumn("Short K"),
    "Long K": st.column_config.TextColumn("Long K"),
    "Width": st.column_config.TextColumn("Width"),
    "S DTE": st.column_config.NumberColumn("S DTE"),
    "L DTE": st.column_config.NumberColumn("L DTE"),
    "Premium": st.column_config.TextColumn("Premium", help="Net credit or debit (x100)"),
    "Max Loss": st.column_config.TextColumn("Max Loss", help="Maximum loss (x100)"),
    "PoP": st.column_config.TextColumn("PoP"),
    "ROC": st.column_config.TextColumn("ROC", help="Return on capital"),
    "Delta": st.column_config.TextColumn("Delta"),
    "Theta": st.column_config.TextColumn("Theta"),
}


def display(res):
    spot = res["spot"]
    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  VIX {res['vix']:.1f}")
    st.caption(f"Deltas: SP Δ{res['sp_delta']:.2f} / LP Δ{res['lp_delta']:.2f} | "
               f"SC Δ{res['sc_delta']:.2f} / LC Δ{res['lc_delta']:.2f}")

    df = pd.DataFrame(res["all_results"])
    if df.empty:
        st.warning("No results.")
        return

    # Group by type
    spread_types = ["Bull Put", "Bear Call", "Bull Call", "Bear Put"]
    credit_types = ["Bull Put", "Bear Call"]
    debit_types = ["Bull Call", "Bear Put"]

    # Best picks
    st.markdown("### Best per Type")
    top_picks = []
    for stype in spread_types:
        sub = df[df["type"] == stype]
        if sub.empty:
            continue
        # Credits: sort by ROC, Debits: sort by ROC
        best = sub.sort_values("roc", ascending=False).iloc[0]
        kind = "Credit" if not best["is_debit"] else "Debit"
        top_picks.append({
            "Type": f"{stype} ({kind})",
            "Short": f"{best['k_short']:,.0f}",
            "Long": f"{best['k_long']:,.0f}",
            "Width": f"{best['width']:,.0f}",
            "S DTE": best["dte"],
            "L DTE": best.get("dte_long", best["dte"]),
            "Premium": f"${best['net_cost']*100:,.0f}",
            "Max Loss": f"${best['max_loss']*100:,.0f}",
            "PoP": f"{best['pop']*100:.0f}%",
            "ROC": f"{best['roc']*100:.0f}%",
            "Delta": f"{best['delta']:.3f}",
        })
    if top_picks:
        st.dataframe(pd.DataFrame(top_picks), use_container_width=True,
                      hide_index=True)

    # OptionStrat links
    all_legs = []
    credit_legs = []
    debit_legs = []
    per_type_data = []
    for stype in spread_types:
        sub = df[df["type"] == stype].sort_values("roc", ascending=False)
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
        if stype in credit_types:
            credit_legs.extend(legs)
        else:
            debit_legs.extend(legs)

    clean_sym = res["symbol"].replace("^", "")

    r1, r2, r3 = st.columns(3)
    with r1:
        if all_legs:
            url = bs.optionstrat_url(res["symbol"],
                sorted(all_legs, key=lambda l: (l["long"], l["option_type"] != "put")))
            if url:
                st.markdown(f"[All Picks]({url})")
            csv = bs.ibkr_basket_csv(res["symbol"], all_legs, tag="Spreads")
            st.download_button("All IBKR CSV", csv,
                                f"spreads_all_{clean_sym}.csv", "text/csv", key="csv_s_all")
    with r2:
        if credit_legs:
            cr_types = [s for s in credit_types if any(st2 == s for st2, _ in per_type_data)]
            url = bs.optionstrat_url(res["symbol"],
                sorted(credit_legs, key=lambda l: (l["long"], l["option_type"] != "put")))
            if url:
                st.markdown(f"[Credits: {' + '.join(cr_types)}]({url})")
            csv = bs.ibkr_basket_csv(res["symbol"], credit_legs, tag="Credits")
            st.download_button("Credits IBKR CSV", csv,
                                f"spreads_cr_{clean_sym}.csv", "text/csv", key="csv_s_cr")
    with r3:
        if debit_legs:
            db_types = [s for s in debit_types if any(st2 == s for st2, _ in per_type_data)]
            url = bs.optionstrat_url(res["symbol"],
                sorted(debit_legs, key=lambda l: (l["long"], l["option_type"] != "put")))
            if url:
                st.markdown(f"[Debits: {' + '.join(db_types)}]({url})")
            csv = bs.ibkr_basket_csv(res["symbol"], debit_legs, tag="Debits")
            st.download_button("Debits IBKR CSV", csv,
                                f"spreads_db_{clean_sym}.csv", "text/csv", key="csv_s_db")

    # Per-type links
    cols = st.columns(len(per_type_data)) if per_type_data else []
    for i, (stype, legs) in enumerate(per_type_data):
        with cols[i]:
            url = bs.optionstrat_url(res["symbol"], legs)
            if url:
                st.markdown(f"[{stype}]({url})")
            csv = bs.ibkr_basket_csv(res["symbol"], legs, tag=stype.replace(" ", ""))
            st.download_button(f"{stype} CSV", csv,
                                f"spread_{stype.replace(' ','_')}_{clean_sym}.csv",
                                "text/csv", key=f"csv_s_{stype}")

    # Detail tables
    sort_by = st.selectbox("Sort tables by", ["ROC", "Premium", "PoP"],
                            key="spread_sort")
    sort_col = {"ROC": "roc", "Premium": "net_cost", "PoP": "pop"}[sort_by]

    for stype in spread_types:
        sub = df[df["type"] == stype].sort_values(sort_col, ascending=False)
        if sub.empty:
            continue
        kind = "Credit" if stype in credit_types else "Debit"
        st.markdown(f"### {stype} ({kind})")
        rows = []
        for _, row in sub.head(10).iterrows():
            rows.append({
                "Short K": f"{row['k_short']:,.0f}",
                "Long K": f"{row['k_long']:,.0f}",
                "Width": f"{row['width']:,.0f}",
                "S DTE": row["dte"],
                "L DTE": row.get("dte_long", row["dte"]),
                "Premium": f"${row['net_cost']*100:,.0f}",
                "Max Loss": f"${row['max_loss']*100:,.0f}",
                "PoP": f"{row['pop']*100:.0f}%",
                "ROC": f"{row['roc']*100:.0f}%",
                "Delta": f"{row['delta']:.3f}",
                "Theta": f"${row['theta']*100:+,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                      hide_index=True, column_config=TABLE_CFG)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Spreads")
    st.caption("Vertical and diagonal spread scanner. "
               "4 delta inputs define the strikes; the scanner finds the best DTE.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="SPX",
            help="Underlying to scan.").upper()
    with c2:
        default_step = 25 if "SPX" in symbol else 5
        strike_step = st.number_input("Strike Step", value=default_step,
            min_value=1, max_value=50,
            help="Strike increment for rounding.")
    with c3:
        diag_spread = st.toggle("Diagonal Spread", value=True,
            help="Credit spread long legs get +20% more DTE. "
                 "Better vega protection. Does NOT apply to debit spreads.")

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        sp_delta = st.number_input("SP Δ (Short Put)", value=0.30,
            min_value=0.05, max_value=0.50, step=0.05, format="%.2f",
            help="Delta for Bull Put short leg. Higher = closer to ATM.")
    with c5:
        lp_delta = st.number_input("LP Δ (Long Put)", value=0.20,
            min_value=0.05, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for Bull Put long leg. Lower = further OTM.")
    with c6:
        sc_delta = st.number_input("SC Δ (Short Call)", value=0.30,
            min_value=0.05, max_value=0.50, step=0.05, format="%.2f",
            help="Delta for Bear Call short leg. Higher = closer to ATM.")
    with c7:
        lc_delta = st.number_input("LC Δ (Long Call)", value=0.20,
            min_value=0.05, max_value=0.45, step=0.05, format="%.2f",
            help="Delta for Bear Call long leg. Lower = further OTM.")

    scan_targets = sorted(set([5, 7, 14, 21, 30]))

    if st.button("Scan Spreads", type="primary", use_container_width=True):
        with st.spinner("Scanning spreads..."):
            try:
                result = compute(symbol, strike_step,
                                  sp_delta, lp_delta, sc_delta, lc_delta,
                                  scan_targets,
                                  lp_dte_offset_pct=20 if diag_spread else 0)
                st.session_state["spread_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "spread_result" not in st.session_state:
        st.info("Set deltas and click Scan.")
        return

    display(st.session_state["spread_result"])


main()
