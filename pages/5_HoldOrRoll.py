"""
Hold or Roll - Analyze whether to hold, close, or roll an option position.

Given your current position and market outlook, compares:
1. Hold to target DTE
2. Close now
3. Top 3 roll candidates (new strike + DTE)
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bs_engine as bs
from data_provider import (
    resolve_spot_price, resolve_options_chain, get_options_chain,
    get_available_expirations, get_dividend_yield, get_risk_free_rate,
    build_smile_curve, interpolate_smile_iv, get_vix,
)

CUSTOM_CSS = """
<style>
    .main .block-container { font-size: 14px; padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 18px; }
    [data-testid="stMetricLabel"] { font-size: 12px; }
    .main h1 { font-size: 22px; margin-bottom: 0.3rem; }
    .main h3 { font-size: 16px; margin-top: 0.6rem; margin-bottom: 0.3rem; }
</style>
"""


def get_strike_step(symbol, spot):
    """Determine strike step for roll candidates."""
    sym = symbol.upper().replace("^", "")
    if sym in ("SPX", "GSPC", "SPXW"):
        return 25 if spot > 2000 else 5
    elif sym in ("NDX", "NDXP"):
        return 50
    elif spot > 500:
        return 5
    elif spot > 100:
        return 1
    return 0.5


def price_option(S, K, T, r, sigma, q, opt_type):
    """Price an option and return all info. Returns None on error."""
    if T <= 0 or sigma <= 0:
        return None
    try:
        res = bs.calculate_all(S, K, T, r, sigma, q, opt_type)
        return res
    except Exception:
        return None


def estimate_iv_for_strike(smile_df, strike, fallback_iv):
    """Get IV from smile curve, with fallback."""
    if smile_df is not None and not smile_df.empty:
        iv = interpolate_smile_iv(smile_df, strike)
        if not np.isnan(iv) and 0.01 < iv < 2.0:
            return iv
    return fallback_iv


def iv_adjustment(spot_now, spot_target, base_iv):
    """Adjust IV based on spot move direction.

    Heuristic: ~4 vol points per 10% spot drop (SPX-like).
    Spot drop -> IV rises, spot rise -> IV drops.
    """
    pct_move = (spot_target - spot_now) / spot_now
    iv_shift = -pct_move * 0.40  # 10% drop -> +4% IV
    return max(base_iv + iv_shift, 0.05)


def _pnl_for_scenario(is_long, entry_price, exit_price):
    """Compute P&L for a position."""
    if is_long:
        return exit_price - entry_price
    else:
        return entry_price - exit_price


def generate_roll_candidates(symbol, spot, current_strike, current_dte,
                              opt_type, is_long, S, r, q, atm_iv,
                              current_price, smile_data, expected_spot,
                              expected_dte_years, step, outlook_dte,
                              close_pnl):
    """
    Generate and evaluate roll candidates with 3-scenario scoring.

    Scores each roll on:
    1. Secured profit: Roll-credit preserves existing gains
    2. Efficiency: P&L relative to roll cost across 3 scenarios
    3. Robustness: Performance when move is 100%, 50%, or 0% of expected
    4. Safety: DTE buffer after outlook period

    Returns list sorted by composite score (descending).
    """
    from scipy.stats import norm

    expirations, chain_ticker = get_available_expirations(symbol)
    today = pd.Timestamp.now().normalize().date()

    # 3 scenarios: full move, half move, no move
    move_delta = expected_spot - S
    scenarios = {
        "full": expected_spot,
        "half": S + move_delta * 0.5,
        "none": S,
    }
    scenario_weights = {"full": 0.50, "half": 0.35, "none": 0.15}

    # IV adjustment per scenario: spot drops -> IV rises, spot rises -> IV drops
    # Uses module-level iv_adjustment() function

    candidates = []

    for exp_str in expirations:
        import datetime
        exp_date = datetime.date.fromisoformat(exp_str)
        new_dte = (exp_date - today).days

        # DTE must be longer than outlook period, not too far out
        if new_dte <= outlook_dte or new_dte > max(current_dte * 3, 180):
            continue

        try:
            chain = get_options_chain(chain_ticker, new_dte)
        except Exception:
            continue

        chain_df = chain["calls"] if opt_type == "call" else chain["puts"]
        if chain_df.empty:
            continue

        smile_df = build_smile_curve(chain_df, S)
        actual_new_dte = chain["dte_actual"]
        T_new = actual_new_dte / 365.0

        # Remaining DTE on new option after outlook period
        remaining_after = actual_new_dte - outlook_dte
        if remaining_after < 1:
            continue
        T_after = remaining_after / 365.0

        # Strike range: focus on OTM region with some ITM
        if opt_type == "put":
            strike_min = S * 0.82
            strike_max = S * 1.05
        else:
            strike_min = S * 0.95
            strike_max = S * 1.18

        available_strikes = sorted(chain_df["strike"].unique())
        relevant_strikes = [k for k in available_strikes
                           if strike_min <= k <= strike_max]

        if len(relevant_strikes) > 15:
            step_n = max(1, len(relevant_strikes) // 12)
            relevant_strikes = relevant_strikes[::step_n]

        for new_strike in relevant_strikes:
            new_iv = estimate_iv_for_strike(smile_df, new_strike, atm_iv)
            new_res = price_option(S, new_strike, T_new, r, new_iv, q, opt_type)
            if new_res is None:
                continue

            new_price = new_res["price"]

            # Roll cost
            if is_long:
                roll_cost = new_price - current_price
            else:
                roll_cost = current_price - new_price

            # P&L for each scenario (with IV adjustment)
            scenario_gross = {}  # Total: roll credit/debit + new option P&L
            scenario_net = {}    # New option P&L only
            for label, target_S in scenarios.items():
                adj_iv = iv_adjustment(S, target_S, new_iv)
                res_at_target = price_option(target_S, new_strike, T_after,
                                              r, adj_iv, q, opt_type)
                if res_at_target is None:
                    scenario_gross[label] = -abs(roll_cost)
                    scenario_net[label] = 0
                    continue
                pos_pnl = _pnl_for_scenario(is_long, new_price,
                                             res_at_target["price"])
                scenario_net[label] = pos_pnl
                scenario_gross[label] = pos_pnl - roll_cost  # roll_cost negative = credit adds

            # Weighted average P&L across scenarios
            weighted_pnl = sum(scenario_gross[k] * scenario_weights[k]
                               for k in scenarios)

            # Net P&L at full move (for display)
            net_pnl_full = scenario_gross["full"]

            # DTE safety factor: bonus for more time after outlook
            dte_ratio = min(remaining_after / max(outlook_dte, 1), 3.0)
            dte_factor = 0.5 + 0.5 * min(dte_ratio / 2.0, 1.0)

            # Capital efficiency score
            # For Long: capital at risk = price of new option
            # For Short: capital at risk = margin (approximate as option price)
            capital_at_risk = max(new_price, 1.0)

            # Return on capital across scenarios (weighted)
            weighted_return = weighted_pnl / capital_at_risk

            # Roll credit bonus: credit rolls are preferred (secure profits)
            # roll_cost negative = credit; positive = debit
            credit_factor = 1.0
            if roll_cost < 0:
                # Credit roll: bonus proportional to credit size vs current price
                credit_ratio = abs(roll_cost) / max(current_price, 1)
                credit_factor = 1.0 + credit_ratio * 0.5  # up to 1.5x bonus
            elif roll_cost > 0:
                # Debit roll: penalty
                debit_ratio = roll_cost / max(current_price, 1)
                credit_factor = max(0.3, 1.0 - debit_ratio)

            # Composite score: efficiency × safety × credit preference
            score = weighted_return * dte_factor * credit_factor * 100

            candidates.append({
                "strike": new_strike,
                "dte": actual_new_dte,
                "expiration": chain["expiration"],
                "iv": new_iv,
                "price": new_price,
                "delta": new_res["delta"],
                "theta": new_res["theta_daily"],
                "roll_cost": roll_cost,
                "pnl_full": scenario_gross["full"],
                "pnl_half": scenario_gross["half"],
                "pnl_none": scenario_gross["none"],
                "net_full": scenario_net["full"],
                "net_half": scenario_net["half"],
                "net_none": scenario_net["none"],
                "net_pnl": net_pnl_full,
                "weighted_pnl": weighted_pnl,
                "remaining_after": remaining_after,
                "score": score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Hold or Roll")

    # ---- Inputs ----
    st.markdown("### Current Position")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        symbol = st.text_input("Symbol", value="^SPX",
                                help="Underlying symbol.").upper()
    with c2:
        cur_strike = st.number_input("Strike", value=6800.0, min_value=1.0,
                                      step=5.0, format="%.0f",
                                      help="Strike of your current option.")
    with c3:
        cur_dte = st.number_input("DTE", value=30, min_value=1, max_value=730,
                                   help="Days to expiration of your current option.")
    with c4:
        opt_type_sel = st.selectbox("Type", ["Put", "Call"],
                                     help="Option type of your current position.")
    with c5:
        pos_sel = st.selectbox("Position", ["Short", "Long"],
                                help="Short = sold/written. Long = bought.")

    st.markdown("### Market Outlook")
    o1, o2, o3, o4, o5 = st.columns([1, 1, 1, 1, 1])
    with o1:
        move_mode = st.radio("Move as", ["Percent", "Price"], horizontal=True,
                              help="How to specify your expected spot move.")
    with o2:
        if move_mode == "Percent":
            move_val = st.number_input("Expected Move %", value=0.0,
                                        min_value=-30.0, max_value=30.0,
                                        step=0.5, format="%.1f",
                                        help="Expected spot change. +2 = up 2%, -3 = down 3%.")
        else:
            move_val = st.number_input("Expected Price", value=6500.0,
                                        min_value=100.0, max_value=99000.0,
                                        step=10.0, format="%.0f",
                                        help="Expected spot price after the move.")
    with o3:
        outlook_dte = st.number_input("Over DTE", value=14, min_value=1,
                                       max_value=365,
                                       help="Timeframe for expected move in days.")
    with o4:
        entry_price = st.number_input("Entry Price", value=0.0,
                                       min_value=0.0, step=1.0, format="%.2f",
                                       help="Price paid/received when opening the position. "
                                            "0 = use current market price (no P&L for Close).")
    with o5:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary", use_container_width=True)

    opt_type = opt_type_sel.lower()
    is_long = pos_sel == "Long"

    # ---- Run ----
    if run:
        with st.spinner("Fetching data and computing rolls..."):
            try:
                result = _compute(symbol, cur_strike, cur_dte, opt_type,
                                   is_long, move_mode, move_val, outlook_dte,
                                   entry_price)
                st.session_state["hor_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    if "hor_result" not in st.session_state:
        st.info("Enter your position and outlook, then click Analyze.")
        return

    _display(st.session_state["hor_result"])


def _compute(symbol, cur_strike, cur_dte, opt_type, is_long,
             move_mode, move_val, outlook_dte, entry_price):
    """Fetch data and compute hold/close/roll scenarios."""

    # Market data
    spot, _ = resolve_spot_price(symbol)
    chain, chain_ticker, spy_fallback = resolve_options_chain(symbol, cur_dte)

    S = spot
    scale = 1.0
    if spy_fallback:
        from data_provider import get_spot_price
        spy_spot = get_spot_price("SPY")
        scale = spot / spy_spot
        S = spy_spot

    q = get_dividend_yield(symbol)
    r = get_risk_free_rate(cur_dte)
    vix = get_vix()

    # Current option IV from smile
    chain_df = chain["calls"] if opt_type == "call" else chain["puts"]
    smile_df = build_smile_curve(chain_df, S)
    working_strike = cur_strike / scale if spy_fallback else cur_strike

    cur_iv = estimate_iv_for_strike(smile_df, working_strike, vix / 100.0)
    actual_dte = chain["dte_actual"]
    T_cur = actual_dte / 365.0

    # Price current option
    cur_res = price_option(S, working_strike, T_cur, r, cur_iv, q, opt_type)
    if cur_res is None:
        raise ValueError("Could not price current option.")

    cur_price = cur_res["price"]
    if spy_fallback:
        cur_price *= scale

    # Expected spot
    if move_mode == "Percent":
        expected_spot = spot * (1 + move_val / 100.0)
    else:
        # Points mode = expected price directly
        expected_spot = move_val

    expected_S = expected_spot / scale if spy_fallback else expected_spot
    outlook_T = outlook_dte / 365.0

    # ---- Scenario 1: HOLD (3 scenarios) ----
    hold_remaining_dte = max(actual_dte - outlook_dte, 1)
    hold_T = hold_remaining_dte / 365.0

    move_delta = expected_S - (S if spy_fallback else spot)
    hold_scenarios = {
        "full": expected_S,
        "half": (S if spy_fallback else spot) + move_delta * 0.5,
        "none": S if spy_fallback else spot,
    }

    hold_pnls = {}
    base_S = S if spy_fallback else spot
    for label, target_S in hold_scenarios.items():
        adj_iv = iv_adjustment(base_S, target_S, cur_iv)
        hold_res = price_option(target_S, working_strike, hold_T, r, adj_iv, q, opt_type)
        if hold_res:
            hp = hold_res["price"] * scale if spy_fallback else hold_res["price"]
            if is_long:
                hold_pnls[label] = hp - cur_price
            else:
                hold_pnls[label] = cur_price - hp
        else:
            hold_pnls[label] = 0

    hold_pnl = hold_pnls["full"]

    # ---- Scenario 2: CLOSE NOW ----
    # P&L from closing at current market price
    if entry_price > 0:
        if is_long:
            close_pnl = cur_price - entry_price  # sell what you bought
        else:
            close_pnl = entry_price - cur_price  # buy back what you sold
    else:
        close_pnl = 0  # no entry price given

    # ---- Scenario 3: ROLL CANDIDATES ----
    step = get_strike_step(symbol, spot)
    candidates = generate_roll_candidates(
        symbol, spot if not spy_fallback else S,
        working_strike, actual_dte, opt_type, is_long,
        S if spy_fallback else spot, r, q,
        cur_iv, cur_price / scale if spy_fallback else cur_price,
        smile_df, expected_S, outlook_T, step, outlook_dte,
        close_pnl
    )

    # Scale back if SPY fallback
    if spy_fallback:
        for c in candidates:
            c["strike"] = c["strike"] * scale
            c["price"] = c["price"] * scale
            c["roll_cost"] = c["roll_cost"] * scale
            c["pnl_full"] = c["pnl_full"] * scale
            c["pnl_half"] = c["pnl_half"] * scale
            c["pnl_none"] = c["pnl_none"] * scale
            c["net_full"] = c["net_full"] * scale
            c["net_half"] = c["net_half"] * scale
            c["net_none"] = c["net_none"] * scale
            c["net_pnl"] = c["net_pnl"] * scale
            c["weighted_pnl"] = c["weighted_pnl"] * scale

    return {
        "symbol": symbol,
        "spot": spot,
        "cur_strike": cur_strike,
        "cur_dte": actual_dte,
        "cur_expiration": chain["expiration"],
        "opt_type": opt_type,
        "is_long": is_long,
        "cur_price": cur_price,
        "cur_iv": cur_iv,
        "cur_delta": cur_res["delta"],
        "cur_theta": cur_res["theta_daily"],
        "r": r,
        "q": q,
        "vix": vix,
        "expected_spot": expected_spot,
        "outlook_dte": outlook_dte,
        "hold_pnl": hold_pnl,
        "hold_pnl_half": hold_pnls["half"],
        "hold_pnl_none": hold_pnls["none"],
        "hold_remaining_dte": hold_remaining_dte,
        "close_pnl": close_pnl,
        "entry_price": entry_price,
        "candidates": candidates[:20],  # top 20
    }


def _display(res):
    """Display results."""
    spot = res["spot"]
    pos_label = "Long" if res["is_long"] else "Short"
    type_label = res["opt_type"].upper()
    move_pct = ((res["expected_spot"] / spot) - 1) * 100

    # ---- Header ----
    st.markdown("---")
    st.markdown(f"### {res['symbol']} @ {spot:,.2f}  |  "
                f"{pos_label} {type_label} {res['cur_strike']:,.0f}  |  "
                f"{res['cur_dte']} DTE ({res['cur_expiration']})")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Option Price", f"${res['cur_price']:,.2f}")
    m2.metric("IV", f"{res['cur_iv']*100:.1f}%")
    m3.metric("Delta", f"{res['cur_delta']:.3f}")
    m4.metric("Theta/day", f"${res['cur_theta']:.4f}")
    m5.metric("VIX", f"{res['vix']:.1f}")

    # ---- Outlook ----
    st.markdown(f"**Outlook:** Spot {res['expected_spot']:,.0f} "
                f"({move_pct:+.1f}%) in {res['outlook_dte']} days")

    # ---- Comparison: Hold vs Close vs Roll ----
    st.markdown("---")
    st.markdown("### Decision: Hold, Close, or Roll")

    top_rolls = res["candidates"][:3]
    cur_price = res["cur_price"]

    def pnl_fmt(val, ref_price=cur_price):
        """Format P&L as dollar + percentage."""
        if ref_price > 0:
            pct = val / ref_price * 100
            return f"${val:+,.0f} ({pct:+.0f}%)"
        return f"${val:+,.0f}"

    # Build comparison rows
    rows = []

    # Hold
    rows.append({
        "Action": "Hold",
        "Strike": f"{res['cur_strike']:,.0f}",
        "DTE": res["hold_remaining_dte"],
        "Roll Credit": "-",
        "Gross (Full)": pnl_fmt(res["hold_pnl"]),
        "Gross (Half)": pnl_fmt(res.get("hold_pnl_half", 0)),
        "Gross (None)": pnl_fmt(res.get("hold_pnl_none", 0)),
        "New Opt P&L": "-",
        "Score": "-",
        "Delta": f"{res['cur_delta']:.3f}",
        "IV": f"{res['cur_iv']*100:.1f}%",
    })

    # Close
    close_pnl = res["close_pnl"]
    if res["entry_price"] > 0:
        close_str = pnl_fmt(close_pnl, res["entry_price"])
    else:
        close_str = "n/a"
    rows.append({
        "Action": "Close",
        "Strike": "-",
        "DTE": 0,
        "Roll Credit": "-",
        "Gross (Full)": close_str,
        "Gross (Half)": close_str,
        "Gross (None)": close_str,
        "New Opt P&L": "-",
        "Score": "-",
        "Delta": "0",
        "IV": "-",
    })

    # Top 3 Rolls
    for i, c in enumerate(top_rolls):
        cost_str = f"${c['roll_cost']:+,.0f}"
        if c["roll_cost"] > 0:
            cost_label = f"{cost_str} debit"
        elif c["roll_cost"] < 0:
            cost_label = f"{cost_str} credit"
        else:
            cost_label = "$0"

        ref = c["price"] if c["price"] > 0 else 1

        rows.append({
            "Action": f"Roll #{i+1}",
            "Strike": f"{c['strike']:,.0f}",
            "DTE": c["dte"],
            "Roll Credit": cost_label,
            "Gross (Full)": pnl_fmt(c["pnl_full"], ref),
            "Gross (Half)": pnl_fmt(c["pnl_half"], ref),
            "Gross (None)": pnl_fmt(c["pnl_none"], ref),
            "New Opt P&L": pnl_fmt(c["net_full"], ref),
            "Score": f"{c['score']:.0f}",
            "Delta": f"{c['delta']:.3f}",
            "IV": f"{c['iv']*100:.1f}%",
        })

    df = pd.DataFrame(rows)

    def highlight_best(row):
        """Highlight the best Net P&L row."""
        styles = [""] * len(row)
        if row["Action"] == "Hold":
            styles = ["background-color: #f0f0f0"] * len(row)
        elif row["Action"] == "Close":
            styles = ["background-color: #f0f0f0"] * len(row)
        elif row["Action"] == "Roll #1":
            styles = ["background-color: #d4edda"] * len(row)
        return styles

    styled = df.style.apply(highlight_best, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # OptionStrat links
    st.markdown("#### Links")
    link_cols = st.columns(min(len(top_rolls) + 1, 4))

    # Current position link
    with link_cols[0]:
        cur_url = bs.optionstrat_url(res["symbol"], [{
            "strike": res["cur_strike"],
            "option_type": res["opt_type"],
            "expiration": res["cur_expiration"],
            "long": res["is_long"],
        }])
        if cur_url:
            st.caption(f"[Current position]({cur_url})")

    # Roll links + IBKR CSV
    for i, c in enumerate(top_rolls):
        if i + 1 < len(link_cols):
            with link_cols[i + 1]:
                roll_legs = [
                    # Close current
                    {"strike": res["cur_strike"],
                     "option_type": res["opt_type"],
                     "expiration": res["cur_expiration"],
                     "long": not res["is_long"]},
                    # Open new
                    {"strike": c["strike"],
                     "option_type": res["opt_type"],
                     "expiration": c["expiration"],
                     "long": res["is_long"]},
                ]
                roll_url = bs.optionstrat_url(res["symbol"], roll_legs)
                if roll_url:
                    st.caption(f"[Roll #{i+1}]({roll_url})")
                csv_data = bs.ibkr_basket_csv(res["symbol"], roll_legs,
                                               tag=f"Roll{i+1}")
                st.download_button(
                    f"#{i+1} IBKR CSV",
                    csv_data,
                    f"roll_{i+1}_{res['symbol']}.csv",
                    "text/csv",
                    key=f"hor_ibkr_{i}",
                )

    # ---- Recommendation ----
    st.markdown("---")
    best_hold_weighted = (res["hold_pnl"] * 0.5 +
                          res["hold_pnl_half"] * 0.35 +
                          res["hold_pnl_none"] * 0.15)
    best_roll_score = top_rolls[0]["score"] if top_rolls else -999999
    close_val = res["close_pnl"]

    # Compare weighted hold P&L vs best roll weighted P&L vs close
    best_roll_weighted = top_rolls[0]["weighted_pnl"] if top_rolls else -999999

    options = {"Hold": best_hold_weighted}
    if top_rolls:
        options["Roll"] = best_roll_weighted
    if res["entry_price"] > 0:
        options["Close"] = close_val

    best_action = max(options, key=options.get)

    if best_action == "Hold":
        st.success(
            f"Hold is the best risk-adjusted option. "
            f"Expected P&L: ${res['hold_pnl']:+,.0f} (full) / "
            f"${res['hold_pnl_half']:+,.0f} (half) / "
            f"${res['hold_pnl_none']:+,.0f} (no move)"
        )
    elif best_action == "Close":
        st.success(f"Close now: P&L ${close_val:+,.0f}")
    elif best_action == "Roll" and top_rolls:
        r1 = top_rolls[0]
        st.success(
            f"Roll to {r1['strike']:,.0f} ({r1['dte']} DTE). "
            f"P&L: ${r1['pnl_full']:+,.0f} (full) / "
            f"${r1['pnl_half']:+,.0f} (half) / "
            f"${r1['pnl_none']:+,.0f} (no move). "
            f"Roll cost: ${r1['roll_cost']:+,.0f}"
        )

    # ---- All candidates table ----
    if len(res["candidates"]) > 3:
        with st.expander(f"All Roll Candidates ({len(res['candidates'])})"):
            all_rows = []
            for c in res["candidates"]:
                all_rows.append({
                    "Strike": c["strike"],
                    "DTE": c["dte"],
                    "Exp": c["expiration"],
                    "Price": f"${c['price']:.2f}",
                    "IV": f"{c['iv']*100:.1f}%",
                    "Delta": f"{c['delta']:.3f}",
                    "Roll Cost": f"${c['roll_cost']:+,.0f}",
                    "Full": f"${c['pnl_full']:+,.0f}",
                    "Half": f"${c['pnl_half']:+,.0f}",
                    "None": f"${c['pnl_none']:+,.0f}",
                    "Score": f"{c['score']:.0f}",
                })
            st.dataframe(pd.DataFrame(all_rows), use_container_width=True,
                         hide_index=True)

    # ---- Scenario Comparison Chart ----
    if top_rolls:
        st.markdown("---")
        st.markdown("### Scenario Comparison")

        import plotly.graph_objects as go

        # Build chart data: Hold + Close + Top 3 Rolls
        labels = ["Hold"]
        full_vals = [res["hold_pnl"]]
        half_vals = [res.get("hold_pnl_half", 0)]
        none_vals = [res.get("hold_pnl_none", 0)]

        if res["entry_price"] > 0:
            labels.append("Close")
            full_vals.append(res["close_pnl"])
            half_vals.append(res["close_pnl"])
            none_vals.append(res["close_pnl"])

        for i, c in enumerate(top_rolls):
            labels.append(f"Roll {c['strike']:,.0f}\n{c['dte']}d")
            full_vals.append(c["pnl_full"])
            half_vals.append(c["pnl_half"])
            none_vals.append(c["pnl_none"])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Full Move", x=labels, y=full_vals,
            marker_color="rgba(50,180,80,0.7)",
            text=[f"${v:+,.0f}" for v in full_vals],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Half Move", x=labels, y=half_vals,
            marker_color="rgba(100,150,255,0.7)",
            text=[f"${v:+,.0f}" for v in half_vals],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="No Move", x=labels, y=none_vals,
            marker_color="rgba(200,200,200,0.7)",
            text=[f"${v:+,.0f}" for v in none_vals],
            textposition="outside",
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)

        fig.update_layout(
            template="plotly_white",
            height=400,
            barmode="group",
            margin=dict(l=50, r=20, t=30, b=40),
            yaxis_title="P&L ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        st.plotly_chart(fig, use_container_width=True)


main()
