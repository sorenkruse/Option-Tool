"""
Black-Scholes Option Pricing Engine with comprehensive Greeks.

Supports European options (correct for SPX, approximation for American-style).
All formulas follow standard Black-Scholes-Merton with continuous dividend yield.

Parameters:
    S     - Spot price of underlying
    K     - Strike price
    T     - Time to expiration in years
    r     - Risk-free interest rate (annualized, decimal)
    sigma - Implied volatility (annualized, decimal)
    q     - Continuous dividend yield (annualized, decimal)
"""

import numpy as np
from scipy.stats import norm


def _validate_inputs(S, K, T, r, sigma, q):
    """Validate inputs and raise clear errors."""
    if S <= 0:
        raise ValueError(f"Spot price must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"Strike price must be positive, got {K}")
    if T <= 0:
        raise ValueError(f"Time to expiration must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"Volatility must be positive, got {sigma}")


def _d1(S, K, T, r, sigma, q):
    """Calculate d1 in Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma, q):
    """Calculate d2 in Black-Scholes formula."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# ---------------------------------------------------------------------------
# Option Prices
# ---------------------------------------------------------------------------

def call_price(S, K, T, r, sigma, q=0.0):
    """European call option price."""
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    """European put option price."""
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def option_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Calculate option price for given type."""
    if option_type.lower() == "call":
        return call_price(S, K, T, r, sigma, q)
    elif option_type.lower() == "put":
        return put_price(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ---------------------------------------------------------------------------
# First-Order Greeks
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Delta: Rate of change of option price with respect to underlying price.
    Call delta: e^(-qT) * N(d1), range [0, 1]
    Put delta:  e^(-qT) * (N(d1) - 1), range [-1, 0]
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    if option_type.lower() == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    Gamma: Rate of change of delta with respect to underlying price.
    Same for calls and puts.
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Theta: Rate of change of option price with respect to time (per year).
    Divide by 365 for daily theta.
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    # Common term: time decay of the option
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type.lower() == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)

    return term1 + term2 + term3


def vega(S, K, T, r, sigma, q=0.0):
    """
    Vega: Rate of change of option price with respect to volatility.
    Returns vega per 1 unit (100%) change in vol.
    Same for calls and puts.
    Divide by 100 for vega per 1% vol change.
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Rho: Rate of change of option price with respect to risk-free rate.
    Returns rho per 1 unit (100%) change in rate.
    Divide by 100 for rho per 1% rate change.
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)
    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# Second-Order Greeks
# ---------------------------------------------------------------------------

def vanna(S, K, T, r, sigma, q=0.0):
    """
    Vanna: Sensitivity of delta to changes in volatility,
    or equivalently sensitivity of vega to changes in underlying price.
    d(Delta)/d(sigma) = d(Vega)/d(S)
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, sigma, q=0.0):
    """
    Volga (Vomma): Sensitivity of vega to changes in volatility.
    d(Vega)/d(sigma) = Vega * d1 * d2 / sigma
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    v = vega(S, K, T, r, sigma, q)
    return v * d1 * d2 / sigma


def charm(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Charm (Delta Decay): Rate of change of delta with respect to time.
    d(Delta)/d(T)
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    charm_common = np.exp(-q * T) * norm.pdf(d1) * (
        2.0 * (r - q) * T - d2 * sigma * np.sqrt(T)
    ) / (2.0 * T * sigma * np.sqrt(T))

    if option_type.lower() == "call":
        return q * np.exp(-q * T) * norm.cdf(d1) - charm_common
    else:
        return -q * np.exp(-q * T) * norm.cdf(-d1) - charm_common


def speed(S, K, T, r, sigma, q=0.0):
    """
    Speed: Rate of change of gamma with respect to underlying price.
    d(Gamma)/d(S)
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    g = gamma(S, K, T, r, sigma, q)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1.0)


def color(S, K, T, r, sigma, q=0.0):
    """
    Color (Gamma Decay): Rate of change of gamma with respect to time.
    d(Gamma)/d(T)
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    return -np.exp(-q * T) * norm.pdf(d1) / (2.0 * S * T * sigma * np.sqrt(T)) * (
        2.0 * q * T + 1.0 + d1 * (2.0 * (r - q) * T - d2 * sigma * np.sqrt(T))
        / (sigma * np.sqrt(T))
    )


def zomma(S, K, T, r, sigma, q=0.0):
    """
    Zomma: Rate of change of gamma with respect to volatility.
    d(Gamma)/d(sigma)
    """
    _validate_inputs(S, K, T, r, sigma, q)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    g = gamma(S, K, T, r, sigma, q)
    return g * (d1 * d2 - 1.0) / sigma


# ---------------------------------------------------------------------------
# Implied Volatility (Newton-Raphson)
# ---------------------------------------------------------------------------

def implied_volatility(market_price, S, K, T, r, q=0.0, option_type="call",
                       max_iter=100, tol=1e-8):
    """
    Calculate implied volatility from market price using Newton-Raphson.

    Returns:
        float: Implied volatility, or NaN if not converged.
    """
    if market_price <= 0:
        return np.nan

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2.0 * np.pi / T) * market_price / S

    # Clamp initial guess
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iter):
        price = option_price(S, K, T, r, sigma, q, option_type)
        v = vega(S, K, T, r, sigma, q)

        if v < 1e-12:
            # Vega too small, try bisection fallback
            return _iv_bisection(market_price, S, K, T, r, q, option_type)

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / v
        sigma = max(0.001, min(sigma, 10.0))  # Keep in reasonable bounds

    # Fallback to bisection if Newton didn't converge
    return _iv_bisection(market_price, S, K, T, r, q, option_type)


def _iv_bisection(market_price, S, K, T, r, q, option_type,
                  low=0.001, high=10.0, max_iter=200, tol=1e-8):
    """Bisection fallback for implied volatility."""
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        price = option_price(S, K, T, r, mid, q, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Comprehensive Greeks Calculator
# ---------------------------------------------------------------------------

def calculate_all(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Calculate option price and all Greeks.

    Returns:
        dict with all computed values.
    """
    _validate_inputs(S, K, T, r, sigma, q)

    result = {
        "price": option_price(S, K, T, r, sigma, q, option_type),
        "delta": delta(S, K, T, r, sigma, q, option_type),
        "gamma": gamma(S, K, T, r, sigma, q),
        "theta_annual": theta(S, K, T, r, sigma, q, option_type),
        "theta_daily": theta(S, K, T, r, sigma, q, option_type) / 365.0,
        "vega": vega(S, K, T, r, sigma, q),
        "vega_pct": vega(S, K, T, r, sigma, q) / 100.0,  # per 1% vol change
        "rho": rho(S, K, T, r, sigma, q, option_type),
        "rho_pct": rho(S, K, T, r, sigma, q, option_type) / 100.0,  # per 1% rate change
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
        "charm": charm(S, K, T, r, sigma, q, option_type),
        "speed": speed(S, K, T, r, sigma, q),
        "color": color(S, K, T, r, sigma, q),
        "zomma": zomma(S, K, T, r, sigma, q),
    }

    # Intrinsic and extrinsic value
    if option_type.lower() == "call":
        result["intrinsic"] = max(S - K, 0.0)
    else:
        result["intrinsic"] = max(K - S, 0.0)
    result["extrinsic"] = result["price"] - result["intrinsic"]

    # Moneyness metrics
    result["moneyness"] = S / K
    result["log_moneyness"] = np.log(S / K)

    return result


# ---------------------------------------------------------------------------
# Put-Call Parity Check
# ---------------------------------------------------------------------------

def put_call_parity_check(S, K, T, r, q, call_px, put_px):
    """
    Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    Returns the deviation from parity.
    """
    theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual_diff = call_px - put_px
    return {
        "theoretical_diff": theoretical_diff,
        "actual_diff": actual_diff,
        "deviation": actual_diff - theoretical_diff,
    }


# ---------------------------------------------------------------------------
# Strike Solvers (reverse lookup: find strike for a target value)
# ---------------------------------------------------------------------------

def _solve_strike(target_func, target_value, S, T, r, sigma, q, option_type,
                  K_low=None, K_high=None, tol=1e-4, max_iter=200):
    """
    Generic bisection solver to find strike K where target_func(K) = target_value.

    target_func: callable(K) -> float, must be monotonic in K for the given range.
    """
    if K_low is None:
        K_low = S * 0.5
    if K_high is None:
        K_high = S * 1.5

    # Ensure brackets contain the target
    f_low = target_func(K_low) - target_value
    f_high = target_func(K_high) - target_value

    # If target is outside range, extend bounds
    for _ in range(10):
        if f_low * f_high < 0:
            break
        if abs(f_low) > abs(f_high):
            K_low *= 0.7
            if K_low < 1:
                K_low = 1
            f_low = target_func(K_low) - target_value
        else:
            K_high *= 1.3
            f_high = target_func(K_high) - target_value

    if f_low * f_high > 0:
        return np.nan  # Cannot bracket solution

    for _ in range(max_iter):
        K_mid = (K_low + K_high) / 2.0
        f_mid = target_func(K_mid) - target_value

        if abs(f_mid) < tol:
            return K_mid

        if f_low * f_mid < 0:
            K_high = K_mid
            f_high = f_mid
        else:
            K_low = K_mid
            f_low = f_mid

    return (K_low + K_high) / 2.0


def solve_strike_for_price(target_price, S, T, r, sigma, q, option_type):
    """Find strike K such that BS price = target_price."""
    def f(K):
        return option_price(S, K, T, r, sigma, q, option_type)
    # Price decreases with strike for calls, increases for puts
    if option_type.lower() == "call":
        return _solve_strike(f, target_price, S, T, r, sigma, q, option_type,
                             K_low=S * 0.3, K_high=S * 1.5)
    else:
        return _solve_strike(f, target_price, S, T, r, sigma, q, option_type,
                             K_low=S * 0.5, K_high=S * 2.0)


def solve_strike_for_delta(target_delta, S, T, r, sigma, q, option_type):
    """Find strike K such that BS delta = target_delta."""
    def f(K):
        return delta(S, K, T, r, sigma, q, option_type)
    return _solve_strike(f, target_delta, S, T, r, sigma, q, option_type,
                         K_low=S * 0.3, K_high=S * 2.0)


def solve_strike_for_theta(target_theta_daily, S, T, r, sigma, q, option_type):
    """Find strike K such that BS theta_daily = target_theta_daily.

    Theta (absolute value) peaks at ATM and declines OTM/ITM.
    We search on the OTM side for the given option type.
    """
    def f(K):
        return theta(S, K, T, r, sigma, q, option_type) / 365.0

    K_atm = S
    theta_atm = f(K_atm)

    # Theta is negative for long options. If target is more negative than ATM,
    # ATM is the most negative possible -> return ATM.
    if option_type.lower() == "call":
        if target_theta_daily < theta_atm:
            return K_atm
        # OTM calls: K > S, theta becomes less negative
        return _solve_strike(f, target_theta_daily, S, T, r, sigma, q, option_type,
                             K_low=K_atm, K_high=S * 2.0)
    else:
        if target_theta_daily < theta_atm:
            return K_atm
        # OTM puts: K < S, theta becomes less negative
        return _solve_strike(f, target_theta_daily, S, T, r, sigma, q, option_type,
                             K_low=S * 0.3, K_high=K_atm)


def solve_strike_for_vega(target_vega_pct, S, T, r, sigma, q, option_type):
    """Find strike K such that BS vega_pct (per 1% IV) = target_vega_pct."""
    def f(K):
        return vega(S, K, T, r, sigma, q) / 100.0
    # Vega peaks at ATM and declines both directions - not monotonic!
    # So we search near-ATM only.  For OTM targets, two solutions exist;
    # we return the one on the appropriate side (OTM for the option type).
    K_atm = S

    # Vega at ATM
    vega_atm = vega(S, K_atm, T, r, sigma, q) / 100.0
    if target_vega_pct > vega_atm:
        return K_atm  # Max vega is at ATM

    # Search on the correct side for the option type
    if option_type.lower() == "call":
        # OTM calls: K > S
        return _solve_strike(f, target_vega_pct, S, T, r, sigma, q, option_type,
                             K_low=K_atm, K_high=S * 2.0)
    else:
        # OTM puts: K < S
        return _solve_strike(f, target_vega_pct, S, T, r, sigma, q, option_type,
                             K_low=S * 0.3, K_high=K_atm)


# ============================================================================
# OptionStrat URL Generation
# ============================================================================

# OptionStrat ticker mapping
_OPTIONSTRAT_SYMBOLS = {
    "SPX": "SPXW", "GSPC": "SPXW", "^SPX": "SPXW", "^GSPC": "SPXW",
    "NDX": "NDXP", "^NDX": "NDXP",
    "SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "AAPL": "AAPL",
    "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL", "META": "META",
    "TSLA": "TSLA", "NVDA": "NVDA",
}


def optionstrat_url(symbol: str, legs: list) -> str | None:
    """
    Build an OptionStrat URL for any option strategy.

    Args:
        symbol: Underlying symbol (e.g. "SPX", "^SPX", "SPY")
        legs: List of dicts, each with:
            - strike: int or float
            - option_type: "call" or "put" (or "C"/"P")
            - expiration: "YYYY-MM-DD" string
            - long: bool (True = bought, False = sold)

    Returns:
        URL string or None if no legs.

    Example:
        legs = [
            {"strike": 6810, "option_type": "put",  "expiration": "2026-03-19", "long": True},
            {"strike": 6870, "option_type": "put",  "expiration": "2026-03-09", "long": False},
            {"strike": 7000, "option_type": "call", "expiration": "2026-03-09", "long": True},
        ]
        url = optionstrat_url("SPX", legs)
        # https://optionstrat.com/build/custom/SPX/
        #   .SPXW260319P6810,-.SPXW260309P6870,.SPXW260309C7000
    """
    if not legs:
        return None

    sym = symbol.upper().replace("^", "")
    leg_sym = _OPTIONSTRAT_SYMBOLS.get(sym, sym)
    # Also try with ^ prefix
    if leg_sym == sym:
        leg_sym = _OPTIONSTRAT_SYMBOLS.get(f"^{sym}", sym)

    parts = []
    for leg in legs:
        strike = int(float(leg["strike"]))

        # option type: normalise to C/P
        ot = leg.get("option_type", leg.get("type", "C"))
        ot = ot[0].upper()  # "call" -> "C", "put" -> "P"

        # expiration: YYYY-MM-DD -> YYMMDD
        exp = leg["expiration"]
        try:
            from datetime import datetime as _dt
            dt = _dt.strptime(exp, "%Y-%m-%d")
            date_code = dt.strftime("%y%m%d")
        except Exception:
            date_code = exp.replace("-", "")

        prefix = "" if leg.get("long", True) else "-"
        parts.append(f"{prefix}.{leg_sym}{date_code}{ot}{strike}")

    return f"https://optionstrat.com/build/custom/{sym}/{','.join(parts)}"
