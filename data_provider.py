"""
Data provider for options pricing tool.

Fetches from Yahoo Finance:
- Spot price
- Options chains (with IV, bid, ask, volume, OI)
- Dividend yield
- Risk-free rate from Treasury yields
- CBOE SKEW Index, VIX

All Yahoo requests go through a TTL cache to minimise API calls.
Default TTL: 5 minutes for prices, 10 minutes for chains/fundamentals.
"""

import datetime
import time
import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# TTL Cache
# ---------------------------------------------------------------------------

_cache: dict = {}

# TTL in seconds
CACHE_TTL_PRICE = 300       # 5 min for spot, VIX, SKEW
CACHE_TTL_CHAIN = 600       # 10 min for option chains
CACHE_TTL_FUNDAMENTAL = 900  # 15 min for div yield, risk-free rate, RV


def _cache_get(key: str):
    """Return cached value if still valid, else None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expires = entry
    if time.time() > expires:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value, ttl: int):
    """Store value with TTL."""
    _cache[key] = (value, time.time() + ttl)


# ---------------------------------------------------------------------------
# SPX ticker resolution
# ---------------------------------------------------------------------------

SPX_SPOT_TICKERS = ["^SPX", "^GSPC"]
SPX_OPTIONS_TICKERS = ["^SPX", "SPX", "^GSPC"]


def _safe_float(val, default=0.0):
    """Convert to float, returning default for NaN/None."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Convert to int, returning default for NaN/None."""
    f = _safe_float(val, float("nan"))
    return default if np.isnan(f) else int(f)


# ---------------------------------------------------------------------------
# Spot Price
# ---------------------------------------------------------------------------

def get_spot_price(ticker: str) -> float:
    """Get current spot price for a ticker (cached)."""
    key = f"spot:{ticker.upper()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)

    # Try multiple approaches - yfinance API varies across versions
    for attr in ["last_price", "lastPrice", "regularMarketPrice", "previousClose"]:
        try:
            fi = tk.fast_info
            price = fi.get(attr, None) if hasattr(fi, "get") else getattr(fi, attr, None)
            if price is not None and not np.isnan(price) and price > 0:
                _cache_set(key, float(price), CACHE_TTL_PRICE)
                return float(price)
        except Exception:
            continue

    # Fallback: history
    hist = tk.history(period="5d")
    if not hist.empty:
        price = float(hist["Close"].iloc[-1])
        _cache_set(key, price, CACHE_TTL_PRICE)
        return price

    raise ValueError(f"Could not fetch price for {ticker}")


def resolve_spot_price(ticker: str) -> tuple:
    """
    Resolve spot price, trying multiple ticker variants for indices.
    Returns (price, ticker_used).
    """
    if ticker.upper() in ("^SPX", "^GSPC", "SPX"):
        for t in SPX_SPOT_TICKERS:
            try:
                price = get_spot_price(t)
                if price > 1000:
                    return price, t
            except Exception:
                continue
        raise ValueError("Could not fetch SPX spot price from any source")

    return get_spot_price(ticker), ticker


# ---------------------------------------------------------------------------
# Dividend Yield
# ---------------------------------------------------------------------------

def get_dividend_yield(ticker: str) -> float:
    """
    Get annualized dividend yield as a decimal (cached).
    """
    lookup_ticker = "SPY" if ticker.upper() in ("^SPX", "^GSPC", "SPX") else ticker
    key = f"divyield:{lookup_ticker.upper()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    tk = yf.Ticker(lookup_ticker)
    try:
        info = tk.info

        # Try yield fields first (these are decimals: 0.0124 = 1.24%)
        for field in ["dividendYield", "trailingAnnualDividendYield"]:
            val = info.get(field)
            if val is not None:
                val = float(val)
                if not np.isnan(val) and val > 0:
                    # Sanity check: yield should be < 0.20 (20%)
                    # If it's > 1.0, Yahoo probably returned a percentage (1.24 = 1.24%)
                    # If it's > 0.20, it's likely a dollar amount or error
                    if val > 1.0:
                        # Likely a percentage value (e.g., 1.24 means 1.24%)
                        val = val / 100.0
                    if val > 0.20:
                        # Still too high - skip this value
                        continue
                    _cache_set(key, val, CACHE_TTL_FUNDAMENTAL)
                    return val

        # Fallback: compute from dividendRate / price if available
        rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if rate is not None and price is not None:
            rate = float(rate)
            price = float(price)
            if not np.isnan(rate) and not np.isnan(price) and price > 0 and rate > 0:
                computed_yield = rate / price
                if computed_yield < 0.20:
                    _cache_set(key, computed_yield, CACHE_TTL_FUNDAMENTAL)
                    return computed_yield

    except Exception:
        pass
    _cache_set(key, 0.0, CACHE_TTL_FUNDAMENTAL)
    return 0.0


# ---------------------------------------------------------------------------
# Risk-Free Rate
# ---------------------------------------------------------------------------

def get_risk_free_rate(dte_days: int) -> float:
    """
    Get risk-free rate from Treasury yields (cached).
    """
    brackets = [
        (180,  "^IRX"),
        (730,  "^FVX"),
        (2555, "^TNX"),
        (9999, "^TYX"),
    ]

    target_ticker = "^IRX"
    for max_dte, ticker in brackets:
        if dte_days <= max_dte:
            target_ticker = ticker
            break

    key = f"rfrate:{target_ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(target_ticker)
        hist = tk.history(period="5d")
        if not hist.empty:
            raw_val = float(hist["Close"].iloc[-1])
            rate = raw_val / 100.0
            if 0.0 < rate < 0.20:
                _cache_set(key, rate, CACHE_TTL_FUNDAMENTAL)
                return rate
    except Exception:
        pass

    return 0.045


# ---------------------------------------------------------------------------
# Market Indicators
# ---------------------------------------------------------------------------

def get_skew_index() -> float:
    """Get current CBOE SKEW Index value (cached)."""
    cached = _cache_get("skew")
    if cached is not None:
        return cached
    try:
        hist = yf.Ticker("^SKEW").history(period="5d")
        if not hist.empty:
            val = float(hist["Close"].iloc[-1])
            _cache_set("skew", val, CACHE_TTL_PRICE)
            return val
    except Exception:
        pass
    return np.nan


def get_vix() -> float:
    """Get current VIX value (cached)."""
    cached = _cache_get("vix")
    if cached is not None:
        return cached
    try:
        hist = yf.Ticker("^VIX").history(period="5d")
        if not hist.empty:
            val = float(hist["Close"].iloc[-1])
            _cache_set("vix", val, CACHE_TTL_PRICE)
            return val
    except Exception:
        pass
    return np.nan


# ---------------------------------------------------------------------------
# Options Chain
# ---------------------------------------------------------------------------

def get_options_chain(ticker: str, dte_target: int) -> dict:
    """Fetch options chain for the expiration closest to the target DTE (cached)."""
    key = f"chain:{ticker.upper()}:{dte_target}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)

    try:
        expirations = tk.options
    except Exception as e:
        raise ValueError(f"Could not fetch options for {ticker}: {e}")

    if not expirations:
        raise ValueError(f"No options expirations available for {ticker}")

    today = datetime.date.today()
    target_date = today + datetime.timedelta(days=dte_target)

    best_exp = None
    best_diff = float("inf")

    for exp_str in expirations:
        exp_date = datetime.date.fromisoformat(exp_str)
        diff = abs((exp_date - target_date).days)
        if diff < best_diff:
            best_diff = diff
            best_exp = exp_str

    if best_exp is None:
        raise ValueError(f"Could not find suitable expiration for {ticker}")

    chain = tk.option_chain(best_exp)
    exp_date = datetime.date.fromisoformat(best_exp)
    actual_dte = max((exp_date - today).days, 1)

    result = {
        "expiration": best_exp,
        "dte_actual": actual_dte,
        "dte_years": actual_dte / 365.0,
        "calls": chain.calls,
        "puts": chain.puts,
    }
    _cache_set(key, result, CACHE_TTL_CHAIN)
    return result


def resolve_options_chain(ticker: str, dte: int) -> tuple:
    """
    Try to get options chain, with SPX fallback logic.
    Returns (chain, ticker_used, is_spy_fallback).
    """
    is_spx = ticker.upper() in ("^SPX", "^GSPC", "SPX")

    if is_spx:
        for t in SPX_OPTIONS_TICKERS:
            try:
                chain = get_options_chain(t, dte)
                if not chain["calls"].empty:
                    return chain, t, False
            except Exception:
                continue
        # Fallback to SPY
        try:
            chain = get_options_chain("SPY", dte)
            return chain, "SPY", True
        except Exception as e:
            raise ValueError(f"Could not fetch options for SPX or SPY: {e}")

    return get_options_chain(ticker, dte), ticker, False


# ---------------------------------------------------------------------------
# Option Lookup and Market Prices
# ---------------------------------------------------------------------------

def find_option_by_strike(chain_df: pd.DataFrame, strike: float):
    """Find the option closest to the given strike."""
    if chain_df is None or chain_df.empty:
        return None
    idx = (chain_df["strike"] - strike).abs().idxmin()
    return chain_df.loc[idx]


def extract_market_prices(opt) -> dict:
    """Extract market prices from an option row, handling NaN safely."""
    if opt is None:
        return {
            "bid": np.nan, "ask": np.nan, "mid": np.nan, "last": np.nan,
            "volume": 0, "open_interest": 0, "market_iv": np.nan,
            "strike": np.nan,
        }

    bid = _safe_float(opt.get("bid"))
    ask = _safe_float(opt.get("ask"))
    last = _safe_float(opt.get("lastPrice"))
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "last": last,
        "volume": _safe_int(opt.get("volume")),
        "open_interest": _safe_int(opt.get("openInterest")),
        "market_iv": _safe_float(opt.get("impliedVolatility"), np.nan),
        "strike": _safe_float(opt.get("strike"), np.nan),
    }


# ---------------------------------------------------------------------------
# IV Smile
# ---------------------------------------------------------------------------

def build_smile_curve(chain_df: pd.DataFrame, spot: float,
                      min_oi: int = 10, iv_cap: float = 2.0,
                      smooth: bool = True) -> pd.DataFrame:
    """
    Build IV smile curve from chain data with filtering and smoothing.

    Filtering:
    - Remove strikes with IV <= 0 or IV > iv_cap (default 200%)
    - Remove strikes with open interest < min_oi (illiquid)
    - Remove strikes where bid = 0 (no real market)

    Smoothing (if enabled):
    - Savitzky-Golay filter (preserves shape better than moving average)
    - Window size adapts to number of data points
    - Original raw data preserved in 'iv_raw' column
    """
    if chain_df is None or chain_df.empty:
        return pd.DataFrame(columns=["strike", "iv", "iv_raw", "moneyness",
                                     "log_moneyness"])

    df = chain_df[["strike", "impliedVolatility"]].copy()
    df = df.rename(columns={"impliedVolatility": "iv"})

    # Add OI and bid for filtering
    if "openInterest" in chain_df.columns:
        df["oi"] = chain_df["openInterest"].apply(
            lambda x: _safe_int(x, 0)
        )
    else:
        df["oi"] = 0

    if "bid" in chain_df.columns:
        df["bid"] = chain_df["bid"].apply(lambda x: _safe_float(x, 0))
    else:
        df["bid"] = 0

    # Filter
    df = df[df["iv"] > 0.001]
    df = df[df["iv"] <= iv_cap]
    df = df[df["oi"] >= min_oi]
    df = df[df["bid"] > 0]

    df = df.sort_values("strike").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["strike", "iv", "iv_raw", "moneyness",
                                     "log_moneyness"])

    # Remove outliers: IV that deviates > 3 sigma from local neighbors
    df["iv_raw"] = df["iv"].copy()
    if len(df) >= 5:
        rolling_med = df["iv"].rolling(5, center=True, min_periods=3).median()
        rolling_std = df["iv"].rolling(5, center=True, min_periods=3).std()
        rolling_std = rolling_std.clip(lower=0.01)  # minimum std
        outlier = (df["iv"] - rolling_med).abs() > 3 * rolling_std
        df.loc[outlier, "iv"] = rolling_med[outlier]

    # Smooth with Savitzky-Golay
    if smooth and len(df) >= 7:
        from scipy.signal import savgol_filter
        # Window must be odd, at most len(df), and >= polyorder+2
        window = min(len(df), 11)
        if window % 2 == 0:
            window -= 1
        window = max(window, 5)
        if window <= len(df):
            df["iv"] = savgol_filter(df["iv"].values, window, polyorder=3)
            # Ensure IV stays positive after smoothing
            df["iv"] = df["iv"].clip(lower=0.001)

    df["moneyness"] = spot / df["strike"]
    df["log_moneyness"] = np.log(df["moneyness"])

    # Drop helper columns
    df = df.drop(columns=["oi", "bid"], errors="ignore")

    return df.reset_index(drop=True)


def interpolate_smile_iv(smile_df: pd.DataFrame, strike: float) -> float:
    """Interpolate IV for a given strike from the smile curve."""
    if smile_df.empty:
        return np.nan

    strikes = smile_df["strike"].values
    ivs = smile_df["iv"].values

    if len(strikes) < 2:
        return float(ivs[0]) if len(ivs) == 1 else np.nan

    if strike <= strikes[0]:
        return float(ivs[0])
    if strike >= strikes[-1]:
        return float(ivs[-1])

    return float(np.interp(strike, strikes, ivs))


def get_atm_iv(chain: dict, spot: float) -> dict:
    """Get ATM implied volatility from the chain."""
    call_opt = find_option_by_strike(chain["calls"], spot)
    put_opt = find_option_by_strike(chain["puts"], spot)

    call_iv = _safe_float(call_opt.get("impliedVolatility"), np.nan) if call_opt is not None else np.nan
    put_iv = _safe_float(put_opt.get("impliedVolatility"), np.nan) if put_opt is not None else np.nan

    return {
        "atm_strike": _safe_float(call_opt.get("strike"), spot) if call_opt is not None else spot,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "avg_iv": np.nanmean([call_iv, put_iv]),
    }


# ---------------------------------------------------------------------------
# Implied Forward / Implied Spot from Put-Call Parity
# ---------------------------------------------------------------------------

def compute_implied_spot(chain: dict, spot_estimate: float, r: float, q: float,
                         T: float) -> dict:
    """
    Compute implied spot from put-call parity using ATM options.

    Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    => S = (C - P + K*e^(-rT)) / e^(-qT)

    Uses the strike closest to spot for maximum reliability.
    Also tries multiple strikes near ATM and averages for robustness.

    Returns dict with implied_spot, strikes_used, and individual estimates.
    """
    calls = chain["calls"]
    puts = chain["puts"]

    if calls.empty or puts.empty:
        return {"implied_spot": spot_estimate, "method": "fallback", "estimates": []}

    # Find strikes that exist in both call and put chains
    call_strikes = set(calls["strike"].values)
    put_strikes = set(puts["strike"].values)
    common_strikes = sorted(call_strikes & put_strikes)

    if not common_strikes:
        return {"implied_spot": spot_estimate, "method": "fallback", "estimates": []}

    # Pick strikes near ATM (within 5% of spot)
    atm_range = spot_estimate * 0.05
    near_atm = [k for k in common_strikes
                if abs(k - spot_estimate) < atm_range]

    if not near_atm:
        # Just use the closest strike
        near_atm = [min(common_strikes, key=lambda k: abs(k - spot_estimate))]

    estimates = []
    for K in near_atm:
        call_row = calls[calls["strike"] == K].iloc[0]
        put_row = puts[puts["strike"] == K].iloc[0]

        c_bid = _safe_float(call_row.get("bid"))
        c_ask = _safe_float(call_row.get("ask"))
        p_bid = _safe_float(put_row.get("bid"))
        p_ask = _safe_float(put_row.get("ask"))

        # Use mid prices if both bid/ask are valid
        c_mid = (c_bid + c_ask) / 2.0 if c_bid > 0 and c_ask > 0 else _safe_float(call_row.get("lastPrice"))
        p_mid = (p_bid + p_ask) / 2.0 if p_bid > 0 and p_ask > 0 else _safe_float(put_row.get("lastPrice"))

        if c_mid <= 0 or p_mid <= 0:
            continue

        # S = (C - P + K*e^(-rT)) / e^(-qT)
        implied_s = (c_mid - p_mid + K * np.exp(-r * T)) / np.exp(-q * T)

        if implied_s > 0:
            estimates.append({"strike": K, "call_mid": c_mid, "put_mid": p_mid,
                              "implied_spot": implied_s})

    if not estimates:
        return {"implied_spot": spot_estimate, "method": "fallback", "estimates": []}

    # Use median to be robust against outliers
    implied_spots = [e["implied_spot"] for e in estimates]
    median_spot = float(np.median(implied_spots))

    return {
        "implied_spot": median_spot,
        "method": "put_call_parity",
        "n_strikes": len(estimates),
        "estimates": estimates,
    }


# ---------------------------------------------------------------------------
# Main Data Fetch
# ---------------------------------------------------------------------------

def fetch_all_data(ticker: str, strike: float, dte: int) -> dict:
    """
    Fetch all data needed for option pricing.

    For SPX: tries SPX options chain directly. If unavailable, falls back
    to SPY. When using SPY fallback, ALL data (spot, strikes, prices) are
    reported in SPY space and scaled back to SPX level using the SPX/SPY ratio.

    Key principle: BS calculation always uses consistent (spot, strike, price)
    from the same source. The IV is the same regardless of scaling.
    """
    is_spx = ticker.upper() in ("^SPX", "^GSPC", "SPX")

    # 1. Get the actual SPX spot price (for display)
    spot, spot_ticker = resolve_spot_price(ticker)

    # 2. Get options chain (may fall back to SPY)
    chain, chain_ticker, spy_fallback = resolve_options_chain(ticker, dte)

    # 3. Determine working parameters
    if spy_fallback:
        # Work in SPY space, then scale results back to SPX
        spy_spot = get_spot_price("SPY")
        scale = spot / spy_spot
        working_strike = strike / scale
        working_spot = spy_spot
    else:
        scale = 1.0
        working_strike = strike
        working_spot = spot

    # 4. Find options by strike (in chain's space)
    call_opt = find_option_by_strike(chain["calls"], working_strike)
    put_opt = find_option_by_strike(chain["puts"], working_strike)

    call_market = extract_market_prices(call_opt)
    put_market = extract_market_prices(put_opt)

    # 5. Scale prices back to original ticker space if needed
    if spy_fallback:
        for mkt in [call_market, put_market]:
            for key in ["bid", "ask", "mid", "last", "strike"]:
                if not np.isnan(mkt[key]):
                    mkt[key] *= scale
        # IV is NOT scaled - same in percentage terms

    # 6. ATM IV and smile (in chain space)
    atm_iv = get_atm_iv(chain, working_spot)

    call_smile = build_smile_curve(chain["calls"], working_spot)
    put_smile = build_smile_curve(chain["puts"], working_spot)

    call_smile_iv = interpolate_smile_iv(call_smile, working_strike)
    put_smile_iv = interpolate_smile_iv(put_smile, working_strike)

    # Scale smile strike axis for display
    if spy_fallback:
        if not call_smile.empty:
            call_smile = call_smile.copy()
            call_smile["strike"] = call_smile["strike"] * scale
        if not put_smile.empty:
            put_smile = put_smile.copy()
            put_smile["strike"] = put_smile["strike"] * scale

    # 7. Other parameters
    div_yield = get_dividend_yield(ticker)
    risk_free = get_risk_free_rate(chain["dte_actual"])
    skew = get_skew_index()
    vix = get_vix()

    # 8. Implied spot from put-call parity (for accurate IV derivation)
    implied_spot_data = compute_implied_spot(
        chain, working_spot, risk_free, div_yield, chain["dte_years"]
    )
    implied_spot_raw = implied_spot_data["implied_spot"]
    implied_spot = implied_spot_raw * scale if spy_fallback else implied_spot_raw

    return {
        "ticker": ticker,
        "chain_ticker": chain_ticker,
        "spy_fallback": spy_fallback,
        "scale_factor": scale,
        "spot": spot,
        "implied_spot": implied_spot,
        "implied_spot_method": implied_spot_data["method"],
        "requested_strike": strike,
        "actual_call_strike": call_market["strike"],
        "actual_put_strike": put_market["strike"],
        "requested_dte": dte,
        "expiration": chain["expiration"],
        "actual_dte": chain["dte_actual"],
        "dte_years": chain["dte_years"],
        "dividend_yield": div_yield,
        "risk_free_rate": risk_free,
        "skew_index": skew,
        "vix": vix,
        "call_market": call_market,
        "put_market": put_market,
        "atm_iv": atm_iv,
        "call_market_iv": call_market["market_iv"],
        "put_market_iv": put_market["market_iv"],
        "call_smile_iv": call_smile_iv,
        "put_smile_iv": put_smile_iv,
        "call_smile": call_smile,
        "put_smile": put_smile,
    }


# ---------------------------------------------------------------------------
# Multi-Expiration Chain Fetcher
# ---------------------------------------------------------------------------

def get_available_expirations(ticker: str) -> tuple:
    """Get all available expiration date strings for a ticker (cached)."""
    key = f"expirations:{ticker.upper()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    is_spx = ticker.upper() in ("^SPX", "^GSPC", "SPX")
    tickers_to_try = SPX_OPTIONS_TICKERS + ["SPY"] if is_spx else [ticker]

    for t in tickers_to_try:
        try:
            tk = yf.Ticker(t)
            exps = tk.options
            if exps:
                result = (list(exps), t)
                _cache_set(key, result, CACHE_TTL_CHAIN)
                return result
        except Exception:
            continue
    return [], ticker


def fetch_chains_for_dte_range(ticker: str, dte_min: int, dte_max: int) -> tuple:
    """
    Fetch options chains for all expirations within the DTE range.

    Returns (results_list, spot, working_spot, scale) where results_list
    contains dicts with: expiration, dte, dte_years, call_smile, put_smile.
    """
    is_spx = ticker.upper() in ("^SPX", "^GSPC", "SPX")
    expirations, chain_ticker = get_available_expirations(ticker)

    if not expirations:
        raise ValueError(f"No expirations available for {ticker}")

    spy_fallback = is_spx and chain_ticker == "SPY"

    spot, _ = resolve_spot_price(ticker)
    if spy_fallback:
        working_spot = get_spot_price("SPY")
        scale = spot / working_spot
    else:
        working_spot = spot
        scale = 1.0

    today = datetime.date.today()
    results = []
    tk = yf.Ticker(chain_ticker)

    for exp_str in expirations:
        exp_date = datetime.date.fromisoformat(exp_str)
        dte_actual = (exp_date - today).days
        if dte_actual < dte_min or dte_actual > dte_max:
            continue

        # Cache individual chain by ticker + expiration
        chain_key = f"rawchain:{chain_ticker.upper()}:{exp_str}"
        chain_data = _cache_get(chain_key)
        if chain_data is None:
            try:
                chain = tk.option_chain(exp_str)
            except Exception:
                continue
            if chain.calls.empty and chain.puts.empty:
                continue
            chain_data = (chain.calls, chain.puts)
            _cache_set(chain_key, chain_data, CACHE_TTL_CHAIN)

        calls_df, puts_df = chain_data

        call_smile = build_smile_curve(calls_df, working_spot)
        put_smile = build_smile_curve(puts_df, working_spot)

        if spy_fallback:
            if not call_smile.empty:
                call_smile = call_smile.copy()
                call_smile["strike"] = call_smile["strike"] * scale
            if not put_smile.empty:
                put_smile = put_smile.copy()
                put_smile["strike"] = put_smile["strike"] * scale

        results.append({
            "expiration": exp_str,
            "dte": dte_actual,
            "dte_years": dte_actual / 365.0,
            "call_smile": call_smile,
            "put_smile": put_smile,
        })

    return results, spot, working_spot, scale


# ---------------------------------------------------------------------------
# Realized Volatility
# ---------------------------------------------------------------------------

def get_realized_volatility(ticker: str, windows: list = None) -> dict:
    """
    Compute realized (historical) volatility from daily close prices (cached).
    """
    if windows is None:
        windows = [20, 30]

    # Resolve ticker for history
    sym = ticker.upper()
    if sym in ("^SPX", "SPX"):
        hist_ticker = "^GSPC"
    else:
        hist_ticker = ticker

    key = f"rv:{hist_ticker.upper()}:{','.join(str(w) for w in windows)}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    max_window = max(windows) + 10  # extra days for safety
    try:
        tk = yf.Ticker(hist_ticker)
        hist = tk.history(period=f"{max_window + 30}d")
    except Exception:
        return {f"rv_{w}": np.nan for w in windows}

    if hist.empty or len(hist) < max(windows) + 1:
        return {f"rv_{w}": np.nan for w in windows}

    closes = hist["Close"].values
    log_returns = np.log(closes[1:] / closes[:-1])

    result = {}
    for w in windows:
        if len(log_returns) >= w:
            recent = log_returns[-w:]
            rv = float(np.std(recent) * np.sqrt(252))  # annualized
            result[f"rv_{w}"] = rv
        else:
            result[f"rv_{w}"] = np.nan

    result["daily_returns"] = log_returns
    _cache_set(key, result, CACHE_TTL_FUNDAMENTAL)
    return result
