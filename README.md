# Options Pricing Engine - Milestone 1

Validated Black-Scholes options pricing engine with Yahoo Finance data.
Serves as the foundation for all future options tools.

## Status

Phase 1 complete. BS calculations validated against OptionStrat:
- OTM options: IV, Delta, Theta, Vega within 5-10% of OptionStrat
- ITM options: Less accurate due to Yahoo's 15-min delayed data and wide spreads
- All Greeks verified against finite differences and Hull textbook values

## Architecture

```
bs_engine.py        Core BS calculations, all Greeks, IV solver
data_provider.py    Yahoo Finance data, implied spot, smile curves
app.py              Streamlit UI (Phase 1: pricing validation)
test_bs_engine.py   Validation tests (run standalone, no network needed)
```

## Key Design Decisions

1. **Implied Spot via Put-Call Parity**: Yahoo's delayed spot causes IV errors
   for ITM options. The implied spot (derived from ATM put-call parity across
   multiple strikes, median) is consistent with option prices.

2. **Dual IV Display**: Method A (Implied Spot + Derived IV from Mid) vs
   Method B (Yahoo Spot + Yahoo IV) for comparison.

3. **Dividend Yield Sanitization**: Yahoo returns inconsistent formats
   (decimal vs percentage vs dollar amount). The provider auto-corrects.

4. **European BS Model**: Correct for SPX. Approximation for American-style
   (SPY, single stocks). Acceptable for Phase 1.

## Greeks Computed

First order: Delta, Gamma, Theta, Vega, Rho
Second order: Vanna, Volga/Vomma, Charm, Speed, Color, Zomma

## Known Limitations

- Yahoo data is 15-min delayed; ITM options with wide spreads are less accurate
- SPX options chain may not be available via Yahoo; falls back to SPY with scaling
- American-style exercise premium not modeled (European BS only)
- No real-time streaming; manual "Calculate" button

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validation

```bash
python test_bs_engine.py
```

## Future Integration Plan

This engine will serve as the base for:
- Volatility scenario analyzer (optimal strike/DTE during vol spikes)
- Crash protection analyzer (sweet spot for OTM puts)
- Portfolio hedging calculator
- Options scanner with parameter sliders (Phase 2)
- IBKR API integration for real-time data
