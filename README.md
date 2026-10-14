# Options Market-Making and Volatility Simulator

An educational Python project comparing three simplified options market-making strategies:

1. **Fixed spread:** constant-width quotes around Black–Scholes fair value.
2. **Inventory aware:** shifts both quotes according to current call inventory.
3. **Inventory and volatility aware:** also widens the spread when simulated volatility rises above its baseline.

## Research question

Can quote adjustments based on inventory and volatility reduce inventory exposure and extreme losses without sacrificing too much profit relative to a fixed-spread strategy?

## What the model does

- Simulates a share price using geometric Brownian motion and a positive, mean-reverting volatility path.
- Prices European calls and puts with Black–Scholes and calculates delta, gamma, vega and daily theta.
- Generates one potential customer valuation per active trading day.
- Gives every strategy the same path and customer shocks for a controlled comparison.
- Tracks cash, call inventory and daily marked-to-market P&L.
- Settles remaining inventory at intrinsic value on expiry.
- Measures P&L volatility, drawdown, inventory exposure and trade count.
- Repeats the comparison across many seeds and tests parameter sensitivity.
- Provides an interactive Streamlit dashboard.

## Quoting logic

```text
fixed bid = fair value - spread / 2
fixed ask = fair value + spread / 2

inventory shift = inventory sensitivity × inventory
adjusted centre = fair value - inventory shift
```

Positive inventory lowers both quotes, encouraging customers to buy calls from the market maker. Negative inventory raises both quotes, encouraging customers to sell calls to the market maker. This creates a tendency for inventory to return towards zero.

The volatility-aware strategy also widens the spread when simulated volatility exceeds its baseline. This is an illustrative risk-control rule, not a calibrated representation of a trading desk.

## Customer model

```text
customer value = Black-Scholes fair value + Z × customer noise
Z ~ Normal(0, 1)
```

The customer buys at the ask when their value is at least the ask, sells at the bid when it is at most the bid, and otherwise does not trade.

## Run the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulation.py
streamlit run app.py
```

Run the validation tests with:

```bash
python -m pytest test_simulator.py
```

`simulation.py` saves charts and CSV tables in `outputs/`.

## Metrics

- Final marked-to-market P&L
- Standard deviation of daily P&L changes
- Maximum drawdown
- Final, maximum absolute and average absolute inventory
- Trade and customer-action counts
- A labelled Sharpe-like path-comparison statistic
- Mean P&L, loss probability and inventory exposure over repeated simulations

## Interpretation

One random seed is an example, not proof that a strategy is superior. The repeated-simulation and sensitivity outputs are the primary comparison, but they remain conditional on the model assumptions and are not estimates of real-world returns.

## Scope and limitations

- European options only; the quoting strategies currently trade calls only.
- One option unit per order; no contract multiplier or variable order size.
- One potential customer per day with normally distributed valuation noise.
- No delta hedging, fees, capital constraints, latency, order book or market impact.
- Black–Scholes assumptions are simplified and volatility is not calibrated to market data.
- Strategy parameters are illustrative rather than production-optimised.
- Daily observations omit intraday price and order-flow dynamics.

Day 0 is the starting observation with 126 trading days remaining. Days 0–125 are active quoting periods. Day 126 represents expiry and settlement rather than another trading period.

## Development approach and AI use

I developed the initial pricing and simulation model, then used generative AI as a development aid to extend and refactor the project, debug implementation issues and challenge modelling assumptions. I reviewed the resulting implementation, validated key Black–Scholes outputs with tests and documented what the results do and do not establish.

## Disclaimer

This project is for education and portfolio demonstration only. It is not investment advice or a production trading system.
