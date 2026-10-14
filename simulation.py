"""Compare simplified fixed, inventory-aware and volatility-aware option quotes."""

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing import black_scholes_prices, calculate_greeks


@dataclass(frozen=True)
class SimulationConfig:
    initial_share_price: float = 100.0
    strike_price: float = 100.0
    expected_return: float = 0.05
    base_volatility: float = 0.20
    interest_rate: float = 0.05
    initial_time_to_expiry: float = 0.5
    num_days: int = 126
    trading_days_per_year: int = 252
    fixed_spread: float = 0.20
    inventory_sensitivity: float = 0.02
    volatility_sensitivity: float = 1.50
    customer_noise: float = 0.25
    volatility_mean_reversion: float = 4.0
    volatility_of_volatility: float = 0.18
    minimum_volatility: float = 0.05


STRATEGY_LABELS = {
    "fixed": "Fixed spread",
    "inventory": "Inventory aware",
    "inventory_volatility": "Inventory and volatility aware",
}


def simulate_market(config=SimulationConfig(), seed=73):
    """Simulate one stock/volatility path and common customer valuations."""
    rng = np.random.default_rng(seed)
    dt = 1 / config.trading_days_per_year
    prices = [config.initial_share_price]
    volatilities = [config.base_volatility]
    times = [config.initial_time_to_expiry]

    for day in range(config.num_days):
        current_price = prices[-1]
        current_volatility = volatilities[-1]
        stock_return = (
            config.expected_return - 0.5 * current_volatility**2
        ) * dt + current_volatility * np.sqrt(dt) * rng.normal()
        prices.append(current_price * np.exp(stock_return))

        next_volatility = current_volatility + (
            config.volatility_mean_reversion
            * (config.base_volatility - current_volatility)
            * dt
            + config.volatility_of_volatility
            * current_volatility
            * np.sqrt(dt)
            * rng.normal()
        )
        volatilities.append(max(next_volatility, config.minimum_volatility))
        times.append(max(config.initial_time_to_expiry - (day + 1) * dt, 0.0))

    calls, puts, greek_rows = [], [], []
    for price, time_remaining, volatility in zip(prices, times, volatilities):
        call, put = black_scholes_prices(
            price, config.strike_price, config.interest_rate, time_remaining, volatility
        )
        calls.append(call)
        puts.append(put)
        greek_rows.append(
            calculate_greeks(
                price, config.strike_price, config.interest_rate, time_remaining, volatility
            )
        )

    market = pd.DataFrame(
        {
            "day": np.arange(config.num_days + 1),
            "share_price": prices,
            "volatility": volatilities,
            "time_to_expiry": times,
            "call_fair_value": calls,
            "put_fair_value": puts,
        }
    )
    market = pd.concat([market, pd.DataFrame(greek_rows)], axis=1)
    market.loc[: config.num_days - 1, "customer_value"] = (
        market.loc[: config.num_days - 1, "call_fair_value"].to_numpy()
        + rng.normal(size=config.num_days) * config.customer_noise
    )
    return market


def calculate_quote(fair_value, volatility, inventory, strategy, config):
    """Calculate one bid/ask pair before observing the customer's decision."""
    inventory_shift = (
        config.inventory_sensitivity * inventory
        if strategy in {"inventory", "inventory_volatility"}
        else 0.0
    )
    spread = config.fixed_spread
    if strategy == "inventory_volatility":
        relative_increase = max(volatility / config.base_volatility - 1.0, 0.0)
        spread *= 1.0 + config.volatility_sensitivity * relative_increase

    centre = fair_value - inventory_shift
    bid = max(centre - spread / 2, 0.0)
    ask = max(centre + spread / 2, bid)
    return bid, ask, centre, spread


def run_strategy(market, strategy, config=SimulationConfig()):
    """Run one strategy and return its cash, inventory and P&L history."""
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"Unknown strategy: {strategy}")

    inventory, cash = 0, 0.0
    rows = [{
        "day": 0, "inventory": inventory, "cash": cash,
        "marked_to_market_pnl": cash,
        "bid": np.nan, "ask": np.nan, "quote_centre": np.nan,
        "quoted_spread": np.nan, "action": "Start",
    }]

    for day in range(config.num_days):
        bid, ask, centre, spread = calculate_quote(
            market.loc[day, "call_fair_value"],
            market.loc[day, "volatility"],
            inventory,
            strategy,
            config,
        )
        customer_value = market.loc[day, "customer_value"]
        if customer_value >= ask:
            action = "Customer buys"
            inventory -= 1
            cash += ask
        elif customer_value <= bid:
            action = "Customer sells"
            inventory += 1
            cash -= bid
        else:
            action = "No trade"

        next_call_value = market.loc[day + 1, "call_fair_value"]
        rows.append({
            "day": day + 1, "inventory": inventory, "cash": cash,
            "marked_to_market_pnl": cash + inventory * next_call_value,
            "bid": bid, "ask": ask, "quote_centre": centre,
            "quoted_spread": spread, "action": action,
        })

    history = pd.DataFrame(rows)
    history["strategy"] = STRATEGY_LABELS[strategy]
    return history


def calculate_metrics(history):
    pnl = history["marked_to_market_pnl"]
    changes = pnl.diff().dropna()
    drawdown = pnl - pnl.cummax()
    daily_volatility = changes.std(ddof=1)
    sharpe_like = (
        np.sqrt(252) * changes.mean() / daily_volatility
        if daily_volatility > 0 else np.nan
    )
    actions = history.loc[history["action"] != "Start", "action"]
    return {
        "strategy": history.loc[0, "strategy"],
        "final_pnl": pnl.iloc[-1],
        "pnl_volatility": daily_volatility,
        "maximum_drawdown": drawdown.min(),
        "final_inventory": history["inventory"].iloc[-1],
        "maximum_absolute_inventory": history["inventory"].abs().max(),
        "average_absolute_inventory": history["inventory"].abs().mean(),
        "trade_count": actions.isin(["Customer buys", "Customer sells"]).sum(),
        "customer_buys": (actions == "Customer buys").sum(),
        "customer_sells": (actions == "Customer sells").sum(),
        "no_trades": (actions == "No trade").sum(),
        "sharpe_like": sharpe_like,
    }


def run_single_simulation(config=SimulationConfig(), seed=73):
    market = simulate_market(config, seed)
    histories = {
        name: run_strategy(market, name, config) for name in STRATEGY_LABELS
    }
    metrics = pd.DataFrame(
        [calculate_metrics(history) for history in histories.values()]
    ).set_index("strategy")
    return market, histories, metrics


def run_many_simulations(config=SimulationConfig(), number_of_seeds=250):
    records = []
    for seed in range(number_of_seeds):
        _, _, metrics = run_single_simulation(config, seed)
        for strategy, row in metrics.iterrows():
            records.append({"seed": seed, "strategy": strategy, **row.to_dict()})
    results = pd.DataFrame(records)
    summary = results.groupby("strategy").agg(
        mean_final_pnl=("final_pnl", "mean"),
        median_final_pnl=("final_pnl", "median"),
        pnl_standard_deviation=("final_pnl", "std"),
        loss_probability=("final_pnl", lambda values: (values < 0).mean()),
        mean_maximum_drawdown=("maximum_drawdown", "mean"),
        mean_maximum_absolute_inventory=("maximum_absolute_inventory", "mean"),
        mean_average_absolute_inventory=("average_absolute_inventory", "mean"),
        mean_trade_count=("trade_count", "mean"),
    )
    return results, summary


def run_parameter_sensitivity(
    spreads=(0.10, 0.20, 0.30),
    inventory_sensitivities=(0.01, 0.02, 0.04),
    number_of_seeds=100,
    base_config=SimulationConfig(),
):
    records = []
    for spread in spreads:
        for sensitivity in inventory_sensitivities:
            values = asdict(base_config)
            values.update(fixed_spread=spread, inventory_sensitivity=sensitivity)
            results, _ = run_many_simulations(SimulationConfig(**values), number_of_seeds)
            selected = results[results["strategy"] == STRATEGY_LABELS["inventory"]]
            records.append({
                "fixed_spread": spread,
                "inventory_sensitivity": sensitivity,
                "mean_final_pnl": selected["final_pnl"].mean(),
                "loss_probability": (selected["final_pnl"] < 0).mean(),
                "mean_maximum_absolute_inventory": selected["maximum_absolute_inventory"].mean(),
            })
    return pd.DataFrame(records)


def create_figures(market, histories):
    figures = {}
    specifications = {
        "share_price": ("Simulated share-price path", "Share price (£)"),
        "option_values": ("European option fair values", "Option value (£)"),
        "inventory": ("Call inventory by quoting strategy", "Call inventory"),
        "pnl": ("Marked-to-market P&L by quoting strategy", "P&L (£)"),
    }
    for name, (title, ylabel) in specifications.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        if name == "share_price":
            ax.plot(market["day"], market["share_price"])
        elif name == "option_values":
            ax.plot(market["day"], market["call_fair_value"], label="Call")
            ax.plot(market["day"], market["put_fair_value"], label="Put")
            ax.legend()
        else:
            column = "inventory" if name == "inventory" else "marked_to_market_pnl"
            for history in histories.values():
                ax.plot(history["day"], history[column], label=history.loc[0, "strategy"])
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.legend()
        ax.set(title=title, xlabel="Trading day", ylabel=ylabel)
        ax.grid(alpha=0.3)
        figures[name] = fig
    return figures


def save_outputs(output_directory="outputs", seed=73, number_of_seeds=250):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig()
    market, histories, metrics = run_single_simulation(config, seed)
    all_results, repeated_summary = run_many_simulations(config, number_of_seeds)
    sensitivity = run_parameter_sensitivity(number_of_seeds=min(number_of_seeds, 100))
    market.to_csv(output_path / "market_path.csv", index=False)
    metrics.to_csv(output_path / "single_path_metrics.csv")
    all_results.to_csv(output_path / "multi_seed_results.csv", index=False)
    repeated_summary.to_csv(output_path / "multi_seed_summary.csv")
    sensitivity.to_csv(output_path / "parameter_sensitivity.csv", index=False)
    for name, figure in create_figures(market, histories).items():
        figure.savefig(output_path / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    return metrics, repeated_summary, sensitivity


if __name__ == "__main__":
    single, repeated, sensitivity = save_outputs()
    pd.set_option("display.max_columns", None)
    print("\nSingle simulated path (seed 73)\n", single.round(3))
    print("\nRepeated-simulation summary\n", repeated.round(3))
    print("\nParameter sensitivity\n", sensitivity.round(3))
