import math

from pricing import black_scholes_prices
from simulation import SimulationConfig, run_single_simulation


def test_at_the_money_black_scholes_values():
    call, put = black_scholes_prices(100, 100, 0.05, 0.5, 0.20)
    assert math.isclose(call, 6.8887, rel_tol=1e-4)
    assert math.isclose(put, 4.4197, rel_tol=1e-4)


def test_expiry_values_are_intrinsic_values():
    call, put = black_scholes_prices(106, 100, 0.05, 0, 0.20)
    assert call == 6
    assert put == 0


def test_history_lengths_and_final_pnl_identity():
    config = SimulationConfig()
    market, histories, _ = run_single_simulation(config, seed=73)
    assert len(market) == config.num_days + 1
    assert all(len(history) == config.num_days + 1 for history in histories.values())
    for history in histories.values():
        expected = history["cash"].iloc[-1] + history["inventory"].iloc[-1] * market["call_fair_value"].iloc[-1]
        assert math.isclose(history["marked_to_market_pnl"].iloc[-1], expected)
