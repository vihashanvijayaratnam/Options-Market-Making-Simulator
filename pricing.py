"""Black-Scholes pricing and Greek calculations for European options."""

import math
from scipy.stats import norm


def calculate_call_intrinsic_value(share_price, strike_price):
    return max(share_price - strike_price, 0.0)


def calculate_put_intrinsic_value(share_price, strike_price):
    return max(strike_price - share_price, 0.0)


def calculate_d1(share_price, strike_price, time_to_expiry, interest_rate, volatility):
    if share_price <= 0 or strike_price <= 0:
        raise ValueError("Share price and strike price must be positive.")
    if time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("Time to expiry and volatility must be positive.")
    numerator = math.log(share_price / strike_price) + (
        interest_rate + 0.5 * volatility**2
    ) * time_to_expiry
    return numerator / (volatility * math.sqrt(time_to_expiry))


def calculate_d2(d1, volatility, time_to_expiry):
    return d1 - volatility * math.sqrt(time_to_expiry)


def calculate_call_price(share_price, strike_price, interest_rate, time_to_expiry, d1, d2):
    return share_price * norm.cdf(d1) - strike_price * math.exp(
        -interest_rate * time_to_expiry
    ) * norm.cdf(d2)


def calculate_put_price(share_price, strike_price, interest_rate, time_to_expiry, d1, d2):
    return strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(
        -d2
    ) - share_price * norm.cdf(-d1)


def black_scholes_prices(share_price, strike_price, interest_rate, time_to_expiry, volatility):
    """Return call and put values, including intrinsic value at expiry."""
    if time_to_expiry <= 0:
        return (
            calculate_call_intrinsic_value(share_price, strike_price),
            calculate_put_intrinsic_value(share_price, strike_price),
        )
    d1 = calculate_d1(share_price, strike_price, time_to_expiry, interest_rate, volatility)
    d2 = calculate_d2(d1, volatility, time_to_expiry)
    return (
        calculate_call_price(share_price, strike_price, interest_rate, time_to_expiry, d1, d2),
        calculate_put_price(share_price, strike_price, interest_rate, time_to_expiry, d1, d2),
    )


def calculate_greeks(share_price, strike_price, interest_rate, time_to_expiry, volatility):
    """Return call/put delta, gamma, vega and daily theta.

    Vega is per one percentage-point volatility change. At expiry, the smooth
    Black-Scholes Greeks are undefined, so NaN is returned.
    """
    if time_to_expiry <= 0:
        return {key: math.nan for key in (
            "call_delta", "put_delta", "gamma", "vega", "call_theta", "put_theta"
        )}
    d1 = calculate_d1(share_price, strike_price, time_to_expiry, interest_rate, volatility)
    d2 = calculate_d2(d1, volatility, time_to_expiry)
    gamma = norm.pdf(d1) / (share_price * volatility * math.sqrt(time_to_expiry))
    vega = share_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
    common = -(share_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
    discounted_strike = interest_rate * strike_price * math.exp(-interest_rate * time_to_expiry)
    return {
        "call_delta": norm.cdf(d1),
        "put_delta": norm.cdf(d1) - 1,
        "gamma": gamma,
        "vega": vega,
        "call_theta": (common - discounted_strike * norm.cdf(d2)) / 365,
        "put_theta": (common + discounted_strike * norm.cdf(-d2)) / 365,
    }


# Backwards-compatible helpers retained from the original learning script.
def calc_call_delta(d1):
    return norm.cdf(d1)


def calc_put_delta(d1):
    return norm.cdf(d1) - 1


def calc_gamma(d1, share_price, volatility, time_to_expiry):
    return norm.pdf(d1) / (share_price * volatility * math.sqrt(time_to_expiry))


def calc_vega(share_price, d1, time_to_expiry):
    return share_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100


def calc_call_theta(share_price, d1, volatility, time_to_expiry, interest_rate, strike_price, d2):
    common = -(share_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
    cash = interest_rate * strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
    return (common - cash) / 365


def calc_put_theta(share_price, d1, volatility, time_to_expiry, interest_rate, strike_price, d2):
    common = -(share_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
    cash = interest_rate * strike_price * math.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2)
    return (common + cash) / 365
