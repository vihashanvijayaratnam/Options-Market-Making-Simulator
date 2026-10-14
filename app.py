#Interactive Streamlit dashboard for the market-making simulator

from dataclasses import replace

import streamlit as st

from simulation import SimulationConfig, create_figures, run_many_simulations, run_single_simulation


st.set_page_config(page_title="Options Market-Making Simulator", layout="wide")
st.title("Options Market-Making Simulator")
st.caption("Compare fixed-spread, inventory-aware and volatility-aware call quoting.")

with st.sidebar:
    st.header("Model controls")
    seed = st.number_input("Random seed", min_value=0, value=73, step=1)
    spread = st.slider("Total base spread (£)", 0.05, 0.60, 0.20, 0.01)
    inventory_sensitivity = st.slider("Inventory sensitivity", 0.00, 0.10, 0.02, 0.005)
    volatility_sensitivity = st.slider("Volatility sensitivity", 0.0, 4.0, 1.5, 0.1)
    customer_noise = st.slider("Customer valuation noise (£)", 0.05, 1.00, 0.25, 0.05)
    number_of_seeds = st.slider("Repeated simulations", 25, 500, 100, 25)

config = replace(
    SimulationConfig(),
    fixed_spread=spread,
    inventory_sensitivity=inventory_sensitivity,
    volatility_sensitivity=volatility_sensitivity,
    customer_noise=customer_noise,
)
market, histories, metrics = run_single_simulation(config, int(seed))
figures = create_figures(market, histories)

st.subheader("Single-path comparison")
st.dataframe(metrics.round(3), use_container_width=True)
left, right = st.columns(2)
with left:
    st.pyplot(figures["share_price"])
    st.pyplot(figures["inventory"])
with right:
    st.pyplot(figures["option_values"])
    st.pyplot(figures["pnl"])

st.subheader("Repeated-simulation evidence")
with st.spinner("Running controlled simulations..."):
    _, summary = run_many_simulations(config, number_of_seeds)
st.dataframe(summary.round(3), use_container_width=True)
st.info(
    "Educational model only. Results depend on simplified assumptions including "
    "one call unit per order, stylised customers and no hedging, fees or market impact."
)
