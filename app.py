import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import pandas as pd
from scipy.stats import norm

class BlackScholes:
    def __init__(self, underlying_price, strike_price, time_to_maturity, risk_free_interest_rate, volatility):
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_interest_rate = risk_free_interest_rate
        self.volatility = volatility

    def calculate_d1(self):
        if self.time_to_maturity == 0 or self.volatility == 0:
            raise ValueError("Time to maturity and volatility must be greater than zero.")
        d1 = (math.log(self.underlying_price / self.strike_price) +
              (self.risk_free_interest_rate + (math.pow(self.volatility, 2)) / 2) * self.time_to_maturity) / \
             (self.volatility * math.sqrt(self.time_to_maturity))
        return d1

    def calculate_d2(self):
        d1 = self.calculate_d1()
        d2 = d1 - (self.volatility * math.sqrt(self.time_to_maturity))
        return d2

    def calculate_call_option(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        return (self.underlying_price * norm.cdf(d1)) - \
               (self.strike_price * math.exp(-self.risk_free_interest_rate * self.time_to_maturity) * norm.cdf(d2))

    def calculate_put_option(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        return (self.strike_price * math.exp(-self.risk_free_interest_rate * self.time_to_maturity) * norm.cdf(-d2)) - \
               (self.underlying_price * norm.cdf(-d1))


st.title("Black Scholes Options HeatMap")

S = st.number_input("Underlying Price (S)", min_value=0.0, value=100.0, step=1.0)
K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
T = st.number_input("Time to Maturity (T) in years", min_value=0.0, value=1.0, step=0.1)
r = st.number_input("Risk-Free Interest Rate (r) in decimal", min_value=0.0, value=0.05, step=0.01)
sigma = st.number_input("Volatility (σ) in decimal", min_value=0.0, value=0.2, step=0.01)

call_price = BlackScholes(S, K, T, r, sigma).calculate_call_option()
put_price = BlackScholes(S, K, T, r, sigma).calculate_put_option()

st.write(f"Call Option Price: {call_price:.2f}")
st.write(f"Put Option Price: {put_price:.2f}")

S_min = 0.8 * S
S_max = 1.2 * S
sigma_min = 0.5 * sigma
sigma_max = 1.5 * sigma

S_range = np.linspace(S_min, S_max, 10)
sigma_range = np.linspace(sigma_min, sigma_max, 10)

S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)

call_prices = np.zeros_like(S_grid)
put_prices = np.zeros_like(S_grid)

for i in range(S_grid.shape[0]):
    for j in range(S_grid.shape[1]):
        call_prices[i, j] = BlackScholes(
            S_grid[i, j], K, T, r, sigma_grid[i, j]).calculate_call_option()
        put_prices[i, j] = BlackScholes(
            S_grid[i, j], K, T, r, sigma_grid[i, j]).calculate_put_option()

sns.set(style="whitegrid")

fmt = ".2f"
x_labels = [f"{x:.2f}" for x in S_range]
y_labels = [f"{y:.2f}" for y in sigma_range]

fig1, ax1 = plt.subplots(figsize=(8,6))
sns.heatmap(call_prices, annot=True, fmt=fmt,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="viridis", cbar_kws={'label': 'Call Price'}, ax=ax1)
ax1.set_title('Call Prices Heatmap')
ax1.set_xlabel('Stock Price (S)')
ax1.set_ylabel('Volatility (σ)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

st.write("### Call Option Prices Heatmap")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(put_prices, annot=True, fmt=fmt,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap="viridis", cbar_kws={'label': 'Put Price'}, ax=ax2)
ax2.set_title('Put Prices Heatmap')
ax2.set_xlabel('Stock Price (S)')
ax2.set_ylabel('Volatility (σ)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

st.write("### Put Option Prices Heatmap")
st.pyplot(fig2)

st.write("---")