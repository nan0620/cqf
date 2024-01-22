import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 100  # Initial asset price
K = 100  # Option strike price
T = 1.0  # Option maturity
r = 0.05  # Risk-free interest rate
sigma_true = 0.2  # Actual volatility
N = 252  # Number of time steps in simulation
M = 100  # Number of Monte Carlo simulations


# Milstein schemes for GBM
def milstein_scheme(S, dt, sigma):
    dW = np.random.normal(0, np.sqrt(dt))
    dS = r * S * dt + sigma * S * dW + 0.5 * sigma ** 2 * S * (dW ** 2 - dt)
    return S + dS


# Calculate option price under actual volatility
def option_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# Monte Carlo simulation of P&L with actual volatility hedging
def monte_carlo_simulation():
    PnL = []
    for _ in range(M):
        S = S0
        dt = T / N
        hedging_portfolio = 0

        for _ in range(N):
            sigma_hat = sigma_true  # Delta hedging with actual volatility
            S = milstein_scheme(S, dt, sigma_hat)
            hedging_portfolio += -delta_hedging(S, sigma_hat, dt)

        option_payoff = max(S - K, 0)
        total_PnL = option_payoff + hedging_portfolio
        PnL.append(total_PnL)

    return np.mean(PnL)


# Calculate Delta hedging position
def delta_hedging(S, sigma, dt):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    hedging_position = -delta
    return hedging_position


# Execute Monte Carlo simulation
simulated_PnL = monte_carlo_simulation()

# Output the result
print(f"Simulated P&L with actual volatility hedging: {simulated_PnL}")

# Plot a sample Monte Carlo path
path_samples = [milstein_scheme(S0, T / N, sigma_true) for _ in range(N)]
plt.plot(path_samples)
plt.title("Monte Carlo Path with Actual Volatility Hedging")
plt.xlabel("Time Steps")
plt.ylabel("Asset Price")
plt.show()
