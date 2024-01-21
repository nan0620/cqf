import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sobol_seq import i4_sobol_generate

# Parameter settings
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1.0   # Time to maturity
r = 0.05  # Risk-free rate
sigma_actual = 0.3  # Actual volatility
mu = r              # Expected return
num_paths = 1000    # Number of paths
num_steps = 252     # Number of time steps
dt = T / num_steps  # Time step size

# Generate Sobol sequence
sobol_seq = i4_sobol_generate(1, num_paths * num_steps).flatten()

# Monte Carlo simulation
PnL_traditional = np.zeros(num_paths)
PnL_small_delta = np.zeros(num_paths)

for i in range(num_paths):
    S = np.zeros(num_steps + 1)
    S[0] = S0

    # Generate paths
    for j in range(1, num_steps + 1):
        Z = stats.norm.ppf(sobol_seq[i * num_steps + j - 1])
        S[j] = S[j - 1] * (1 + mu * dt + sigma_actual * np.sqrt(dt) * Z)

    # Calculate option Delta hedging
    d1 = (np.log(S[:-1] / K) + (r + 0.5 * sigma_actual ** 2) * (T - np.arange(num_steps) * dt)) / (sigma_actual * np.sqrt(T - np.arange(num_steps) * dt))
    delta_traditional = stats.norm.cdf(d1)
    delta_small = delta_traditional * 0.9  # Using a smaller delta for hedging

    # Calculate P&L
    option_payoff = np.maximum(S[-1] - K, 0)
    PnL_traditional[i] = option_payoff - np.cumsum(delta_traditional * (S[1:] - S[:-1]))[-1]
    PnL_small_delta[i] = option_payoff - np.cumsum(delta_small * (S[1:] - S[:-1]))[-1]

# Analyze results
mean_PnL_traditional = np.mean(PnL_traditional)
std_PnL_traditional = np.std(PnL_traditional)
mean_PnL_small_delta = np.mean(PnL_small_delta)
std_PnL_small_delta = np.std(PnL_small_delta)

# Plotting the results
plt.figure(figsize=(14, 7))

# Histogram of traditional Delta hedging P&L
plt.subplot(1, 2, 1)
plt.hist(PnL_traditional, bins=50, alpha=0.7, label='Traditional Delta Hedging')
plt.hist(PnL_small_delta, bins=50, alpha=0.7, label='Small Delta Hedging')
plt.xlabel('P&L')
plt.ylabel('Frequency')
plt.title('Histogram of P&L')
plt.legend()

# P&L paths over time for a few paths
plt.subplot(1, 2, 2)
sample_paths = np.random.choice(num_paths, size=10, replace=False)
for i in sample_paths:
    plt.plot(np.cumsum(delta_traditional * (S[1:] - S[:-1])), label=f'Path {i+1} Traditional')
    plt.plot(np.cumsum(delta_small * (S[1:] - S[:-1])), label=f'Path {i+1} Small Delta', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative P&L')
plt.title('P&L Paths Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Print the results
print(f"Mean P&L for traditional volatility hedging: {mean_PnL_traditional}")
print(f"Standard deviation of P&L for traditional volatility hedging: {std_PnL_traditional}")
print(f"Mean P&L for small delta hedging: {mean_PnL_small_delta}")
print(f"Standard deviation of P&L for small delta hedging: {std_PnL_small_delta}")

