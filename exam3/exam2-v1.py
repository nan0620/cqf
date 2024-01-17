import numpy as np


def simulate_path(s0, mu, sigma, horizon, timesteps, n_sims):
    # set the random seed for reproducibility
    np.random.seed(2023)

    # read parameters
    S0 = s0  # initial spot price
    r = mu  # mu = rf in risk neutral framework
    T = horizon  # time horizon
    t = timesteps  # number of time steps
    n = n_sims  # number of simulations

    # define dt
    dt = T / t  # length of time interval

    # simulate 'n' asset price paths with 't' time steps
    S = np.zeros((t, n))
    S[0] = S0

    for i in range(0, t - 1):
        w = np.random.standard_normal(n)
        S[i + 1] = S[i] * (1 + r * dt + sigma * np.sqrt(dt) * w)

    return S


def asian_option_price(S, K, r, T):
    # calculate the average price of each path
    average_price = np.mean(S, axis=0)

    # calculate the payoff of the Asian option
    payoff = np.maximum(average_price - K, 0)

    # calculate the discounted expected payoff
    discounted_payoff = np.exp(-r * T) * payoff

    # calculate the option price
    option_price = np.mean(discounted_payoff)

    return option_price


def lookback_option_price(S, r, T):
    # calculate the maximum price of each path
    max_price = np.max(S, axis=0)

    # calculate the payoff of the lookback option
    payoff = max_price - S[0]

    # calculate the discounted expected payoff
    discounted_payoff = np.exp(-r * T) * payoff

    # calculate the option price
    option_price = np.mean(discounted_payoff)

    return option_price


# Example usage
S = simulate_path(100, 0.05, 0.2, 1, 252, 10000)
K = 100  # strike price
r = 0.05  # risk-free interest rate
T = 1  # time to expiration

asian_option = asian_option_price(S, K, r, T)
lookback_option = lookback_option_price(S, r, T)

print("Asian Option Price:", asian_option)
print("Lookback Option Price:", lookback_option)
