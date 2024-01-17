import numpy as np


def test():
    # Define parameters
    S = 100
    E = 100
    T = 1
    t = 0
    N = 252
    dt = (T - t) / N
    sigma = 0.2
    r = 0.05

    # Generate N random numbers from standard normal distribution
    random_numbers = np.random.standard_normal(N)

    # Initialize array to store simulated stock prices
    stock_prices = np.zeros(N + 1)
    stock_prices[0] = S

    # Simulate stock prices using Euler-Maruyama scheme
    for i in range(N):
        drift = r * stock_prices[i] * dt
        diffusion = sigma * stock_prices[i] * np.sqrt(dt) * random_numbers[i]
        stock_prices[i + 1] = stock_prices[i] + drift + diffusion

    # Calculate payoff for Asian or lookback option
    payoff = np.maximum(stock_prices[-1] - E, 0)

    # Calculate discounted payoff
    discounted_payoff = np.exp(-r * (T - t)) * payoff

    # Calculate expected value of discounted payoff
    expected_payoff = np.mean(discounted_payoff)

    print("Expected value of discounted payoff:", expected_payoff)


def compute_price(S0, E, T, delta, r, num_simulations, num_steps):
    # step size of time
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    # discount factor for value
    discount_factor = np.exp(-r * T)
    # store option prices
    option_prices = []

    for _ in range(num_simulations):
        # start price
        S = S0
        # sum
        S_sum = S
        # minimum stock price
        S_min = S
        # maximum stock price
        S_max = S

        for _ in range(num_steps):
            dW = np.random.normal(0, sqrt_dt)
            # Euler-Maruyama
            S += r * S * dt + delta * S * dW
            S_sum += S
            S_min = min(S_min, S)
            S_max = max(S_max, S)

        S_avg = S_sum / (num_steps + 1)
        payoff = max(S_avg - E, 0)  # Asian option payoff
        option_prices.append(payoff)

    option_price = discount_factor * np.mean(option_prices)
    return option_price


def calculate_asian_option_price(S0, E, T, delta, r, num_simulations, num_steps):
    option_price = compute_price(S0, E, T, delta, r, num_simulations, num_steps)
    discounted_option_price = np.exp(-r * T) * option_price
    return discounted_option_price


def compute_price(S0, T, r, sigma, num_simulations):
    # generate random stock price paths
    S = np.zeros((num_simulations, 2))
    S[:, 0] = S0
    S[:, 1] = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(num_simulations))

    # calculate lookback option payoff
    lookback_payoff = np.max(S, axis=1) - S0

    # calculate option price
    option_price = np.mean(lookback_payoff) * np.exp(-r * T)

    return option_price


def calculate_lookback_option_price(S0, T, r, sigma, num_simulations):
    option_price = compute_price(S0, T, r, sigma, num_simulations)
    return option_price


if __name__ == '__main__':
    S = 100  # 初始股票价格
    r = 0.05  # 无风险利率
    T = 1  # 期权到期时间
    t = 0  # 当前时间
    N = 252  # 时间步数
    dt = (T - t) / N  # 时间间隔
    sigma = 0.2  # 波动率
    E = 100  # 期权执行价格
    num_simulations = 10000  # 模拟次数

    option_price = compute_price(S, E, T, sigma, r)
    print("Asian Option Price:", option_price)

    option_price = calculate_asian_option_price(S, E, T, sigma, r, num_simulations, N)
    print("Asian Option Price:", option_price)
