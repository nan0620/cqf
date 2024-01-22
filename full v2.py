from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, stats
from scipy.stats.qmc import Sobol
from matplotlib import pyplot as plt
from sobol_seq import i4_sobol_generate
from tqdm.contrib.concurrent import process_map

# 步骤 1: 整理数据并计算必要的变量
df = pd.read_csv('processed_option_data.csv', low_memory=False)

# 步骤 2: 使用新模型进行模拟
# Set parameters
params = {
    'S0': df['UNDERLYING_LAST'].values,  # Initial underlying asset price
    'K': df['STRIKE'].values,  # Option strike price
    'T': df['DTE'].values / 252.0,  # Time expiry (in years, assuming 252 trading days in a year)
    'r': 0.05,  # Risk-free rate
    'sigma_actual': df['C_IV'].values,  # Actual volatility
    'mu': 0.05,  # Expected rate of return, assumed to be equal to the risk-free rate
    'num_paths': len(df),  # options in the dataset
    'num_steps': 252  # time steps simulated
}
# Pre-calculate constants
dt = params['T'] / params['num_steps']
sobol_seq = i4_sobol_generate(1, params['num_paths'] * params['num_steps']).reshape(params['num_paths'], params['num_steps'])

def simulate_wrapper(path_id):
    return simulate_path(path_id, params, sobol_seq)

def simulate_path(path_id, params, sobol_seq):
    # Initialize arrays
    S = np.zeros(params['num_steps'] + 1)
    S[0] = params['S0'][path_id]

    # Simulate the path
    for j in range(1, params['num_steps'] + 1):
        Z = stats.norm.ppf(sobol_seq[path_id, j - 1])
        S[j] = S[j - 1] + (params['r'] * S[j - 1] * dt[path_id] + params['sigma_actual'][path_id] * S[j - 1] * np.sqrt(dt[path_id]) * Z + 0.5 * params['sigma_actual'][path_id] ** 2 * S[j - 1] * (
                    Z ** 2 - 1) * dt[path_id])

    # Ensure that S does not contain zero or negative values
    S = np.maximum(S, 1e-10)

    # Calculate option prices and hedging
    C = np.maximum(S - params['K'][path_id], 0)

    # Prevent division by zero or negative numbers
    time_to_maturity = np.maximum(params['T'][path_id] - np.arange(params['num_steps'] + 1) * dt[path_id], 1e-10)

    # Real volatility hedging
    d1_actual = (np.log(S / params['K'][path_id]) + (params['r'] + 0.5 * params['sigma_actual'][path_id] ** 2) * time_to_maturity) / (params['sigma_actual'][path_id] * np.sqrt(time_to_maturity))
    delta_actual = stats.norm.cdf(d1_actual)  # Black-Scholes Delta

    # Calculate P&L
    # Initial portfolio: -C indicates purchase of options
    portfolio_actual = -C[0]
    portfolio_iv = -C[0]
    for j in range(params['num_steps']):
        # Real volatility hedging
        portfolio_actual += delta_actual[j] * (S[j + 1] - S[j])
        # Use Delta hedging in my data
        portfolio_iv += df['C_DELTA'].values[path_id] * (S[j + 1] - S[j])
    # Final portfolio value plus option value at expiry
    PnL_actual = portfolio_actual + C[-1]
    PnL_iv = portfolio_iv + C[-1]

    return PnL_actual, PnL_iv

# 模拟所有路径并计算 P&L
tasks = range(params['num_paths'])
# Multi-processing
num_processes = cpu_count()
with Pool(processes=num_processes) as pool:
    results = process_map(simulate_wrapper, tasks, max_workers=num_processes, desc='Simulating Paths', chunksize=1)
# Unpack results
PnL_actual, PnL_iv = zip(*results)

df['PnL_actual'] = PnL_actual
df['PnL_iv'] = PnL_iv

# 步骤 3: 参数的滚动估计和校准
# 步骤 3：使用 Hull-White 模型进行滚动估算参数 a, b, c
# 对于每个行权期，我们需要根据历史数据滚动估算 Hull-White 模型的参数。由于这是一个复杂的过程，我们需要首先确定参数估计的目标函数，然后使用 SLSQP（Sequential Least Squares Programming）优化器进行参数估计。
from scipy.optimize import minimize
# 假设 historical_data 是一个包含历史利率数据的 DataFrame
# 以及每个行权期的相关数据

def estimate_hull_white_parameters(historical_data):
    # 定义目标函数
    def objective(params):
        a, b, c = params
        # 使用 a, b, c 计算 Hull-White 模型的预测值
        # ...
        # 计算预测误差
        prediction_error = ...
        return np.sum(prediction_error ** 2)

    # 参数估计
    result = minimize(objective, x0=[0.1, 0.1, 0.1], method='SLSQP')
    return result.x

# 滚动估算参数
estimated_params = []
for expiration_date in historical_data['expiration_dates'].unique():
    data_subset = historical_data[historical_data['expiration_date'] == expiration_date]
    params = estimate_hull_white_parameters(data_subset)
    estimated_params.append(params)

# 将参数保存到 DataFrame
params_df = pd.DataFrame(estimated_params, columns=['a', 'b', 'c'])

# 步骤 4: 回归分析拟合 Delta 和 Gamma
import statsmodels.api as sm

# 假设 option_data 是一个包含期权价格、股票价格和其他相关变量的 DataFrame
# Delta 和 Gamma 是您希望拟合的目标变量

# 拟合 Delta
X = option_data[['stock_price', 'other_variables']]  # 自变量
X = sm.add_constant(X)  # 添加常数项
y_delta = option_data['delta']  # 因变量
model_delta = sm.OLS(y_delta, X).fit()

# 拟合 Gamma
y_gamma = option_data['gamma']  # 因变量
model_gamma = sm.OLS(y_gamma, X).fit()

# 输出回归结果
print(model_delta.summary())
print(model_gamma.summary())



# 步骤 5: 模型验证和绘图
# 验证模型的一个方法是观察参数 a, b, c 随时间的变化情况，以及绘制相关的图像来分析模型的表现。
import matplotlib.pyplot as plt

# 参数随时间的变化情况
plt.plot(params_df['a'], label='a')
plt.plot(params_df['b'], label='b')
plt.plot(params_df['c'], label='c')
plt.legend()
plt.title('Hull-White Parameters Over Time')
plt.show()

# 绘制 δMV−δBS 的图像
# 假设我们有模型计算的 Delta (δMV) 和 Black-Scholes 计算的 Delta (δBS)
delta_difference = option_data['delta_model'] - option_data['delta_BS']
plt.plot(delta_difference)
plt.title('Difference between Model Delta and Black-Scholes Delta')
plt.show()

# 绘制 IV 预期变化和 Delta 的对比图
# 假设我们有隐含波动率 (IV) 和 Delta 的数据
plt.plot(option_data['IV'], label='Implied Volatility')
plt.plot(option_data['delta'], label='Delta')
plt.legend()
plt.title('Comparison of Expected Changes in IV and Delta')
plt.show()



# 绘制对冲盈利图
plt.scatter(data['C_DELTA'], data['PnL_actual'], label='Actual Hedging P&L', alpha=0.5)
plt.scatter(data['C_DELTA'], data['PnL_iv'], label='IV Hedging P&L', alpha=0.5)
plt.axhline(y=0.15, color='r', linestyle='--', label='15% Hedging Gain')
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('Hedging P&L')
plt.title('Hedging P&L vs Delta')
plt.legend()
plt.show()

# 步骤 6:计算套期保值收益率
# 套期保值收益率可以通过比较保值策略的终值与未保值策略的终值来计算。这通常涉及到对股票和期权头寸的动态调整。
# 假设我们有股票和期权的时间序列数据，以及相应的保值比率
# hedge_ratio 是 Delta，表示每持有一份期权需要持有的股票数量

hedge_ratio = option_data['delta']
stock_prices = option_data['stock_price']
option_prices = option_data['option_price']

# 初始化资产和现金头寸
cash_position = 0
asset_position = 0

# 逐步调整头寸并计算收益率
for i in range(1, len(stock_prices)):
    # 调整股票头寸
    stock_position_change = hedge_ratio[i] - hedge_ratio[i - 1]
    asset_position += stock_position_change * stock_prices[i]

    # 更新现金头寸
    cash_position -= stock_position_change * stock_prices[i]

    # 计算期权价值变化
    option_value_change = option_prices[i] - option_prices[i - 1]

    # 更新现金头寸
    cash_position += option_value_change

# 计算最终的套期保值收益率
final_value = asset_position * stock_prices[-1] + cash_position
hedge_return = final_value / np.abs(option_prices[0]) - 1

print(f'Hedging Return: {hedge_return:.2%}')

# 查找满足15%对冲盈利的Delta桶和到期日
profitable_trades = data[data['PnL_actual'] >= 0.15]
print("满足15%对冲盈利的交易:")
print(profitable_trades[['EXPIRE_DATE', 'C_DELTA', 'PnL_actual']])