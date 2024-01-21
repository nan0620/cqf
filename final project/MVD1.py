import numpy as np
import scipy.stats as stats
from sobol_seq import i4_sobol_generate
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 读取SPX期权数据
file_path = '/Users/nan/Desktop/Final Project/cqf/spx_eod_202310.csv'
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
df.columns = df.columns.str.strip().str.replace("'", '', regex=False).str.replace('[', '', regex=False).str.replace(']', '', regex=False)

# 设置参数
S0 = df['UNDERLYING_LAST'].values  # 初始标的资产价格
K = df['STRIKE'].values  # 期权行权价
T = df['DTE'].values / 252.0  # 时间到期（年份为单位，假设一年有252个交易日）
r = 0.05  # 无风险利率
sigma_actual = df['C_IV'].values / 100.0  # 实际波动率，将百分比转化为小数
mu = r  # 预期回报率，假设与无风险利率相等
num_paths = len(df)  # 数据集中的期权数量
num_steps = 252  # 模拟的时间步数

# 初始化结果数组
PnL_actual = np.zeros(num_paths)
PnL_iv = np.zeros(num_paths)
dt = T / num_steps

# 生成 Sobol 序列
sobol_seq = i4_sobol_generate(1, num_paths * num_steps)
sobol_seq = sobol_seq.reshape(num_paths, num_steps)

with tqdm(total=num_paths, desc='Simulating Paths', unit='path') as pbar:
    for i in range(num_paths):
        S = np.zeros(num_steps + 1)
        S[0] = S0[i]

        # 生成路径
        for j in range(1, num_steps + 1):
            Z = stats.norm.ppf(sobol_seq[i, j - 1])
            # Milstein scheme
            S[j] = S[j - 1] + r * S[j - 1] * dt[i] + sigma_actual[i] * S[j - 1] * np.sqrt(dt[i]) * Z \
                   + 0.5 * sigma_actual[i] ** 2 * S[j - 1] * (Z ** 2 - 1) * dt[i]

        # 确保 S 不包含零或负值
        S = np.maximum(S, 1e-10)

        # 计算期权价格和 Delta 对冲
        C = np.maximum(S - K[i], 0)  # 欧式看涨期权的内在价值

        # 防止除以零或负数
        time_to_maturity = np.maximum(T[i] - np.arange(num_steps + 1) * dt[i], 1e-10)

        # 实际波动率对冲
        d1_actual = (np.log(S / K[i]) + (r + 0.5 * sigma_actual[i] ** 2) * time_to_maturity) / (sigma_actual[i] * np.sqrt(time_to_maturity))
        delta_actual = stats.norm.cdf(d1_actual)  # Black-Scholes Delta

        # 计算 P&L
        portfolio_actual = -C[0]  # 初始投资组合：-C 表示购买期权
        portfolio_iv = -C[0]
        for j in range(num_steps):
            portfolio_actual += delta_actual[j] * (S[j + 1] - S[j])  # 实际波动率对冲
            portfolio_iv += df['C_DELTA'].values[i] * (S[j + 1] - S[j])  # 使用您的数据中的 Delta 对冲
        PnL_actual[i] = portfolio_actual + C[-1]  # 最终投资组合价值加上期权到期时的价值
        PnL_iv[i] = portfolio_iv + C[-1]

        # 更新进度条
        pbar.update(1)
# 完成模拟后关闭进度条
pbar.close()
print('已完成模拟，正在分析结果...\n')

# 分析结果
mean_PnL_actual = np.mean(PnL_actual)
std_PnL_actual = np.std(PnL_actual)
mean_PnL_iv = np.mean(PnL_iv)
std_PnL_iv = np.std(PnL_iv)

print(f"Mean P&L for actual volatility hedging: {mean_PnL_actual}")
print(f"Standard deviation of P&L for actual volatility hedging: {std_PnL_actual}")
print(f"Mean P&L for implied volatility hedging: {mean_PnL_iv}")
print(f"Standard deviation of P&L for implied volatility hedging: {std_PnL_iv}")

# 可视化实际波动率对冲的P&L分布
plt.figure(figsize=(10, 6))
plt.hist(PnL_actual, bins=50, color='blue', alpha=0.5, label='Actual Volatility Hedging')
plt.xlabel('P&L')
plt.ylabel('Frequency')
plt.title('P&L Distribution - Actual Volatility Hedging')
plt.legend()
plt.grid(True)
plt.show()

# 可视化使用您的数据中的Delta对冲的P&L分布
plt.figure(figsize=(10, 6))
plt.hist(PnL_iv, bins=50, color='red', alpha=0.5, label='Using Your Data for Delta Hedging')
plt.xlabel('P&L')
plt.ylabel('Frequency')
plt.title('P&L Distribution - Using Your Data for Delta Hedging')
plt.legend()
plt.grid(True)
plt.show()
