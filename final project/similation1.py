import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from sobol_seq import i4_sobol_generate

# 参数设置
S0 = 100  # 初始股价
K = 100  # 行权价
T = 1.0  # 到期时间
r = 0.05  # 无风险利率
sigma_actual = 0.3  # 实际波动率
sigma_iv = 0.2  # 隐含波动率
mu = r  # 预期收益率
num_paths = 1000  # 路径数量
num_steps = 252  # 时间步数量

# 生成Sobol序列
sobol_seq = i4_sobol_generate(1, num_paths * num_steps)
sobol_seq = sobol_seq.reshape(num_paths, num_steps)

# 蒙特卡洛模拟
PnL_actual = np.zeros(num_paths)
PnL_iv = np.zeros(num_paths)
dt = T / num_steps

for i in range(num_paths):
    S = np.zeros(num_steps + 1)
    S[0] = S0

    # 生成路径
    for j in range(1, num_steps + 1):
        Z = norm.ppf(sobol_seq[i, j - 1])
        S[j] = S[j - 1] * np.exp((mu - 0.5 * sigma_actual ** 2) * dt + sigma_actual * np.sqrt(dt) * Z)

    # 对冲策略实施
    delta = np.zeros(num_steps + 1)
    for j in range(num_steps):
        d1 = (np.log(S[j] / K) + (r + 0.5 * sigma_iv ** 2) * (T - j * dt)) / (sigma_iv * np.sqrt(T - j * dt))
        delta[j] = norm.cdf(d1)
    delta[-1] = 1 if S[-1] > K else 0  # 到期时Delta

    # 计算P&L
    PnL_actual[i] = max(S[-1] - K, 0) - np.sum(delta[:-1] * (S[1:] - S[:-1]))
    PnL_iv[i] = max(S[-1] - K, 0) - np.sum(delta[:-1] * (S[1:] - S[:-1]))  # 假设隐含波动率已知

# 分析结果
mean_PnL_actual = np.mean(PnL_actual)
mean_PnL_iv = np.mean(PnL_iv)
std_PnL_actual = np.std(PnL_actual)
std_PnL_iv = np.std(PnL_iv)

print(f"Mean P&L for actual volatility hedging: {mean_PnL_actual}")
print(f"Mean P&L for implied volatility hedging: {mean_PnL_iv}")
print(f"Standard deviation of P&L for actual volatility hedging: {std_PnL_actual}")
print(f"Standard deviation of P&L for implied volatility hedging: {std_PnL_iv}")
