import numpy as np
import scipy.stats as stats
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

# 蒙特卡洛模拟
PnL_actual = np.zeros(num_paths)
PnL_iv = np.zeros(num_paths)
dt = T / num_steps

# 生成Sobol序列
sobol_seq = i4_sobol_generate(1, num_paths * num_steps)
sobol_seq = sobol_seq.reshape(num_paths, num_steps)

for i in range(num_paths):
    S = np.zeros(num_steps + 1)
    S[0] = S0

    # 生成路径
    for j in range(1, num_steps + 1):
        Z = stats.norm.ppf(sobol_seq[i, j - 1])
        # Milstein
        S[j] = S[j - 1] + r * S[j - 1] * dt + sigma_actual * S[j - 1] * np.sqrt(dt) * Z \
               + 0.5 * sigma_actual ** 2 * S[j - 1] * (Z ** 2 - 1) * dt

    # 确保 S 不包含零或负值
    S = np.maximum(S, 1e-10)

    # 计算期权价格和Delta对冲
    C = np.maximum(S - K, 0)  # 欧式看涨期权内在价值

    # 防止除以零或负数
    time_to_maturity = np.maximum(T - np.arange(num_steps + 1) * dt, 1e-10)

    # 实际波动率对冲
    d1_actual = (np.log(S / K) + (r + 0.5 * sigma_actual ** 2) * time_to_maturity) / (
            sigma_actual * np.sqrt(time_to_maturity))
    delta_actual = stats.norm.cdf(d1_actual)  # Black-Scholes Delta

    # 隐含波动率对冲
    d1_iv = (np.log(S / K) + (r + 0.5 * sigma_iv ** 2) * time_to_maturity) / (
            sigma_iv * np.sqrt(time_to_maturity))
    delta_iv = stats.norm.cdf(d1_iv)  # Black-Scholes Delta

    # 计算P&L
    portfolio_actual = -C[0]  # 初始投资组合：-C表示购买期权
    portfolio_iv = -C[0]
    for j in range(num_steps):
        portfolio_actual += delta_actual[j] * (S[j + 1] - S[j])  # 实际波动率对冲
        portfolio_iv += delta_iv[j] * (S[j + 1] - S[j])  # 隐含波动率对冲
    PnL_actual[i] = portfolio_actual + C[-1]  # 最终组合价值加上期权到期价值
    PnL_iv[i] = portfolio_iv + C[-1]

# 分析结果
mean_PnL_actual = np.mean(PnL_actual)
std_PnL_actual = np.std(PnL_actual)
mean_PnL_iv = np.mean(PnL_iv)
std_PnL_iv = np.std(PnL_iv)

print(f"Mean P&L for actual volatility hedging: {mean_PnL_actual}")
print(f"Standard deviation of P&L for actual volatility hedging: {std_PnL_actual}")
print(f"Mean P&L for implied volatility hedging: {mean_PnL_iv}")
print(f"Standard deviation of P&L for implied volatility hedging: {std_PnL_iv}")
