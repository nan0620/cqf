import numpy as np
import scipy.stats as stats
from sobol_seq import i4_sobol_generate

# 参数设置
S0 = 100  # 初始股价
K = 100  # 行权价
T = 1.0  # 到期时间
r = 0.05  # 无风险利率
sigma_actual = 0.3  # 实际波动率
mu = r  # 预期收益率
num_paths = 1000  # 路径数量
num_steps = 252  # 时间步数量

# 蒙特卡洛模拟
PnL = np.zeros(num_paths)
Gamma_t_values = []  # 用于存储各个路径上的Gamma值

dt = T / num_steps

# 生成Sobol序列
sobol_seq = i4_sobol_generate(1, num_paths * num_steps)
sobol_seq = sobol_seq.reshape(num_paths, num_steps)

for i in range(num_paths):
    S = np.zeros(num_steps + 1)
    S[0] = S0
    gamma_path = np.zeros(num_steps)  # 存储每个时间步的Gamma值

    # 生成路径
    for j in range(1, num_steps + 1):
        Z = stats.norm.ppf(sobol_seq[i, j - 1])
        S[j] = S[j - 1] + r * S[j - 1] * dt + sigma_actual * S[j - 1] * np.sqrt(dt) * Z \
               + 0.5 * sigma_actual ** 2 * S[j - 1] * (Z ** 2 - 1) * dt

    S = np.maximum(S, 1e-10)

    # 计算期权价格和Delta、Gamma对冲
    d1 = (np.log(S / K) + (r + 0.5 * sigma_actual ** 2) * (T - np.arange(num_steps + 1) * dt)) / (
            sigma_actual * np.sqrt(T - np.arange(num_steps + 1) * dt))
    d2 = d1 - sigma_actual * np.sqrt(T - np.arange(num_steps + 1) * dt)
    C = S * stats.norm.cdf(d1) - K * np.exp(-r * (T - np.arange(num_steps + 1) * dt)) * stats.norm.cdf(d2)
    delta = stats.norm.cdf(d1)
    gamma = stats.norm.pdf(d1) / (S * sigma_actual * np.sqrt(T - np.arange(num_steps + 1) * dt))

    # 计算P&L和记录Gamma
    portfolio = -C[0]  # 初始投资组合：-C表示购买期权
    for j in range(num_steps):
        portfolio += delta[j] * (S[j + 1] - S[j])  # Delta对冲
        gamma_path[j] = gamma[j]
    PnL[i] = portfolio + C[-1]  # 最终组合价值加上期权到期价值
    Gamma_t_values.append(gamma_path)

# 分析结果
mean_PnL = np.mean(PnL)
std_PnL = np.std(PnL)
mean_Gamma_t = np.mean(Gamma_t_values, axis=0)  # 计算每个时间步的平均Gamma值

print(f"Mean P&L: {mean_PnL}")
print(f"Standard deviation of P&L: {std_PnL}")

# 绘图展示Gamma随时间的变化
import matplotlib.pyplot as plt
plt.plot(mean_Gamma_t)
plt.xlabel('Time Steps')
plt.ylabel('Mean Gamma')
plt.title('Mean Gamma Over Time')
plt.show()
