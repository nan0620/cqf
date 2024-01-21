import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from sobol_seq import i4_sobol_generate

# 参数设置
S0 = 100  # 初始股价
K = 100  # 行权价
T = 1.0  # 到期时间
r = 0.05  # 无风险利率
sigma_actual = 0.3  # 实际波动率
sigma_imp = 0.2    # 隐含波动率
mu = r             # 预期收益率
num_paths = 1000   # 路径数量
num_steps = 252    # 时间步数量
dt = T / num_steps # 时间步长度

# 生成Sobol序列
sobol_seq = i4_sobol_generate(1, num_paths * num_steps).flatten()

# 蒙特卡洛模拟
PnL = np.zeros(num_paths)
time_steps = np.linspace(0, T, num_steps + 1)


for i in range(num_paths):
    S = np.zeros(num_steps + 1)
    S[0] = S0
    r_squared_term = r**2 * dt
    sigma_imp_term = sigma_imp * dt

    # 生成路径
    for j in range(1, num_steps + 1):
        Z = stats.norm.ppf(sobol_seq[i * num_steps + j - 1])
        S[j] = S[j - 1] * (1 + mu * dt + sigma_actual * np.sqrt(dt) * Z)

    # 计算期权Delta对冲
    d1 = (np.log(S[:-1] / K) + (r + 0.5 * sigma_imp**2) * (T - np.arange(num_steps) * dt)) / (sigma_imp * np.sqrt(T - np.arange(num_steps) * dt))
    delta = stats.norm.cdf(d1)

    # 计算对冲成本
    cost_of_hedging = np.cumsum(delta * (S[1:] - S[:-1]))

    # 计算P&L
    final_option_value = max(S[-1] - K, 0)
    PnL[i] = final_option_value - cost_of_hedging[-1] - (r_squared_term - sigma_imp_term)

# 分析结果
mean_PnL = np.mean(PnL)
std_PnL = np.std(PnL)

print(f"Mean P&L: {mean_PnL}")
print(f"Standard deviation of P&L: {std_PnL}")

# 输出图像展示
plt.figure(figsize=(12, 6))

# P&L的直方图
plt.subplot(1, 2, 1)
plt.hist(PnL, bins=50, color='blue', edgecolor='black')
plt.title('Histogram of P&L')
plt.xlabel('P&L')
plt.ylabel('Frequency')

# P&L随时间变化的路径
plt.subplot(1, 2, 2)
for i in range(min(10, num_paths)):  # 只绘制部分路径
    plt.plot(time_steps, np.cumsum(np.insert(delta * (S[1:] - S[:-1]), 0, 0)), label=f'Path {i+1}')
plt.title('P&L Paths Over Time')
plt.xlabel('Time (Years)')
plt.ylabel('Cumulative P&L')
plt.legend()

plt.tight_layout()
plt.show()

