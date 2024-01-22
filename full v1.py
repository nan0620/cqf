from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# # 步骤 1: 整理数据并计算必要的变量
# # 加载CSV文件
# data = pd.read_csv('/Users/nanjiang/cqf/spx_eod_2021-2023_combined.csv', low_memory=False)
# # 确保日期列是日期类型
# data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])
# # 确保C_LAST是数值类型
# data['C_LAST'] = pd.to_numeric(data['C_LAST'], errors='coerce')
# # 排序数据
# data.sort_values(by='QUOTE_DATE', inplace=True)
#
# # 选择需要用到的列，专注于看涨期权
# columns_of_interest = ['QUOTE_DATE', 'UNDERLYING_LAST', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV', 'C_LAST', 'STRIKE', 'EXPIRE_STRUCTURE']
# data = data[columns_of_interest]
#
# # 我需要用到
# # 1. ∆St，标的资产价格的日变化量
# data['∆St'] = data['UNDERLYING_LAST'].diff()
# # 2. ∆St/St，标的资产价格的日变化率
# data['∆St/St'] = data['UNDERLYING_LAST'].pct_change()
# # 3. ∆Vt，期权价格的日变化量，data['C_LAST'].diff()
# data['∆Vt'] = data['C_LAST'].diff()
# # 4. St，标的资产价格，data['UNDERLYING_LAST']
# # 5. IV，隐含波动率，data['C_IV']
# # 6. Delta，期权价值相对于标的资产价格变化的敏感度，data['C_DELTA']
# # 7. Vega，期权价格对波动率变化的敏感度，data['C_VEGA']
# # 8. δ2，Delta的平方，用于捕捉期权价格变动相对于Delta变动的非线性特征
# data['C_DELTA_2'] = data['C_DELTA'] ** 2
# # 保存处理过的数据为新的CSV文件
# data.to_csv('/Users/nanjiang/cqf/processed_option_data.csv', index=False)

# 步骤 2: 计算依赖变量和运行拟合
data = pd.read_csv('processed_option_data.csv', low_memory=False)
# 首行的∆St和∆Vt将为NaN，需要被剔除
data.dropna(inplace=True)


# 定义模型函数
def model(params, X):
    a, b, c = params
    return a * X['C_IV'] + b * X['C_DELTA'] + c * X['C_VEGA']


# 定义损失函数
def loss_function(params, X, Y):
    predictions = model(params, X)
    return ((predictions - Y) ** 2).mean()  # Mean Squared Error


# 准备自变量X和因变量Y
X = data[['C_IV', 'C_DELTA', 'C_VEGA', '∆Vt']]
Y = data['∆Vt']

# 使用SLSQP进行优化
# 初始参数猜测
initial_guess = np.array([0.1, 0.1, 0.1])

# 运行优化器
result = minimize(loss_function, initial_guess, args=(X, Y), method='SLSQP')

# 输出最优参数
if result.success:
    fitted_params = result.x
    print('最优参数:', fitted_params)
else:
    raise ValueError(result.message)

# 步骤 3: 参数的滚动估计和校准
# 滚动窗口估计：使用Hull White的3M滚动窗口，然后每天移动窗口（3 × 22个观测值）。
# 参数校准：参数a, b, c对于每个到期日都是时间依赖的。
# 设置滚动窗口大小
window_size = 3 * 22  # 3个月，每月约22个交易日


# 定义处理每个窗口的函数
def process_window(start, data, window_size, initial_guess):
    end = start + window_size
    window_data = data.iloc[start:end]
    X_window = window_data[['C_DELTA', 'C_IV', 'C_VEGA']]
    Y_window = window_data['∆Vt']
    result = minimize(loss_function, initial_guess, args=(X_window, Y_window), method='SLSQP')
    if result.success:
        return result.x
    else:
        # 如果优化失败，添加NaN。注意这里是3个NaN，因为我们有3个参数
        return [np.nan, np.nan, np.nan]


# 获取CPU核心数
num_cores = cpu_count()

# 使用process_map来并行处理
results = process_map(process_window, range(len(data) - window_size + 1), [data] * (len(data) - window_size + 1), [window_size] * (len(data) - window_size + 1),
                      [initial_guess] * (len(data) - window_size + 1), max_workers=5, chunksize=1)

# 将结果转换为DataFrame
rolling_params_df = pd.DataFrame(results, columns=['a', 'b', 'c'])

# # 滚动窗口估计
# rolling_params = []
# for start in tqdm(range(len(data) - window_size + 1)):
#     end = start + window_size
#     window_data = data.iloc[start:end]
#     result = minimize(loss_function, initial_guess, args=(window_data[['C_DELTA', 'C_IV', 'C_VEGA']], window_data['∆Vt']), method='SLSQP')
#     if result.success:
#         rolling_params.append(result.x)
#     else:
#         rolling_params.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  # 如果优化失败，添加NaN
#
# # 将滚动参数转换为DataFrame
# rolling_params_df = pd.DataFrame(rolling_params, columns=['a', 'b', 'c'])

# 步骤 4: 模型验证
# 查看a, b, c的变化：使用回归作为拟合工具，并检查这些参数的统计显著性。
# 检查δMV−δBS的形状：查看是否呈现（倒置的）抛物线形状。
# 绘制IV相对于Delta的预期变化图
# 绘制参数随时间的变化
rolling_params_df.plot(title='Rolling Parameters over Time')
plt.xlabel('Time')
plt.ylabel('Parameter Value')
plt.show()

import numpy as np

# # 计算模型估计的Delta
# data['δMV'] = np.nan  # 创建一个新列来存储模型估计的Delta
# for i in range(len(data)):
#     row = data.iloc[i]
#     params = rolling_params_df.iloc[i]  # 使用滚动估计的参数
#     data.at[i, 'δMV'] = model(params, row[['C_DELTA', 'C_IV', 'C_VEGA']])

# 检查δMV−δBS的形状
data['δMV_minus_δBS'] = data['δMV'] - data['C_DELTA']

# 绘制IV变化 vs Delta
plt.scatter(data['C_DELTA'], data['C_IV'], label='Actual IV Change', alpha=0.5)
plt.scatter(data['C_DELTA'], data['C_IV'] + data['δMV_minus_δBS'], label='Expected IV Change (IV + δMV−δBS)', alpha=0.5)
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('IV Change')
plt.title('Expected vs Actual IV Change')
plt.legend()
plt.show()

# 计算E[∆σimp]
data['E_∆σimp'] = data['C_IV'] + data['δMV_minus_δBS']

# 统计E[∆σimp]的平均值
mean_E_σimp = data['E_∆σimp'].mean()
print(f"平均E[∆σimp]: {mean_E_σimp}")

# 计算对冲盈利
data['Hedging_Gain'] = -data['E_∆σimp'] * data['C_VEGA'] * data['∆St'] / data['C_IV']

# 绘制对冲盈利图
plt.scatter(data['C_DELTA'], data['Hedging_Gain'], label='Hedging Gain', alpha=0.5)
plt.axhline(y=0.15, color='r', linestyle='--', label='15% Hedging Gain')
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('Hedging Gain')
plt.title('Hedging Gain vs Delta')
plt.legend()
plt.show()

# 查找满足15%对冲盈利的Delta桶和到期日
profitable_trades = data[data['Hedging_Gain'] >= 0.15]
print("满足15%对冲盈利的交易:")
print(profitable_trades[['EXPIRE_DATE', 'C_DELTA', 'Hedging_Gain']])
