import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/nanjiang/cqf/new_processed_option_data_with_abc.csv')

# 这里的delta_bs可以被理解为模型估计的值，其中a、b、c是模型参数。
# 在我的模型中，关于Delta的偏导数相对简单，因为模型是线性的。模型估计的Delta可以直接从我的模型中获取，即系数b。
MODEL_DELTA_EST = data['PARAM_B']

# 检查δMV−δBS的形状
data['δMV'] = data['∆St']

# 绘制IV变化 vs Delta
plt.scatter(data['C_DELTA'], data['C_IV'], label='Actual IV Change', alpha=0.5)
plt.scatter(data['C_DELTA'], data['C_IV'] + data['δMV'], label='Expected IV Change (IV + δMV−δBS)', alpha=0.5)
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('IV Change')
plt.title('Expected vs Actual IV Change')
plt.legend()
plt.show()

# 计算E[∆σimp]
data['E_∆σimp'] = data['C_IV'] + data['δMV']

# 统计E[∆σimp]的平均值
mean_E_σimp = data['E_∆σimp'].mean()
print(f"平均E[∆σimp]: {mean_E_σimp}")

# 计算对冲盈利
data['Hedging_Gain'] = -data['E_∆σimp'] * data['C_VEGA'] * data['∆Vt'] / data['C_IV']
data.to_csv('/Users/nanjiang/cqf/15_percent_profitable_trades.csv', index=False)

# 查找满足15%对冲盈利的Delta桶和到期日
profitable_trades = data[data['Hedging_Gain'] >= 0.15]
percent = len(profitable_trades) / len(data['Hedging_Gain'])
print(percent)

print("满足15%对冲盈利的交易:")
print(profitable_trades[['QUOTE_DATE', 'C_DELTA', 'Hedging_Gain']])
