import pandas as pd

data_path = 'processed_option_data.csv'
data = pd.read_csv(data_path)
# data.drop('δ2', axis=1)

# print(data.head())
# print(data.info())
#
# print(data.describe())

print('data.isnull().sum()',data.isnull().sum())
# 可以选择填充缺失值或删除含有缺失值的行
# data.fillna(method='ffill', inplace=True)  # 前向填充
data.dropna(inplace=True)  # 删除缺失值

import matplotlib.pyplot as plt

# 以∆Vt为例，绘制箱线图
plt.boxplot(data['∆Vt'].dropna())
plt.title('Box Plot of ∆Vt')
plt.show()

# 可以使用IQR规则来过滤异常值
Q1 = data['∆Vt'].quantile(0.25)
Q3 = data['∆Vt'].quantile(0.75)
IQR = Q3 - Q1
filter = (data['∆Vt'] >= Q1 - 1.5 * IQR) & (data['∆Vt'] <= Q3 + 1.5 * IQR)
data = data.loc[filter]

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化（Z-score标准化）
scaler = StandardScaler()
data[['∆St', '∆St/St', 'RETURNS', '∆Vt']] = scaler.fit_transform(data[['∆St', '∆St/St', 'RETURNS', '∆Vt']])

# 或者归一化到[0, 1]区间
# scaler = MinMaxScaler()
# data[['∆St', '∆St/St', 'RETURNS', '∆Vt']] = scaler.fit_transform(data[['∆St', '∆St/St', 'RETURNS', '∆Vt']])

from scipy.stats import norm
import seaborn as sns

# 绘制直方图
sns.histplot(data['∆Vt'], kde=True)
plt.title('Histogram of ∆Vt')
plt.show()

# 绘制Q-Q图
import scipy.stats as stats

stats.probplot(data['∆Vt'], dist="norm", plot=plt)
plt.title('Q-Q Plot of ∆Vt')
plt.show()

data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])
data.set_index('QUOTE_DATE', inplace=True)

# 绘制时间序列图
data['∆Vt'].plot()
plt.title('Time Series Plot of ∆Vt')
plt.show()

# 进行ADF检验来测试平稳性
from statsmodels.tsa.stattools import adfuller

# 进行ADF检验
result = adfuller(data['∆Vt'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 根据ADF检验结果，如果p-value小于显著性水平（通常为0.05），则拒绝原假设，认为序列是平稳的。