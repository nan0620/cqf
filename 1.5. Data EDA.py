import pandas as pd

data_path = 'processed_option_data.csv'
data = pd.read_csv(data_path)
# data.drop('δ2', axis=1)

# print(data.head())
# print(data.info())
#
# print(data.describe())

print('data.isnull().sum()', data.isnull().sum())
# You can choose to fill missing values or delete rows with missing values
# data.fillna(method='ffill', inplace=True)  # Forward fill
data.dropna(inplace=True)  # Delete rows with missing values

import matplotlib.pyplot as plt

# For example, plotting a boxplot for ∆Vt
plt.boxplot(data['∆Vt'].dropna())
plt.title('Box Plot of ∆Vt')
plt.show()

# You can use the IQR rule to filter outliers
Q1 = data['∆Vt'].quantile(0.25)
Q3 = data['∆Vt'].quantile(0.75)
IQR = Q3 - Q1
filter = (data['∆Vt'] >= Q1 - 1.5 * IQR) & (data['∆Vt'] <= Q3 + 1.5 * IQR)
data = data.loc[filter]

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (Z-score standardization)
scaler = StandardScaler()
data[['∆St', '∆St/St', 'RETURNS', '∆Vt']] = scaler.fit_transform(data[['∆St', '∆St/St', 'RETURNS', '∆Vt']])

# Or normalize to the [0, 1] range
# scaler = MinMaxScaler()
# data[['∆St', '∆St/St', 'RETURNS', '∆Vt']] = scaler.fit_transform(data[['∆St', '∆St/St', 'RETURNS', '∆Vt']])

from scipy.stats import norm
import seaborn as sns

# Plot a histogram
sns.histplot(data['∆Vt'], kde=True)
plt.title('Histogram of ∆Vt')
plt.show()

# Plot a Q-Q plot
import scipy.stats as stats

stats.probplot(data['∆Vt'], dist="norm", plot=plt)
plt.title('Q-Q Plot of ∆Vt')
plt.show()

data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])
data.set_index('QUOTE_DATE', inplace=True)

# Plot a time series graph
data['∆Vt'].plot()
plt.title('Time Series Plot of ∆Vt')
plt.show()

# Perform an ADF test to test for stationarity
from statsmodels.tsa.stattools import adfuller

# Perform the ADF test
result = adfuller(data['∆Vt'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Based on the ADF test result, if the p-value is less than the significance level (usually 0.05), reject the null hypothesis, and consider the series stationary.
