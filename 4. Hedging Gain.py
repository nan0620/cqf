import pandas as pd
from matplotlib import pyplot as plt

# Read data from a CSV file
data = pd.read_csv('/new_processed_option_data_with_abc.csv')

# Here, delta_bs can be understood as the model estimate, where a, b, and c are model parameters.
# In my model, the partial derivative with respect to Delta is relatively simple because the model is linear.
# The model-estimated Delta can be directly obtained from my model, which is the coefficient b.
MODEL_DELTA_EST = data['PARAM_B']

# Check the shape of δMV−δBS
data['δMV'] = data['∆St']

# Plot IV Change vs. Delta
plt.scatter(data['C_DELTA'], data['C_IV'], label='Actual IV Change', alpha=0.5)
plt.scatter(data['C_DELTA'], data['C_IV'] + data['δMV'], label='Expected IV Change (IV + δMV−δBS)', alpha=0.5)
plt.xlabel('Delta (C_DELTA)')
plt.ylabel('IV Change')
plt.title('Expected vs Actual IV Change')
plt.legend()
plt.show()

# Calculate E[∆σimp]
data['E_∆σimp'] = data['C_IV'] + data['δMV']

# Calculate the mean of E[∆σimp]
mean_E_σimp = data['E_∆σimp'].mean()
print(f"Average E[∆σimp]: {mean_E_σimp}")

# Calculate hedging gains
data['Hedging_Gain'] = -data['E_∆σimp'] * data['C_VEGA'] * data['∆Vt'] / data['C_IV']
data.to_csv('/Users/nanjiang/cqf/15_percent_profitable_trades.csv', index=False)

# Find Delta buckets and expiration dates that meet the 15% hedging gain
profitable_trades = data[data['Hedging_Gain'] >= 0.15]
percent = len(profitable_trades) / len(data['Hedging_Gain'])
print(percent)

print("Trades with 15% or more hedging gains:")
print(profitable_trades[['QUOTE_DATE', 'C_DELTA', 'Hedging_Gain']])
