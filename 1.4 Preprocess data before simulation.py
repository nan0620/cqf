import numpy as np
import pandas as pd

# Step 1: Organize data and calculate necessary variables
# Load the CSV file
data = pd.read_csv('/Users/nanjiang/cqf/spx_eod_2021-2023_combined.csv', low_memory=False)
# Ensure the DATE column is of date type
data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])
# Ensure C_LAST is of numeric type
data['C_LAST'] = pd.to_numeric(data['C_LAST'], errors='coerce')
# Sort the data
data.sort_values(by='QUOTE_DATE', inplace=True)

# Step 2: Calculate dependent variables and run fitting
# Select columns of interest, focusing on call options
columns_of_interest = ['QUOTE_DATE', 'UNDERLYING_LAST', 'DTE', 'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV', 'C_LAST', 'STRIKE', 'EXPIRE_STRUCTURE']
data = data[columns_of_interest]

# Variables needed
# 1. ∆St, daily change in underlying asset price
data['∆St'] = data['UNDERLYING_LAST'].diff()
# 2. ∆St/St, daily change rate in underlying asset price
data['∆St/St'] = data['UNDERLYING_LAST'].pct_change()
data['RETURNS'] = np.log(data['UNDERLYING_LAST'] / data['UNDERLYING_LAST'].shift(1))

# 3. ∆Vt, daily change in option price
data['∆Vt'] = data['C_LAST'].diff()
# 4. St, underlying asset price
# 5. IV, implied volatility
# 6. Delta, sensitivity of option value to changes in underlying asset price
# 7. Vega, sensitivity of option price to changes in volatility
# 8. δ2, square of Delta
data['δ2'] = data['C_DELTA'] ** 2
data.dropna(inplace=True)
data.to_csv('/Users/nanjiang/cqf/processed_option_data.csv', index=False)
