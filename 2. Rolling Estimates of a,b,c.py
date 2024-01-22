from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from arch import arch_model
from pandas import concat
from scipy.optimize import minimize
from tqdm.contrib.concurrent import process_map

# Create an empty DataFrame to store estimation results
results = pd.DataFrame()

# Define a function for parameter estimation
def estimate_params(iv, delta, vega, dVt):
    # Define the optimization objective function
    def objective(params, iv, delta, vega, dVt):
        a, b, c = params
        # Define how to calculate δBS from IV, Delta, and Vega
        delta_bs = a * iv + b * delta + c * vega
        # Define the objective function, such as squared error
        return np.sum((delta_bs - dVt) ** 2)

    # Optimize parameters using SLSQP
    initial_guess = np.array([0.1, 0.1, 0.1])  # Initial guess
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]  # Parameter bounds
    result = minimize(objective, initial_guess, args=(iv, delta, vega, dVt), method='SLSQP', bounds=bounds)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed")

# Define a function to estimate volatility using the GARCH model
def estimate_volatility(returns):
    # Remove NaN values resulting from calculating returns
    returns = returns.dropna()
    # Fit a GARCH(1,1) model
    garch11 = arch_model(returns, p=1, q=1)
    res = garch11.fit(disp='off')  # Turn off output
    # Extract estimated volatility from the model
    estimated_volatility = res.conditional_volatility
    return estimated_volatility

# Define a function to process each subtask
def process_subtask(subtask_data):
    # Load CSV data
    data_path = 'processed_option_data.csv'
    data = pd.read_csv(data_path)

    # Convert the date column to datetime type
    data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])

    # Define the rolling window size
    rolling_window_size = 3 * 22
    results = pd.DataFrame()

    # Iterate over each expiration structure
    for term in subtask_data['EXPIRE_STRUCTURE'].unique():
        term_data = subtask_data[subtask_data['EXPIRE_STRUCTURE'] == term]

        # Iterate over each strike price
        for strike in term_data['STRIKE'].unique():
            strike_data = term_data[term_data['STRIKE'] == strike]

            # Rolling window estimation
            for i in range(rolling_window_size, len(strike_data)):
                window_data = strike_data.iloc[i - rolling_window_size:i]

                # Extract other parameters from the rolling window data
                iv = window_data['C_IV']
                delta = window_data['C_DELTA']
                vega = window_data['C_VEGA']
                dVt = window_data['∆Vt']
                dSt = window_data['∆St']

                # Use GARCH model and SLSQP optimization to estimate parameters a, b, c
                try:
                    param_a, param_b, param_c = estimate_params(iv, delta, vega, dVt)

                    # Add the estimation results to the results DataFrame
                    results = concat([results, pd.DataFrame({
                        'QUOTE_DATE': [window_data.iloc[-1]['QUOTE_DATE']],
                        'STRIKE': [strike],
                        'TERM': [term],
                        'PARAM_A': [param_a],
                        'PARAM_B': [param_b],
                        'PARAM_C': [param_c],
                        'C_IV': [iv.iloc[-1]],
                        'C_DELTA': [delta.iloc[-1]],
                        'C_VEGA': [vega.iloc[-1]],
                        '∆Vt': [dVt.iloc[-1]],
                        '∆St': [dSt.iloc[-1]]
                    })], ignore_index=True)
                except ValueError:
                    print(f"Optimization failed for window ending on {window_data.iloc[-1]['QUOTE_DATE']}")

    # Return the results of this subtask
    return results

def main():
    # Load data
    data_path = 'processed_option_data.csv'
    data = pd.read_csv(data_path)
    data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])

    # Define the number of subtasks, typically equal to the number of CPU cores
    num_subtasks = cpu_count()

    # Split the data into subtasks
    subtasks = np.array_split(data, num_subtasks)

    # Use tqdm's process_map to display a progress bar
    # Set max_workers parameter to specify the number of processes, and adjust chunksize as needed
    results_list = process_map(process_subtask, subtasks, max_workers=num_subtasks, chunksize=1)

    # Merge results from all subtasks
    results = pd.concat(results_list, ignore_index=True)

    # Output the results
    print(results)
    results.to_csv('new_processed_option_data_with_abc.csv', index=False)

if __name__ == '__main__':
    main()
