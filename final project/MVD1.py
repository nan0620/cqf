import numpy as np
import scipy.stats as stats
from sobol_seq import i4_sobol_generate
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map


# Read SPX option data
file_path = 'spx_eod_2023/spx_eod_202312.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.replace("'", '').str.replace('[', '').str.replace(']', '')

# Set parameters
params = {
    'S0': df['UNDERLYING_LAST'].values,     # Initial underlying asset price
    'K': df['STRIKE'].values,               # Option strike price
    'T': df['DTE'].values / 252.0,          # Time expiry (in years, assuming 252 trading days in a year)
    'r': 0.05,                              # Risk-free rate
    'sigma_actual': df['C_IV'].values,      # Actual volatility
    'mu': 0.05,                             # Expected rate of return, assumed to be equal to the risk-free rate
    'num_paths': len(df),                   # options in the dataset
    'num_steps': 252                        # time steps simulated
}

# Pre-calculate constants
dt = params['T'] / params['num_steps']
sobol_seq = i4_sobol_generate(1, params['num_paths'] * params['num_steps']).reshape(params['num_paths'], params['num_steps'])

def simulate_wrapper(path_id):
    return simulate_path(path_id, params, sobol_seq)

def simulate_path(path_id, params, sobol_seq):
    # Initialize arrays
    S = np.zeros(params['num_steps'] + 1)
    S[0] = params['S0'][path_id]
    
    # Simulate the path
    for j in range(1, params['num_steps'] + 1):
        Z = stats.norm.ppf(sobol_seq[path_id, j - 1])
        S[j] = S[j - 1] + (params['r'] * S[j - 1] * dt[path_id] + params['sigma_actual'][path_id] * S[j - 1] * np.sqrt(dt[path_id]) * Z + 0.5 * params['sigma_actual'][path_id] ** 2 * S[j - 1] * (Z ** 2 - 1) * dt[path_id])
    
    # Ensure that S does not contain zero or negative values
    S = np.maximum(S, 1e-10)
    
    # Calculate option prices and hedging
    C = np.maximum(S - params['K'][path_id], 0)

    # Prevent division by zero or negative numbers
    time_to_maturity = np.maximum(params['T'][path_id] - np.arange(params['num_steps'] + 1) * dt[path_id], 1e-10)

    # Real volatility hedging
    d1_actual = (np.log(S / params['K'][path_id]) + (params['r'] + 0.5 * params['sigma_actual'][path_id] ** 2) * time_to_maturity) / (params['sigma_actual'][path_id] * np.sqrt(time_to_maturity))
    delta_actual = stats.norm.cdf(d1_actual)    # Black-Scholes Delta
    
    # Calculate P&L
    # Initial portfolio: -C indicates purchase of options
    portfolio_actual = -C[0]
    portfolio_iv = -C[0]
    for j in range(params['num_steps']):
        # Real volatility hedging
        portfolio_actual += delta_actual[j] * (S[j + 1] - S[j])
        # Use Delta hedging in my data
        portfolio_iv += df['C_DELTA'].values[path_id] * (S[j + 1] - S[j])
    # Final portfolio value plus option value at expiry
    PnL_actual = portfolio_actual + C[-1]
    PnL_iv = portfolio_iv + C[-1]
    
    return PnL_actual, PnL_iv

if __name__ =='__main__':
    print(cpu_count())
    num_processes = cpu_count()
    tasks = range(params['num_paths'])

    # Multi-processing
    with Pool(processes=num_processes) as pool:
        results = process_map(simulate_wrapper, tasks, max_workers=num_processes, desc='Simulating Paths', chunksize=1)

    # # Multi-threading
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     tasks = range(params['num_paths'])
    #     # results = list(executor.map(lambda x: simulate_path(x, params, sobol_seq), range(params['num_paths'])))
    #     results = list(tqdm(executor.map(lambda x: simulate_path(x, params, sobol_seq), tasks), total=params['num_paths'], desc='Simulating Paths'))

    # Unpack results
    PnL_actual, PnL_iv = zip(*results)

    # Analysis and visualization
    mean_PnL_actual = np.mean(PnL_actual)
    std_PnL_actual = np.std(PnL_actual)
    mean_PnL_iv = np.mean(PnL_iv)
    std_PnL_iv = np.std(PnL_iv)

    print(f'Mean P&L for actual volatility hedging: {mean_PnL_actual}')
    print(f'Standard deviation of P&L for actual volatility hedging: {std_PnL_actual}')
    print(f'Mean P&L for implied volatility hedging: {mean_PnL_iv}')
    print(f'Standard deviation of P&L for implied volatility hedging: {std_PnL_iv}')

    # Visualised P&L distribution of real volatility hedges
    plt.figure(figsize=(10, 6))
    plt.hist(PnL_actual, bins=50, color='blue', alpha=0.5, label='Actual Volatility Hedging')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.title('P&L Distribution - Actual Volatility Hedging')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualising the P&L distribution using Delta hedging in my data
    plt.figure(figsize=(10, 6))
    plt.hist(PnL_iv, bins=50, color='red', alpha=0.5, label='Using My Data for Delta Hedging')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.title('P&L Distribution - Using My Data for Delta Hedging')
    plt.legend()
    plt.grid(True)
    plt.show()
