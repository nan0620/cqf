import os
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Set the paths
directory = '/Users/nanjiang/cqf/spx_eod_unhandled'
# directory = '/Users/nanjiang/cqf/spx_eod_2021'
processed_directory = '/Users/nanjiang/cqf/spx_eod_handled'

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Full path of the file
        file_path = os.path.join(directory, filename)
        # Read the file
        df = pd.read_csv(file_path, low_memory=False)
        pd.set_option('display.max_columns', None)
        # Process header row, remove spaces and single quotes
        df.columns = df.columns.str.strip().str.replace("'", '', regex=False).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        print("Number of rows:", len(df))

        # Process term structure
        expire_structures = []
        for index, row in df.iterrows():
            dte = row['DTE']
            if dte <= 7:
                expire_structure = "Exclude"
            elif 15 <= dte <= 45:  # 1M ± 15 days
                expire_structure = "1M"
            elif 75 <= dte <= 105:  # 3M ± 15*3 days
                expire_structure = "3M"
            elif 165 <= dte <= 195:  # 6M ± 15*6 days
                expire_structure = "6M"
            elif 255 <= dte <= 285:  # 9M ± 15*9 days
                expire_structure = "9M"
            elif 345 <= dte <= 375:  # 12M ± 15*12 days
                expire_structure = "12M"
            else:
                expire_structure = "Exclude"
            expire_structures.append(expire_structure)
        # Add a new column to the DataFrame
        df['EXPIRE_STRUCTURE'] = expire_structures
        # Keep rows where EXPIRE_STRUCTURE column is not 'Exclude'
        df = df[df['EXPIRE_STRUCTURE'] != 'Exclude']
        print("Number of rows:", len(df))

        # Process delta range
        delta_min = 0.45
        delta_max = 0.55
        df['C_DELTA'] = pd.to_numeric(df['C_DELTA'], errors='coerce')  # Convert the column to numeric, handling non-numeric values
        df = df[(df['C_DELTA'] > delta_min) & (df['C_DELTA'] < delta_max)]
        print("Number of rows:", len(df))

        # Save the processed data to a new file
        new_file_path = os.path.join(processed_directory, 'new_processed_' + filename)
        df.to_csv(new_file_path, index=False)

        print(f'File {filename} has been processed and saved as {new_file_path}')
