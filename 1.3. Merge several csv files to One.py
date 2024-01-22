import os
import pandas as pd

# Set the folder path containing CSV files
folder_path = '/Users/nanjiang/cqf/spx_eod_handled'

# Create an empty list of DataFrames to store data from each CSV file
dataframes = []

# Traverse all CSV files in the folder and read them into the list of DataFrames
file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith('.csv')])
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    # Read the CSV file and append it to the list of DataFrames
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Use the concat function to merge the list of DataFrames into one large DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame as a new CSV file
combined_csv_path = '/Users/nanjiang/cqf/spx_eod_2021-2023_combined.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f'The merged CSV file has been saved as: {combined_csv_path}')
