import os
import pandas as pd

# 设置包含CSV文件的文件夹路径
folder_path = '/Users/nanjiang/cqf/spx_eod_handled'

# 创建一个空的DataFrame列表，用于存储每个CSV文件的数据
dataframes = []

# 遍历文件夹中的所有CSV文件并按顺序读取到DataFrame列表中
file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith('.csv')])
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    # 读取CSV文件并将其添加到DataFrame列表中
    df = pd.read_csv(file_path)
    dataframes.append(df)

# 使用concat函数将DataFrame列表合并成一个大的DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# 将合并后的DataFrame保存为一个新的CSV文件
combined_csv_path = '/Users/nanjiang/cqf/spx_eod_2021-2023_combined.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f'合并后的CSV文件已保存为：{combined_csv_path}')
