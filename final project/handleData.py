import numpy as np
import pandas as pd
from scipy.stats import linregress

# 读取文件
file_path = '/Users/nan/Downloads/spx_eod_2023q4-ai4uc9/spx_eod_202312.csv'
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
# 处理标题行，删除空格和单引号
df.columns = df.columns.str.strip().str.replace("'", '', regex=False).str.replace('[','', regex=False).str.replace(']','', regex=False)
print("Number of rows:", len(df))

# 处理term structure
expire_structures = []
for index, row in df.iterrows():
    dte = row['DTE']
    if dte <= 7:
        expire_structure = "Exclude"
    elif 15 <= dte <= 45:  # 1M ± 15天
        expire_structure = "1M"
    elif 75 <= dte <= 105:  # 3M ± 15*3天
        expire_structure = "3M"
    elif 165 <= dte <= 195:  # 6M ± 15*6天
        expire_structure = "6M"
    elif 255 <= dte <= 285:  # 9M ± 15*9天
        expire_structure = "9M"
    elif 345 <= dte <= 375:  # 12M ± 15*12天
        expire_structure = "12M"
    else:
        expire_structure = "Exclude"
    expire_structures.append(expire_structure)
# 将新列添加到数据框
df['EXPIRE_STRUCTURE'] = expire_structures
# 只保留 EXPIRE_STRUCTURE 列不等于 'Exclude' 的行
df = df[df['EXPIRE_STRUCTURE'] != 'Exclude']
print("Number of rows:", len(df))

# 处理delta范围
delta_min = 0.45
delta_max = 0.55
df = df[(df['C_DELTA'] > delta_min) & (df['C_DELTA'] < delta_max)]
print("Number of rows:", len(df))

# # 计算模型参数a,b,c
# # 创建空列表来存储计算得到的a, b, c的值
# a_values = []
# b_values = []
# c_values = []
# # ΔV = a * delta + b * gamma +c
# # ΔV = 相邻两行的call_last的差
# df['DELTA_V'] = df['call_last'].diff()
# # 手动将第一行的ΔV设置为零
# df['DELTA_V'].iloc[0] = 0
# for index, row in df.iterrows():
#     # 提取需要的列数据
#     delta = row['C_DELTA']
#     gamma = row['C_GAMMA']
#     delta_v = row['DELTA_V']
#     # 使用最小二乘法计算a, b, c
#     x = np.array([delta, gamma])
#     y = np.array(delta_v)
#     slope, intercept, r_value, p_value, std_err = linregress(x, y)
#     a_values.append(slope)
#     b_values.append(intercept)
#     c_values.append(0)  # c值暂时设为0，您可以根据需要修改
# # 将a, b, c的值添加到数据框中
# df['a'] = a_values
# df['b'] = b_values
# df['c'] = c_values
# # 打印包含新列的数据框
# print(df.head())

# 将结果保存到新的 CSV 文件
df.to_csv(file_path.split('/')[-1], index=False)


# 将结果保存到新的 CSV 文件
df.to_csv(file_path.split('/')[-1], index=False)