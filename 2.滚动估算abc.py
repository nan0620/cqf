import time
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from arch import arch_model
from matplotlib import pyplot as plt
from pandas import concat
from scipy.optimize import minimize
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# 创建一个空的DataFrame来存储估计结果
results = pd.DataFrame()

# 定义估计参数的函数
def estimate_params(iv, delta, vega, dVt):
    # 定义优化目标函数
    def objective(params, iv, delta, vega, dVt):
        a, b, c = params
        # 定义如何从IV、Delta、Vega计算δBS
        delta_bs = a * iv + b * delta + c * vega
        # 定义目标函数，比如平方误差
        return np.sum((delta_bs - dVt) ** 2)

    # 用SLSQP优化参数
    initial_guess = np.array([0.1, 0.1, 0.1])  # 初始猜测值
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]  # 参数界限
    result = minimize(objective, initial_guess, args=(iv, delta, vega, dVt), method='SLSQP', bounds=bounds)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed")


# 定义使用GARCH模型估计波动率的函数
def estimate_volatility(returns):
    # 清除因为计算收益率而产生的NaN值
    returns = returns.dropna()
    # 拟合GARCH(1,1)模型
    garch11 = arch_model(returns, p=1, q=1)
    res = garch11.fit(disp='off')  # 关闭输出
    # 提取模型估计的波动率
    estimated_volatility = res.conditional_volatility
    return estimated_volatility


# 定义一个处理每个子任务的函数
def process_subtask(subtask_data):
    # 加载CSV数据
    data_path = 'processed_option_data.csv'
    data = pd.read_csv(data_path)

    # 转换日期列为datetime类型
    data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])

    # 定义滚动窗口大小
    rolling_window_size = 3 * 22
    results = pd.DataFrame()

    # # 对每个DTE值进行循环
    # for dte in subtask_data['DTE'].unique():
    #     dte_data = subtask_data[subtask_data['DTE'] == dte]
    #
    #     # 对每个行权价进行循环
    #     for strike in dte_data['STRIKE'].unique():
    #         strike_data = dte_data[dte_data['STRIKE'] == strike]

    # 对每个期权到期结构进行循环
    for term in subtask_data['EXPIRE_STRUCTURE'].unique():
        term_data = subtask_data[subtask_data['EXPIRE_STRUCTURE'] == term]

        # 对每个行权价进行循环
        for strike in term_data['STRIKE'].unique():
            strike_data = term_data[term_data['STRIKE'] == strike]

            # 滚动窗口估算
            for i in range(rolling_window_size, len(strike_data)):
                window_data = strike_data.iloc[i - rolling_window_size:i]

                # 提取滚动窗口数据中的其他参数
                iv = window_data['C_IV']
                delta = window_data['C_DELTA']
                vega = window_data['C_VEGA']
                dVt = window_data['∆Vt']
                dSt = window_data['∆St']
                # 使用GARCH模型和SLSQP优化来估计参数a, b, c
                try:
                    param_a, param_b, param_c = estimate_params(iv, delta, vega, dVt)

                    # 将估计结果添加到结果DataFrame中
                    results = concat([results, pd.DataFrame({
                        'QUOTE_DATE': [window_data.iloc[-1]['QUOTE_DATE']],
                        'STRIKE': [strike],
                        'TERM': [term],
                        # 'DTE': [dte],
                        'PARAM_A': [param_a],
                        'PARAM_B': [param_b],
                        'PARAM_C': [param_c],
                        'C_IV': [iv.iloc[-1]],
                        'C_DELTA': [delta.iloc[-1]],
                        'C_VEGA': [vega.iloc[-1]],
                        '∆Vt': [dVt.iloc[-1]],
                        '∆St':[dSt.iloc[-1]]

                    })], ignore_index=True)
                except ValueError:
                    print(f"Optimization failed for window ending on {window_data.iloc[-1]['QUOTE_DATE']}")

    # 返回该子任务的结果
    return results


def main():
    # 加载数据
    data_path = 'processed_option_data.csv'
    data = pd.read_csv(data_path)
    data['QUOTE_DATE'] = pd.to_datetime(data['QUOTE_DATE'])

    # 定义子任务的数量，通常等于CPU核心数
    num_subtasks = cpu_count()

    # 将数据分割成子任务
    subtasks = np.array_split(data, num_subtasks)

    # 使用tqdm的process_map来显示进度条
    # max_workers参数设置进程数，chunksize可以根据需要调整
    results_list = process_map(process_subtask, subtasks, max_workers=num_subtasks, chunksize=1)

    # 合并所有子任务的结果
    results = pd.concat(results_list, ignore_index=True)

    # 输出结果
    print(results)
    results.to_csv('new_processed_option_data_with_abc.csv', index=False)


if __name__ == '__main__':
    main()

