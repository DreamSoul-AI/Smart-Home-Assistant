import pandas as pd
from scipy.interpolate import interp1d, make_interp_spline, UnivariateSpline
from scipy.signal import resample
import numpy as np
import os


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def data_interpolate(df):
    df['ts'] = pd.to_datetime(df['ts'], format='%Y-%m-%d %H:%M:%S.%f')

    plt.figure(figsize=(14, 7))  # 设置图形大小
    plt.scatter(df['ts'], df['d_value'], color='black', label='Original Data Points', alpha=0.6, s=30)

    df_i = df['ts'].dt.floor('s')
    df_i = df_i.astype(np.int64)

    # 确保'ts'列的数据类型为int64
    df['ts'] = df['ts'].astype(np.int64)

    # 提取x和y用于插值
    x = df['ts'].values
    y = df['d_value'].values

    y_min = y.min()
    y_max = y.max()

    # 使用指定的kind进行插值
    f1 = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
    f2 = UnivariateSpline(x, y, k=2, s=500)
    f3 = interp1d(x, y, kind='previous', bounds_error=False, fill_value=np.nan)
    f5 = make_interp_spline(x, y, k=3)

    # 计算新的时间戳范围和步长
    start_ts = df_i.values.min()
    end_ts = df_i.values.max()
    step_ns = 1_000_000_000  # 每秒的纳秒数
    new_ts = np.arange(start_ts, end_ts + step_ns + step_ns, step_ns)

    # 执行插值
    ynew1 = f1(new_ts)
    ynew2 = f2(new_ts)
    ynew3 = f3(new_ts)
    ynew4 = resample(y, len(new_ts))
    # ynew5 = np.clip(f5(new_ts), y_min, y_max)

    # 转换新的时间戳为datetime格式
    new_ts_datetime = pd.to_datetime(new_ts, unit='ns')

    # 构建结果DataFrame
    result_df = pd.DataFrame({
        'ts': new_ts_datetime,
        'd_value_Interp1dLinear': ynew1,
        'd_value_UnivariateSpline': ynew2,
        'd_value_Interp1dPrevious': ynew3,
        # 'd_value_FFT_resample': ynew4,
        # 'd_value_make_interp_spline': ynew5
    })

    # 保存结果到CSV文件（可选，根据需要调用）
    result_df.to_csv('output.csv', index=False)

    # 绘制第一条曲线 ynew1
    plt.plot(result_df['ts'], result_df['d_value_Interp1dLinear'], color='red', label='Interp1dLinear Method', linewidth=2)

    # 绘制第二条曲线 ynew2
    plt.plot(result_df['ts'], result_df['d_value_UnivariateSpline'], color='green', label='UnivariateSpline Method', linewidth=2)

    # 绘制第三条曲线 ynew3
    plt.plot(result_df['ts'], result_df['d_value_Interp1dPrevious'], color='blue', label='Interp1dPrevious Method', linewidth=2)

    # # 绘制第四条曲线 ynew4
    # plt.plot(result_df['ts'], result_df['d_value_FFT_resample'], color='yellow', label='FFT_resample Method', linewidth=2)

    # # 绘制第五条曲线 ynew5
    # plt.plot(result_df['ts'], result_df['d_value_make_interp_spline'], color='blue', label='make_interp_spline Method', linewidth=2)

    # 添加标题和坐标轴标签
    plt.title('Comparison of Interpolation Methods')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True, which="both", ls="--", alpha=0.3)

    # 优化x轴日期时间显示（可选，根据实际需要调整）
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))

    # 显示图形
    plt.show()
    return result_df


data_path = '#####'  # 填写你的数据路径
file_name = ['data_2011_L006.csv', 'data_2011_LS017.csv', 'data_2012_M006.csv']

path = [os.path.join(data_path, i) for i in file_name]

data = pd.read_csv(path[1])
# data = data[0:100]

data_i = data_interpolate(data.copy())

