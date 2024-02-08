import pandas as pd
import matplotlib.pyplot as plt
import os


def process_and_plot_data(filename, output_dir=''):
    split_name = filename.split('.')
    split_name_forward = split_name[0]
    file_name = split_name_forward.split('\\')[-1]

    # 读取dataset文件
    dataset = pd.read_csv(filename)
    df = dataset[['ts', 'd_type']].copy()
    # 将controller出现时定为1，否则为0
    df['d_type'] = df['d_type'].apply(lambda x: 1 if x == 'controller' else 0)
    # df.to_csv('data01.csv',index = False)
    # df =pd.read_csv('data01.csv')
    # 将ts列中时间转换成datetime类型
    df['ts'] = pd.to_datetime(df['ts'], format='%Y-%m-%d %H:%M:%S')
    df1 = df.copy()
    # 将索引的时间部分转换为只包含小时
    df1['hour'] = df['ts'].dt.hour
    # 对其按0,1，~23共24个小时内controller出现的次数进行计数，忽略日期因素，用于了解其出现的小时分布情况
    grouped = df1.groupby('hour')['d_type'].sum()
    # 画柱状图
    grouped.plot(kind='bar', figsize=(10, 6), rot=45)
    plt.xlabel('hour')
    plt.ylabel('sum')
    plt.title('Bar charts')
    output_path1 = output_dir + '\\' + file_name + '_1.png'

    plt.savefig(output_path1)

    # 在原始的df上将时间列为索引
    df.set_index('ts', inplace=True)
    # 将时间戳按天进行分组，并计算每组中'1'的数量
    df2 = df.groupby(df.index.date)['d_type'].agg(['sum', 'count'])
    # 计算1占总次数的百分比
    df2['ratio'] = df2['sum'] / df2['count']
    # 画双坐标图
    bar_data = df2['sum']
    line_data = df2['ratio']
    fig, ax1 = plt.subplots()
    # 绘制柱状图并指定颜色（例如，蓝色）
    ax1.bar(bar_data.index, bar_data.values, color='blue', alpha=1)  # 'alpha' 参数可以调整透明度
    # 创建与ax1共享x轴的第二个y轴用于绘制折线图
    ax2 = ax1.twinx()
    # 绘制折线图并指定另一种颜色（例如，红色）
    ax2.plot(line_data.index, line_data.values, color='red', alpha=0.2, linestyle='-', linewidth=0.5)
    # 设置两个坐标轴的标签
    ax1.set_xlabel('day')
    ax1.set_ylabel('sum')
    ax2.set_ylabel('占比')

    plt.title('Bar charts and scatter plots')
    output_path2 = output_dir + '\\' + file_name + '_2.png'
    plt.savefig(output_path2)


def process(folder_path='', output_dir=''):
    file_names_list = []
    # 使用os.walk遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名是否符合data_0000.csv格式
            if file.startswith('data_') and file.endswith('.csv') and len(file[5:-4]) == 4:
                # 符合条件则添加到列表
                file_names_list.append(os.path.join(root, file))
    for i in file_names_list:
        process_and_plot_data(i, output_dir)
