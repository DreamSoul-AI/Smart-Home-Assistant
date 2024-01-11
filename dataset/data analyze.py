import pandas as pd
import matplotlib.pyplot as plt
filename = 'data_2015.csv'
# 读取dataset文件
dataset = pd.read_csv(filename)
df = dataset[['ts','d_type']].copy()
df['d_type'] = df['d_type'].apply(lambda x: 1 if x == 'controller' else 0)
df.to_csv('data1.csv',index = False)
# 创建散点图
plt.scatter(df['ts'], df['d_type'])
# 添加标题和轴标签
plt.title('散点图')
plt.xlabel('时间')
plt.ylabel('controller')
# 显示图形
plt.show()