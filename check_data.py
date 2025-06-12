import pandas as pd

# 读取前10行数据
df = pd.read_csv('data_2015.csv', nrows=10)

print("前5行数据：")
print(df.head())
print("\n列名：", df.columns.tolist())
print("数据形状：", df.shape)
print("\n数据类型：")
print(df.dtypes) 