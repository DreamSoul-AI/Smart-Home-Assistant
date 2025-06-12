import pandas as pd
import os

print("=== 数据集状态分析 ===\n")

# 检查data_2015.csv
if os.path.exists('data_2015.csv'):
    print("✓ 找到 data_2015.csv 文件")
    
    # 读取前几行看看数据格式
    df = pd.read_csv('data_2015.csv', nrows=5)
    print("\n数据格式预览：")
    print(df)
    
    print("\n列名：", df.columns.tolist())
    
    # 检查是否包含所需的列
    required_columns = ['ts', 'd_name', 'd_value']
    has_all_columns = all(col in df.columns for col in required_columns)
    
    if has_all_columns:
        print("✓ 包含所有必需的列：", required_columns)
    else:
        missing_columns = [col for col in required_columns if col not in df.columns]
        print("✗ 缺少必需的列：", missing_columns)
        
    # 读取更多数据来分析
    df_full = pd.read_csv('data_2015.csv', nrows=1000)
    print(f"\n数据集大小（前1000行）：{len(df_full)} 行")
    
    # 检查d_name的唯一值
    unique_devices = df_full['d_name'].unique()
    print(f"\n设备数量：{len(unique_devices)}")
    print("部分设备名称：", unique_devices[:10])
    
else:
    print("✗ 未找到 data_2015.csv 文件")

print("\n" + "="*50 + "\n")

# 检查env.csv
if os.path.exists('env.csv'):
    print("✓ 找到 env.csv 文件")
    
    env_df = pd.read_csv('env.csv')
    print("\nenv.csv 预览：")
    print(env_df.head())
    
    print(f"\n环境配置条目数：{len(env_df)}")
    print("设备类型：", env_df['d_type'].unique())
    
else:
    print("✗ 未找到 env.csv 文件")

print("\n" + "="*50 + "\n")

# 检查目录结构
print("当前目录结构：")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"📁 {item}/")
    else:
        print(f"📄 {item}")

print("\n" + "="*50 + "\n")

# 分析数据预处理状态
print("数据预处理状态分析：")

# 检查是否有SmartHome期望的目录结构
expected_dirs = ['data', 'data/SmartHome', 'data/SmartHome/raw', 
                 'data/SmartHome/preprocessed', 'data/SmartHome/processed']

missing_dirs = []
for dir_path in expected_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path} 存在")
    else:
        print(f"✗ {dir_path} 不存在")
        missing_dirs.append(dir_path)

if missing_dirs:
    print(f"\n⚠️ 缺少 {len(missing_dirs)} 个必需的目录")
    print("\n结论：数据集尚未完成预处理")
    print("\n建议的下一步：")
    print("1. 创建必需的目录结构")
    print("2. 将data_2015.csv移动到正确的位置")
    print("3. 运行预处理流程")
else:
    print("\n✓ 所有必需的目录都存在")
    print("\n结论：数据集可能已经完成预处理，请检查preprocessed文件夹中的内容") 