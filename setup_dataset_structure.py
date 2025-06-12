import os
import shutil
import pandas as pd

print("=== 设置SmartHome数据集结构 ===\n")

# 创建必需的目录结构
directories = [
    'data',
    'data/SmartHome',
    'data/SmartHome/raw',
    'data/SmartHome/preprocessed',
    'data/SmartHome/preprocessed/hh105',
    'data/SmartHome/preprocessed/hh105/all',
    'data/SmartHome/preprocessed/hh105/year',
    'data/SmartHome/preprocessed/hh105/year/device',
    'data/SmartHome/processed',
    'data/SmartHome/processed/hh105',
    'data/SmartHome/processed/hh105/2015'
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ 创建目录: {directory}")
    else:
        print(f"- 目录已存在: {directory}")

print("\n=== 组织数据文件 ===\n")

# 复制data_2015.csv到正确的位置
if os.path.exists('data_2015.csv'):
    # 1. 保存到year文件夹
    year_path = 'data/SmartHome/preprocessed/hh105/year/data_2015.csv'
    if not os.path.exists(year_path):
        shutil.copy2('data_2015.csv', year_path)
        print(f"✓ 复制 data_2015.csv 到 {year_path}")
    
    # 2. 读取数据并按设备分割
    print("\n正在按设备分割数据...")
    df = pd.read_csv('data_2015.csv')
    unique_devices = df['d_name'].unique()
    
    device_dir = 'data/SmartHome/preprocessed/hh105/year/device'
    for device in unique_devices:
        device_df = df[df['d_name'] == device]
        device_file = os.path.join(device_dir, f'data_2015_{device}.csv')
        device_df.to_csv(device_file, index=False)
    
    print(f"✓ 已创建 {len(unique_devices)} 个设备文件")

# 复制env.csv
if os.path.exists('env.csv'):
    env_dest = 'data/SmartHome/preprocessed/hh105/env.csv'
    if not os.path.exists(env_dest):
        shutil.copy2('env.csv', env_dest)
        print(f"\n✓ 复制 env.csv 到 {env_dest}")

# 创建data_info.json（如果需要）
import json

data_info = {
    "hh105": {
        "column_index": {
            "date": 1,
            "time": 2,
            "d_name": 3,
            "d_value": 4
        },
        "rooms": ["Bathroom", "Bedroom", "DiningRoom", "Kitchen", "LivingRoom", "OutsideDoor"]
    }
}

data_info_path = 'data/SmartHome/data_info.json'
if not os.path.exists(data_info_path):
    with open(data_info_path, 'w') as f:
        json.dump(data_info, f, indent=4)
    print(f"\n✓ 创建 {data_info_path}")

print("\n=== 完成！===")
print("\n现在您可以：")
print("1. 运行 src/make_dataset.py 来创建最终的数据集")
print("2. 或者直接使用已经组织好的数据进行训练")
print("\n数据集结构已经准备就绪！") 