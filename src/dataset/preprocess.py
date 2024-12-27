# -*- coding: utf-8 -*-
# 基础库
import os
import pandas as pd
import json
import re

from scipy.interpolate import interp1d
import numpy as np


data_wash_dict = {
    0: {
        'action_type': 'drop',
        'sub_action_type': 'contain',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': ['TAP_COUNT', 'HOLD_DEPRESS', 'HOLD_RELEASE', 'RELEASE']
    },
    1: {
        'action_type': 'replace',
        'sub_action_type': 'equal',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': {"ON": 1.0, "OFF": 0.0, "OPEN": 1.0, "CLOSE": 0.0}
    },
    2: {
        'action_type': 'drop',
        'sub_action_type': 'contain',
        'column_name': 'd_name',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': ['BATP', 'ZB', 'HOME']
    },
    3: {
        'action_type': 'drop',
        'sub_action_type': 'equal',
        'column_name': 'd_name',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': ['c'],
        'target_value': []
    },
    4: {
        'action_type': 'replace',
        'sub_action_type': 'c_equal',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': 'd_name',
            'c_type': 'contain',
            'c_value1': ['ButtonDown'],
            'c_value2': None
        },
        'input_value': 0.0
    },
    5: {
        'action_type': 'replace',
        'sub_action_type': 'c_equal',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': 'd_name',
            'c_type': 'contain',
            'c_value1': ['ButtonUp'],
            'c_value2': None
        },
        'input_value': 1.0
    },
    6: {
        'action_type': 'replace',
        'sub_action_type': 'contain',
        'column_name': 'd_name',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': {'ButtonUp': 'Button', 'ButtonDown': 'Button'}
    },
    7: {
        'action_type': 'replace',
        'sub_action_type': 'c_equal',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': 'd_name',
            'c_type': 'contain',
            'c_value1': ['L0'],
            'c_value2': None
        },
        'input_value': {'100': '1'}
    },
    8: {
        'action_type': 'drop',
        'sub_action_type': 'cant_trans_to_num',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': None
    },
    9: {
        'action_type': 'trans',
        'sub_action_type': 'value_type',
        'column_name': 'd_name',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': 'str'
    },
    10: {
        'action_type': 'trans',
        'sub_action_type': 'value_type',
        'column_name': 'd_value',
        'condition': {
            'c_column_name': None,
            'c_type': None,
            'c_value1': None,
            'c_value2': None
        },
        'input_value': float
    }
}


class Preprocess:
    def __init__(self, root_path, proj_name, split_by='year'):
        self.root_path = root_path  # 根路径
        self.proj_path = os.path.join(root_path, proj_name)  # 项目路径
        self.raw_data_path = os.path.join(self.proj_path, 'raw')  # 原始数据路径
        self.processed_data_path = os.path.join(self.proj_path, 'preprocessed')  # 处理后数据路径
        self.data_info_path = os.path.join(self.proj_path, 'data_info.json')  # 每个数据集所选列的json文件保存路径

        self.dataset_names = []  # 数据集名称

        self.directory_structure = {}  # 目录结构

        # 初始化
        self.build_directory_structure_dict()
        self.create_directory()
        self.data_info_dict = self.load_data_info()  # 数据集信息

        # 处理数据
        self.process_dataset(split_by=split_by)

    # 初始化函数
    def build_directory_structure_dict(self, sub_folders=('all', 'year', 'device')):

        for dir_name in [self.root_path, self.proj_path, self.raw_data_path, self.processed_data_path]:
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

        filenames = os.listdir(self.raw_data_path)

        for filename in filenames:
            dataset_dict = {}
            for sub_folder in sub_folders:
                dataset_dict[sub_folder] = []
            filename = filename.split('.')[0]
            self.dataset_names.append(filename)
            self.directory_structure[filename] = dataset_dict
        return

    def create_directory(self, directory_structure=None, base_path=None):

        if directory_structure is None:
            directory_structure = self.directory_structure

        if base_path is None:
            base_path = self.processed_data_path

        for key, value in directory_structure.items():
            new_path = os.path.join(base_path, key)

            # dict，递归本函数
            if isinstance(value, dict):
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                self.create_directory(value, new_path)
            # list，最终目录
            if isinstance(value, list):
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
        return

    def load_data_info(self):
        if not os.path.exists(self.data_info_path):
            self.json_act(self.data_info_path, 'dump')
        return self.json_act(self.data_info_path, 'load')

    def process_dataset(self, dataset_names=None, split_by='year'):
        dataset = []
        if dataset_names is None:
            dataset_names = self.dataset_names

        data_info = self.json_act(self.data_info_path, 'load')

        for dataset_name in dataset_names:
            processed_dataset_path = os.path.join(self.processed_data_path, dataset_name)

            column_indexes = data_info[dataset_name]['column_index']
            df_column_names = [key for key in column_indexes.keys()]
            df_column_index = [value for value in column_indexes.values()]
            dataset_path = os.path.join(self.raw_data_path, dataset_name + '.txt')

            all_data_path = os.path.join(processed_dataset_path, 'all')
            year_data_path = os.path.join(processed_dataset_path, 'year')
            output_path = os.path.join(processed_dataset_path, 'device')

            print('\n\n=======Preprocessing dataset {}=======\n'.format(dataset_name))

            df = pd.read_csv(dataset_path, sep='\t\t\t\t|\t\t\t|\t\t|\t|     |    |   |  | ', header=None,
                             engine='python')

            # 重命名列
            df = df[df_column_index].copy()
            df.rename(columns=dict(zip(df_column_index, df_column_names)), inplace=True)

            df['time'] = df['time'].apply(lambda x: x if '.' in x else x + '.000000')
            df['ts'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S.%f')
            df = df.drop(['date', 'time'], axis=1)

            ## date和time列合并，对毫秒位四舍五入
            # df['ts'] = self.round_timestamp(df['ts'])

            # 数据清洗
            data_washer = DataframeWasher(data_wash_dict)
            df = data_washer.wash(df)

            ## 查看异常的str数据
            # str_values = df[df['d_value'].apply(lambda x: isinstance(x, str))]['d_value']
            # special_value = list(str_values.unique())
            # print(special_value)

            d_name = df['d_name'].drop_duplicates().sort_values().tolist()  # 传感器名称 list

            # 保存env.csv
            if not os.path.exists(os.path.join(processed_dataset_path, 'env.csv')):
                self.process_env(dataset_name, d_name)

            # 保存all文件夹下的data.csv
            df.to_csv(os.path.join(all_data_path, 'data.csv'), index=False)

            if split_by == 'year':
                # 按年份分割数据，并保存在year文件夹下
                df['year'] = df['ts'].dt.year
                unique_years = df['year'].unique()
                for year in unique_years:
                    year_df = df[df['year'] == year].copy()
                    year_df.drop(columns=['year'], inplace=True)
                    year_df.to_csv(os.path.join(year_data_path, 'data_{}.csv'.format(year)), index=False)
                df.drop(columns=['year'], inplace=True)

                # 按d_name分割年份数据，并保存在device文件夹下
                filenames = os.listdir(year_data_path)
                for filename in filenames:
                    if filename.split('.')[-1] == 'csv':
                        data = pd.read_csv(os.path.join(year_data_path, filename))
                        for name in d_name:
                            data_i = data[data['d_name'] == name]
                            if not data_i.empty:
                                save_path_i = os.path.join(output_path,
                                                           '{}_{}.csv'.format(filename.split('.')[0], name))
                                data_i.to_csv(save_path_i, index=False)
            if split_by == 'd_name':
                # 直接按照d_name分割数据
                for name in d_name:
                    data_i = df[df['d_name'] == name]
                    if not data_i.empty:
                        save_path_i = os.path.join(output_path, 'data_{}.csv'.format(name))
                        data_i.to_csv(save_path_i, index=False)

            common_ts_path = os.path.join(processed_dataset_path, 'common_ts')
            self.process_csv_files(output_path, common_ts_path)

            print('=======Preprocessing Finished=======\n\n'.format(dataset_name))

    def process_env(self, dataset_name, d_name):
        concat_content = []

        room_trans_dict = {'level': 1, 'd_type': 'room', 'd_name': '/'}
        room_names = self.json_act(self.data_info_path, 'load')[dataset_name]['rooms']

        sensor_trans_dict = {
            'D0': {'level': 2, 'd_type': 'sensor', 'd_func': 'door', 'lt': 'cls'},
            'LS': {'level': 2, 'd_type': 'sensor', 'd_func': 'light', 'lt': 'reg'},
            'M0': {'level': 2, 'd_type': 'sensor', 'd_func': 'motion', 'lt': 'cls'},
            'MA': {'level': 2, 'd_type': 'sensor', 'd_func': 'ambient', 'lt': 'cls'},
            'T0': {'level': 2, 'd_type': 'sensor', 'd_func': 'temperature', 'lt': 'reg'},
            'T1': {'level': 2, 'd_type': 'sensor', 'd_func': 'temperature', 'lt': 'reg'},
            'L0': {'level': 2, 'd_type': 'sensor', 'd_func': 'lamp', 'lt': 'cls'},
            'Bu': {'level': 3, 'd_type': 'controller', 'd_func': 'button', 'lt': 'cls'}
        }

        df = pd.DataFrame(columns=['level', 'd_name', 'd_type', 'd_func', 'lt'])

        for name in d_name:
            matched_key = None
            for key in sensor_trans_dict.keys():
                if key in name:
                    matched_key = key
                    break
            if matched_key:
                new_row = {**sensor_trans_dict[matched_key], 'd_name': name}
                concat_content.append(new_row)
            else:
                concat_content.append({'d_name': name, 'level': 'unknown', 'd_type': 'unknown', 'd_func': 'unknown'})

        for room_name in room_names:
            concat_content.append({**room_trans_dict, 'd_func': room_name})
        concat_df = pd.DataFrame(concat_content)
        df = pd.concat([df, concat_df], ignore_index=True).sort_values(by=['level', 'd_name'])
        df.to_csv(os.path.join(self.processed_data_path, dataset_name, 'env.csv'), index=False)
        return

    def process_csv_files(self, file_path, output_path, date_col='ts'):
        # 确保输出目录存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 获取输入目录下所有csv文件列表
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

        # 存储所有数据框和它们对应的最小最大日期
        dfs = []
        min_max_dates = []

        # 读取csv文件并转换日期时间列为datetime类型
        for file in csv_files:
            df_path = os.path.join(file_path, file)
            df = pd.read_csv(df_path, parse_dates=[date_col])
            dfs.append(df)
            min_max_dates.append((df[date_col].min(), df[date_col].max()))

        # 找到所有文件中日期时间重叠的部分
        common_start = max(min_date for min_date, _ in min_max_dates)
        common_end = min(max_date for _, max_date in min_max_dates)

        # 如果没有重叠的时间段，则返回
        if common_start > common_end:
            print("No overlapping time periods found.")
            return

        # 仅保留重叠时间段内的数据，并保存到新的目录
        for i, file in enumerate(csv_files):
            f = self.get_interpolate_function(dfs[i])
            df_filtered = self.interpolate_data(f, common_start, common_end, freq=300)
            df_filtered.to_csv(os.path.join(output_path, file), index=False)


    @staticmethod
    def get_interpolate_function(dataset):
        data = dataset.copy()
        data['ts'] = data['ts'].astype(np.int64)
        x = data['ts'].values
        y = data['d_value'].values
        f = interp1d(x, y, kind='previous', bounds_error=False, fill_value=np.nan)
        return f

    @staticmethod
    def interpolate_data(f, start_ts, end_ts, freq=1, ttype='T'):
        start_ts = start_ts.floor(ttype)
        end_ts = end_ts.floor(ttype)

        start_ts = start_ts.value
        end_ts = end_ts.value

        s_to_ns = 1_000_000_000  # 每秒的纳秒数
        step_ns = freq * s_to_ns

        start_ts += step_ns
        end_ts += step_ns

        ts_new = np.arange(start_ts, end_ts, step_ns)

        y_new = f(ts_new)
        ts = pd.to_datetime(ts_new, unit='ns')
        df = pd.DataFrame({'ts': ts, 'd_value': y_new})

        return df

    @staticmethod
    def json_act(path, action, content=None):
        if content is None:
            content = {}
        if action == 'load':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        if action == 'dump':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
            return

    @staticmethod
    def round_timestamp(s):
        mic_second = s.dt.microsecond
        max_digits = mic_second.astype(str).str.len().max()
        rounded_seconds = (mic_second / (10 ** max_digits)).round()
        s = s.dt.floor('s')
        s += pd.to_timedelta(rounded_seconds, unit='s')
        return s


class DataframeWasher:
    def __init__(self, wash_dict):
        self.wash_dict = wash_dict
        self.wash_step_len = len(wash_dict)

    def wash(self, df):
        for step in range(self.wash_step_len):
            task_info = self.wash_dict[step]
            action_type = task_info['action_type']
            sub_action_type = task_info['sub_action_type']
            column_name = task_info['column_name']
            condition = task_info['condition']
            input_value = task_info['input_value']
            method_name = 'action_{}_{}'.format(action_type, sub_action_type)
            if 'c_' in sub_action_type:
                method_to_call = getattr(self, method_name, None)
                if callable(method_to_call):
                    df = method_to_call(df, column_name, input_value, condition)
                    # print('============={}============='.format(step))
                    # print(df)
            else:
                method_to_call = getattr(self, method_name, None)
                if callable(method_to_call):
                    df = method_to_call(df, column_name, input_value)
                    # print('============={}============='.format(step))
                    # print(df)
        return df

    def action_drop_equal(self, df, column_name, input_value: list):
        return df[~df[column_name].isin(input_value)]

    def action_drop_contain(self, df, column_name, input_value: list):
        pattern = '|'.join(map(re.escape, input_value))
        return df[~df[column_name].astype(str).str.contains(pattern)]

    def action_replace_equal(self, df, column_name, input_value: dict):
        df.loc[:, column_name] = df.loc[:, column_name].astype(str).replace(input_value)
        return df

    def action_replace_contain(self, df, column_name, input_value: dict):
        for k, v in input_value.items():
            df.loc[:, column_name] = df.loc[:, column_name].astype(str).str.replace(k, v)
        return df

    def action_replace_c_equal(self, df, column_name, input_value, condition: dict):
        c_column_name = condition['c_column_name']
        c_type = condition['c_type']
        c_value1 = condition['c_value1']
        # c_value2 = condition['c_value2']

        if c_type == 'contain':
            pattern = '|'.join(map(re.escape, c_value1))
            if isinstance(input_value, list):
                df.loc[df[c_column_name].astype(str).str.contains(pattern), column_name] = input_value
            if isinstance(input_value, dict):
                df.loc[df[c_column_name].astype(str).str.contains(pattern), column_name] = df.loc[
                    df[c_column_name].astype(str).str.contains(pattern), column_name].astype(str).replace(input_value)
            return df

        if c_type == 'equal':
            df.loc[df[c_column_name].isin(c_value1), column_name] = input_value
            return df

    def action_trans_value_type(self, df, column_name, input_value):
        df[column_name] = df[column_name].astype(input_value)
        return df

    def action_drop_cant_trans_to_num(self, df, column_name, input_value):
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df.dropna(subset=[column_name], inplace=True)
        return df


if __name__ == '__main__':
    pass
