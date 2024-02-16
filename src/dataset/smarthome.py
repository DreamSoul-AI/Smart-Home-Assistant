import bisect
import numpy as np
import os
import pandas as pd
import torch
from multiprocessing import Pool
from tqdm import tqdm
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class SmartHome(Dataset):
    data_name = 'SmartHome'

    def __init__(self, root, split, subset, seq_len=1200, hop_len=300, pred_len=300, min_len=1):
        self.root = os.path.expanduser(root)  # 替换root中的~为当前系统的用户目录***
        self.split = split
        self.transform = None
        self.subset = self.parse_subset(subset)
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pred_len = pred_len
        self.min_len = min_len
        self.process()
        self.other = {}

    def parse_subset(self, subset):
        parsed_subset = []
        subset_list = subset.split('-')
        for i in range(len(subset_list)):
            subset_i_list = subset_list[i].split('~')
            room_set = subset_i_list[0]
            if len(subset_i_list) == 1:
                year_set = []
                filenames = os.listdir(os.path.join(self.raw_folder, room_set))
                for filename in filenames:
                    if filename.startswith('data_') and filename.endswith('.csv'):
                        year_set_i = os.path.splitext(filename)[0].split('_')[1]
                        year_set.append(year_set_i)
            else:
                year_set = subset_i_list[1:]
            for j in range(len(year_set)):
                parsed_subset_i_j = '{}~{}'.format(room_set, year_set[j])
                parsed_subset.append(parsed_subset_i_j)
        return parsed_subset

    def configure(self, subset=None, seq_len=None, hop_len=None, min_len=None,
                  pred_len=None,
                  transform=None):  # 用于更新参数，子集（subset），序列长度（seq_len），跳跃长度（hop_len），最小长度（min_len）和预测长度（pred_len）
        if subset:
            self.subset = subset
        if seq_len:
            self.seq_len = seq_len
        if hop_len:
            self.hop_len = hop_len
        if min_len:
            self.min_len = min_len
        if pred_len:
            self.pred_len = pred_len
        self.transform = transform
        self.configuration = '{}_{}_{}_{}'.format(self.seq_len, self.hop_len, self.min_len, self.pred_len)
        return

    def __getitem__(self, index):  # 覆盖Dataset中__getitem__
        subset_index = bisect.bisect_left(self.length, index + 1)
        index_ = index if subset_index == 0 else index - self.length[subset_index - 1]
        subset = self.subset[subset_index]
        input = {k: self.data[subset][k][index_] for k in self.data[subset]}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        length = self.length[-1]
        return length

    @property  # 将方法转化为属性，访问该属性时不再需要加括号（ 如processed_folder() ）
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property  # 同上
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        self.configure()
        if not check_exists(self.raw_folder):  # 当root\raw文件夹不存在时，抛出报错
            self.download()
        for subset in self.subset:
            room_set, year_set = subset.split('~')
            data_path = os.path.join(self.processed_folder, room_set, year_set, self.configuration)  # 数据子集路径
            print(f'data_path: {data_path}')
            if not check_exists(data_path):  # 如果子集路径不存在，建立子集路径
                makedir_exist_ok(data_path)
                train_set, test_set = self.make_data(room_set, year_set)
                save(train_set, os.path.join(data_path, 'train'))
                save(test_set, os.path.join(data_path, 'test'))
        self.data, self.meta = self.load_data()
        return

    def download(self):  # 抛出报错NotImplementedError
        raise NotImplementedError

    def load_data(self):
        data, meta = {}, {}
        self.length = []
        length = 0
        for subset in self.subset:
            room_set, year_set = subset.split('~')
            data[subset], meta[subset] = load(os.path.join(self.processed_folder, room_set, year_set,
                                                           self.configuration, self.split))
            length += len(data[subset]['data'])
            self.length.append(length)
        return data, meta

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(),
                                                                     self.root, self.split)
        return fmt_str

    @property
    def normalization(self):
        # Constants
        days_per_year = 366
        hours_per_day = 24
        minutes_per_hour = 60
        seconds_per_minute = 60

        # Calculation
        seconds_per_year = days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute
        return seconds_per_year

    def make_data(self, room_set, year_set):
        print('----------------make_data ({}, {})-------------------'.format(room_set, year_set))
        data = pd.read_csv(os.path.join(self.raw_folder, room_set, 'data_{}.csv'.format(year_set)), delimiter=',')
        subset_ratio = 0.1  # make it small for test
        split_index = int(subset_ratio * len(data))
        data = data[:split_index]
        env_path = os.path.join(self.raw_folder, room_set, 'env.csv')
        if os.path.exists(env_path):
            env = pd.read_csv(env_path, delimiter=',')
        else:
            env = None
        data = data[['ts', 'd_name', 'd_type', 'd_func', 'd_value']]

        data['ts'] = pd.to_datetime(data['ts'])

        data.loc[data['d_func'] == 'lamp', 'd_value'] = data.loc[data['d_func'] == 'lamp', 'd_value'] / 100.0
        data.loc[data['d_func'] == 'light', 'd_value'] = data.loc[data['d_func'] == 'light', 'd_value'] / 100.0
        data.loc[data['d_func'] == 'temperature', 'd_value'] = data.loc[
                                                                   data['d_func'] == 'temperature', 'd_value'] / 50.0

        split_ratio = 0.9  # need to change this
        split_index = int(split_ratio * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]

        print(f'-----------len(train_data): {len(train_data)}')
        print(f'-----------len(test_data): {len(test_data)}')

        unique_count = len(train_data['d_type'].unique())
        print('Number of unique d_type in train_data: {}'.format(unique_count))
        unique_count = len(test_data['d_type'].unique())
        print('Number of unique d_type in test_data: {}'.format(unique_count))

        train_data, train_start_times = self.batchify(train_data)
        test_data, test_start_times = self.batchify(test_data)

        print(f'-----------len(train_start_times): {len(train_start_times)}')
        print(f'-----------len(test_start_times): {len(test_start_times)}')

        train_meta = (train_start_times, env)
        test_meta = (test_start_times, env)
        return (train_data, train_meta), (test_data, test_meta)

    def process_chunk(self, chunk_args):
        chunk, seq_len, min_len, pred_len, dataset = chunk_args  # 读取元组数据
        controller_data = dataset[dataset['d_type'] == 'controller']
        data = {'data': [], 'target': [], 't_start': []}
        for t_start in tqdm(chunk, desc="Processing chunk", leave=False):  # 遍历chunk中的start_times
            t_end = t_start + seq_len
            data_i = dataset[
                (dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]  # 以t_start为起点，t_end为终点，在dataset中获取data_i
            if len(data_i) < min_len:  # data_i小于最小长度时结束处理
                continue
            t_pred_end = t_end + pred_len
            target_i = controller_data[(controller_data['ts'] < t_pred_end) & (controller_data[
                                                                                   'ts'] >= t_end)]  # 以t_end为起点，t_pred_end_j为终点，在controller_data中获取target_i_j，即在预测范围内的controller_data数据
            data['t_start'].append(t_start)
            data['data'].append(data_i)  # 每个序列数据
            data['target'].append(target_i)  # 每个序列后预测范围内的controller_data数据
        return data

    def batchify(self, dataset):
        seq_len = pd.Timedelta(seconds=self.seq_len)  # 序列长度
        hop_len = pd.Timedelta(seconds=self.hop_len)  # 跳跃长度
        min_len = self.min_len  # 最小长度
        pred_len = pd.Timedelta(seconds=self.pred_len)  # 预测长度列表，可包含多个预测长度
        start_times = pd.date_range(start=dataset.iloc[0]['ts'], end=dataset.iloc[-1]['ts'] - seq_len,
                                    freq=hop_len)  # 以第一个时间为起点，以300秒为间隔，获取开始时间列表

        # print(dataset['d_type'].unique())
         # 控制器数据

        # Split start_times into chunks
        n_chunks = 4  # Number of chunks, can be adjusted
        chunks = np.array_split(start_times, n_chunks)  # 将start_timies平均切割为4份
        args = [(chunk, seq_len, min_len, pred_len, dataset) for chunk in
                chunks]  # 将每份数据chunk、序列长度、最小长度、预测长度、全部控制器数据、全部数据组成元组，将各元组以列表形式保存到args中

        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_chunk, args),
                                total=len(chunks)))  # 将args传递给self.process_chunk函数在一个池中的独立进程上并行处理，处理结果保存到列表results中

        # Combine results
        data = {'data': [], 'target': [], 't_start': []}
        for result in results:
            data['data'].extend(result['data'])
            data['target'].extend(result['target'])
            data['t_start'].extend(result['t_start'])
        return data, start_times

    # def batchify(self, dataset):
    #     seq_len = pd.Timedelta(seconds=self.seq_len)
    #     hop_len = pd.Timedelta(seconds=self.hop_len)
    #     min_len = self.min_len
    #     pred_len = pd.Timedelta(seconds=self.pred_len)
    #     start_times = pd.date_range(start=dataset.iloc[0]['ts'],
    #                                 end=dataset.iloc[-1]['ts'] - seq_len, freq=hop_len)
    #     controller_data = dataset[dataset['d_type'] == 'controller']
    #     data = {'data': [], 'target': [], 't_start': []}
    #     for i in tqdm(range(len(start_times))):
    #         t_start = start_times[i]
    #         t_end = t_start + seq_len
    #         data_i = dataset[(dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]
    #         if len(data_i) < min_len:
    #             continue
    #         t_pred_end = t_end + pred_len
    #         target_i = controller_data[(controller_data['ts'] < t_pred_end) & (controller_data['ts'] >= t_end)]
    #         data['t_start'].append(t_start)
    #         data['data'].append(data_i)
    #         data['target'].append(target_i)
    #     return data, start_times
