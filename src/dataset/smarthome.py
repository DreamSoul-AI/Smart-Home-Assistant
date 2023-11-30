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

    def __init__(self, root, split, subset, seq_len=1200, hop_len=300, pred_len=(300,), min_len=2):
        self.root = os.path.expanduser(root)  # 替换root中的~为当前系统的用户目录***
        self.split = split
        self.transform = None
        self.subset = subset
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pred_len = pred_len
        self.min_len = min_len
        self.process()
        self.other = {}

    def configure(self, subset=None, seq_len=None, hop_len=None, min_len=None,
                  pred_len=None):  # 用于更新参数，子集（subset），序列长度（seq_len），跳跃长度（hop_len），最小长度（min_len）和预测长度（pred_len）
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
        self.configuration = '{}_{}_{}_{}'.format(self.seq_len, self.hop_len, self.min_len,
                                                  '-'.join(map(str, self.pred_len)))
        return

    def __getitem__(self, index):  # 覆盖Dataset中__getitem__
        subset_index = bisect.bisect_left(self.length, index + 1)
        index_ = index if subset_index == 0 else index - self.length[subset_index - 1]
        subset = self.subset[subset_index]
        input = {k: self.data[subset][k][index_] for k in self.data[subset]}
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
            data_path = os.path.join(self.processed_folder, subset, self.configuration)  # 数据子集路径
            print(f'data_path: {data_path}')
            if not check_exists(data_path):  # 如果子集路径不存在，建立子集路径
                makedir_exist_ok(data_path)
                train_set, test_set = self.make_data(subset)
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
            data[subset], meta[subset] = load(os.path.join(self.processed_folder, subset, self.configuration,
                                                           self.split))
            length += len(data[subset]['data'])
            self.length.append(length)
        return data, meta

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(),
                                                                     self.root, self.split)
        return fmt_str

    def make_data(self, subset):
        print('----------------make_data-------------------')
        data = pd.read_csv(os.path.join(self.raw_folder, subset, 'data.csv'), delimiter=',')
        subset_ratio = 0.01 # make it small for test
        split_index = int(subset_ratio * len(data))
        data = data[:split_index]
        env = pd.read_csv(os.path.join(self.raw_folder, subset, 'env.csv'), delimiter=',')
        data = data[['ts', 'd_name', 'd_type', 'd_func', 'd_value']]

        data['ts'] = pd.to_datetime(data['ts'])

        # Normalization
        time_start = data['ts'].iloc[0]
        time_end = data['ts'].iloc[-1]
        time_range = time_end - time_start
        data['ts_normalized'] = (data['ts'] - time_start) / time_range

        data.loc[data['d_func'] == 'lamp', 'd_value'] = data.loc[data['d_func'] == 'lamp', 'd_value'] / 100.0
        data.loc[data['d_func'] == 'light', 'd_value'] = data.loc[data['d_func'] == 'light', 'd_value'] / 100.0
        data.loc[data['d_func'] == 'temperature', 'd_value'] = data.loc[
                                                                   data['d_func'] == 'temperature', 'd_value'] / 50.0

        # print('Normalization test max:{},min:{}'.format(data['d_value'].max(), data['d_value'].min()))
        # print('Normalization test max:{},min:{}'.format(data['ts_normalized'].max(), data['ts_normalized'].min()))

        split_ratio = 0.9
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
        chunk, seq_len, min_len, pred_len, controller_data, dataset = chunk_args  # 读取元组数据
        data = {'data': [], 'target': [], 'detect': []}
        for t_start in tqdm(chunk, desc="Processing chunk", leave=False):  # 遍历chunk中的start_times
            t_end = t_start + seq_len
            data_i = dataset[
                (dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]  # 以t_start为起点，t_end为终点，在dataset中获取data_i
            if len(data_i) < min_len:  # data_i小于最小长度时结束处理
                continue
            target_i = []
            detect_i = []
            for j in range(len(pred_len)):  # 遍历所有预测长度
                pred_len_j = pred_len[j]  # 给pred_len_j赋值
                t_pred_end_j = t_end + pred_len_j
                target_i_j = controller_data[
                    (controller_data['ts'] < t_pred_end_j) & (controller_data[
                                                                  'ts'] >= t_end)]  # 以t_end为起点，t_pred_end_j为终点，在controller_data中获取target_i_j，即在预测范围内的controller_data数据
                detect_i_j = 1 if not target_i_j.empty else 0  # 当target_i_j不为空，detect_i_j = 1，反之为0
                target_i.append(target_i_j)
                detect_i.append(detect_i_j)
            data['data'].append(data_i)  # 每个序列数据
            data['target'].append(target_i)  # 每个序列后预测范围内的controller_data数据
            data['detect'].append(detect_i)  # 每个序列后预测范围内是否有controller_data
        return data

    def batchify(self, dataset):
        seq_len = pd.Timedelta(seconds=self.seq_len)  # 序列长度
        hop_len = pd.Timedelta(seconds=self.hop_len)  # 跳跃长度
        min_len = self.min_len  # 最小长度
        pred_len = [pd.Timedelta(seconds=p) for p in self.pred_len]  # 预测长度列表，可包含多个预测长度
        start_times = pd.date_range(start=dataset.iloc[0]['ts'], end=dataset.iloc[-1]['ts'] - seq_len,
                                    freq=hop_len)  # 以第一个时间为起点，以300秒为间隔，获取开始时间列表
        controller_data = dataset[dataset['d_type'] == 'controller']  # 控制器数据

        # Split start_times into chunks
        n_chunks = 4  # Number of chunks, can be adjusted
        chunks = np.array_split(start_times, n_chunks)  # 将start_timies平均切割为4份
        args = [(chunk, seq_len, min_len, pred_len, controller_data, dataset) for chunk in
                chunks]  # 将每份数据chunk、序列长度、最小长度、预测长度、全部控制器数据、全部数据组成元组，将各元组以列表形式保存到args中

        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_chunk, args),
                                total=len(chunks)))  # 将args传递给self.process_chunk函数在一个池中的独立进程上并行处理，处理结果保存到列表results中

        # Combine results
        data = {'data': [], 'target': [], 'detect': []}
        for result in results:
            data['data'].extend(result['data'])
            data['target'].extend(result['target'])
            data['detect'].extend(result['detect'])

        return data, start_times



    # def batchify(self, dataset):
    #     from tqdm import tqdm
    #     seq_len = pd.Timedelta(seconds=self.seq_len)
    #     hop_len = pd.Timedelta(seconds=self.hop_len)
    #     min_len = self.min_len
    #     pred_len = []
    #     for i in range(len(self.pred_len)):
    #         pred_len.append(pd.Timedelta(seconds=self.pred_len[i]))
    #     start_times = pd.date_range(start=dataset.iloc[0]['ts'],
    #                                 end=dataset.iloc[-1]['ts'] - seq_len, freq=hop_len)
    #     controller_data = dataset[dataset['d_type'] == 'controller']
    #     data = {'data': [], 'target': [], 'detect': []}
    #     for i in tqdm(range(len(start_times))):
    #         t_start = start_times[i]
    #         t_end = t_start + seq_len
    #         data_i = dataset[(dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]
    #         if len(data_i) < min_len:
    #                 break
    #         controller_data_i = dataset[(dataset['ts'] >= t_start) & (dataset['d_type'] == 'controller')]
    #         target_i = []
    #         detect_i = []
    #         for j in range(len(pred_len)):
    #             pred_len_j = pred_len[j]
    #             t_pred_end_j = t_end + pred_len_j
    #             target_i_j = controller_data[
    #                     (controller_data['ts'] < t_pred_end_j) & (controller_data['ts'] >= t_end)]
    #             detect_i_j = 1 if not target_i_j.empty else 0
    #             target_i.append(target_i_j)
    #             detect_i.append(detect_i_j)
    #         data['data'].append(data_i)
    #         data['target'].append(target_i)
    #         data['detect'].append(detect_i)
    #     return data, start_times

    # seq_len = 1200, hop_len = 300, pred_len = (300,), min_len = 2