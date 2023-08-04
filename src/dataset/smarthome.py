import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class SmartHome(Dataset):
    data_name = 'SmartHome'
    supported_subsets = ['hh103']

    def __init__(self, root, split, subset):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = None
        self.subset = subset
        if not check_exists(self.processed_folder):
            self.process()
        self.data, self.meta = self.load_data()
        self.other = {}
        self.length = []
        length = 0
        for k in self.subset:
            length += len(self.data[k])
            self.length.append(length)
        self.pred_len = None
        self.hop_len = None
        self.seq_len = None

    def configure(self, seq_len=None, hop_len=None, pred_len=None):
        if seq_len:
            self.seq_len = pd.Timedelta(seconds=seq_len)
        if hop_len:
            self.hop_len = pd.Timedelta(seconds=hop_len)
        if pred_len:
            self.pred_len = []
            for i in range(len(pred_len)):
                self.pred_len.append(pd.Timedelta(seconds=pred_len[i]))
        self.start_times = {}
        for k in self.subset:
            self.start_times[k] = pd.date_range(start=self.data[k].iloc[0]['ts'],
                                                end=self.data[k].iloc[-1]['ts'] - self.seq_len, freq=self.hop_len)
        return

    def __getitem__(self, index):
        subset_index = np.searchsorted(self.length, index, side='right') - 1
        subset = self.subset[subset_index]
        t_start = self.start_times[subset][index]
        t_end = t_start + self.seq_len
        data = self.data[subset][(self.data[subset]['ts'] >= t_start) & (self.data[subset]['ts'] < t_end)]
        target = []
        detect = []
        for i in range(len(self.pred_len)):
            t_pred_end_i = t_start + self.pred_len[i]
            target_i = self.data[subset][(self.data[subset]['ts'] >= t_start) &
                                         (self.data[subset]['ts'] < t_pred_end_i) &
                                         (self.data[subset]['d_type'] == 'controller')]
            detect_i = 1 if not target_i.empty else 0
            target.append(target_i)
            detect.append(detect_i)
        input = {'data': data, 'target': target, 'detect': detect}
        return input

    def __len__(self):
        length = 0
        for k in self.subset:
            length += len(self.start_times[k])
        return length

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        for subset in self.supported_subsets:
            train_set, test_set, meta = self.make_data(subset)
            save(train_set, os.path.join(self.processed_folder, subset, 'train'))
            save(test_set, os.path.join(self.processed_folder, subset, 'test'))
            save(meta, os.path.join(self.processed_folder, subset, 'meta'))
        return

    def download(self):
        raise NotImplementedError

    def load_data(self):
        data, target, meta = {}, {}, {}
        for subset in self.subset:
            data[subset] = load(os.path.join(self.processed_folder, subset, self.split))
            meta[subset] = load(os.path.join(self.processed_folder, subset, 'meta'))
        return data, meta

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(),
                                                                     self.root, self.split)
        return fmt_str

    def make_data(self, subset):
        data = pd.read_csv(os.path.join(self.raw_folder, subset, 'data.csv'), delimiter=',')
        env = pd.read_csv(os.path.join(self.raw_folder, subset, 'env.csv'), delimiter=',')
        data = data[['e_datetime', 'did', 'type', 'func', 'd_value']]
        data.rename(columns={'e_datetime': 'ts', 'did': 'd_name', 'type': 'd_type', 'func': 'd_func'}, inplace=True)
        data['ts'] = pd.to_datetime(data['ts'])
        split_ratio = 0.9
        split_index = int(split_ratio * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        meta = env
        return train_data, test_data, meta
