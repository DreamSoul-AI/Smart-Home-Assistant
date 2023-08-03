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

    def __getitem__(self, index):
        data, target = torch.tensor(self.data[index]), torch.tensor(self.target[index])
        input = {'data': data, 'target': target}
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**input, **other}
        return input

    def __len__(self):
        return len(self.data)

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
        split_ratio = 0.8
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        # print(train_data['d_type'].value_counts())
        # print(test_data['d_type'].value_counts())
        return train_data, test_data, env
