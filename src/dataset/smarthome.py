import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load


class SmartHome(Dataset):
    data_name = 'SmartHome'
    supported_subsets = ['hh101']

    def __init__(self, root, split, subset):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = None
        self.subset = subset
        if not check_exists(self.processed_folder):
            self.process()
        self.data, self.target, self.meta = self.load_data()
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
            data[subset], target[subset] = load(os.path.join(self.processed_folder, subset, self.split))
            meta[subset] = load(os.path.join(self.processed_folder, subset, 'meta'))
        return data, target, meta

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(),
                                                                     self.root, self.split)
        return fmt_str

    def make_data(self, subset):
        data = pd.read_csv(os.path.join(self.raw_folder, subset, 'data.csv'), header=None, delimiter=',')
        env = pd.read_csv(os.path.join(self.raw_folder, subset, 'env.csv'), header=None, delimiter=',')
        print(data)
        print(env)
        exit()
        train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes = list(map(str, list(range(10))))
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        target_size = len(classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)
