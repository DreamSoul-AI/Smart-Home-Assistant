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
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = None
        self.subset = subset
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pred_len = pred_len
        self.min_len = min_len
        self.process()
        self.other = {}

    def configure(self, subset=None, seq_len=None, hop_len=None, min_len=None, pred_len=None):
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

    def __getitem__(self, index):
        subset_index = bisect.bisect_left(self.length, index + 1)
        index_ = index if subset_index == 0 else index - self.length[subset_index - 1]
        subset = self.subset[subset_index]
        input = {k: self.data[subset][k][index_] for k in self.data[subset]}
        return input

    def __len__(self):
        length = self.length[-1]
        return length

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        self.configure()
        if not check_exists(self.raw_folder):
            self.download()
        for subset in self.subset:
            data_path = os.path.join(self.processed_folder, subset, self.configuration)
            if not check_exists(data_path):
                makedir_exist_ok(data_path)
                train_set, test_set = self.make_data(subset)
                save(train_set, os.path.join(data_path, 'train'))
                save(test_set, os.path.join(data_path, 'test'))
        self.data, self.meta = self.load_data()
        return

    def download(self):
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
        data = pd.read_csv(os.path.join(self.raw_folder, subset, 'data.csv'), delimiter=',')
        subset_ratio = 0.01
        split_index = int(subset_ratio * len(data))
        data = data[:split_index]
        env = pd.read_csv(os.path.join(self.raw_folder, subset, 'env.csv'), delimiter=',')
        data = data[['e_datetime', 'did', 'type', 'func', 'd_value']]
        data.rename(columns={'e_datetime': 'ts', 'did': 'd_name', 'type': 'd_type', 'func': 'd_func'}, inplace=True)
        data['ts'] = pd.to_datetime(data['ts'])
        split_ratio = 0.9
        split_index = int(split_ratio * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        unique_count = train_data['d_type'].value_counts()
        print(unique_count)
        unique_count = test_data['d_type'].value_counts()
        print(unique_count)
        train_data, train_start_times = self.batchify(train_data)
        test_data, test_start_times = self.batchify(test_data)
        train_meta = (train_start_times, env)
        test_meta = (test_start_times, env)
        return (train_data, train_meta), (test_data, test_meta)

    # def batchify(self, dataset):
    #     from tqdm import tqdm
    #     seq_len = pd.Timedelta(seconds=self.seq_len)
    #     hop_len = pd.Timedelta(seconds=self.hop_len)
    #     pred_len = []
    #     for i in range(len(self.pred_len)):
    #         pred_len.append(pd.Timedelta(seconds=self.pred_len[i]))
    #     start_times = pd.date_range(start=dataset.iloc[0]['ts'],
    #                                 end=dataset.iloc[-1]['ts'] - seq_len, freq=hop_len)
    #     data = {'data': [], 'target': [], 'detect': []}
    #     for i in tqdm(range(len(start_times))):
    #         t_start = start_times[i]
    #         t_end = t_start + seq_len
    #         data_i = dataset[(dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]
    #         controller_data_i = dataset[(dataset['ts'] >= t_start) & (dataset['d_type'] == 'controller')]
    #         target_i = []
    #         detect_i = []
    #         for j in range(len(pred_len)):
    #             pred_len_j = pred_len[j]
    #             t_pred_end_j = t_start + pred_len_j
    #             target_i_j = controller_data_i[controller_data_i['ts'] < t_pred_end_j]
    #             detect_i_j = 1 if not target_i_j.empty else 0
    #             target_i.append(target_i_j)
    #             detect_i.append(detect_i_j)
    #         data['data'].append(data_i)
    #         data['target'].append(target_i)
    #         data['detect'].append(detect_i)
    #     return data, start_times

    def process_chunk(self, chunk_args):
        chunk, seq_len, min_len, pred_len, controller_data, dataset = chunk_args
        data = {'data': [], 'target': [], 'detect': []}
        for t_start in tqdm(chunk, desc="Processing chunk", leave=False):
            t_end = t_start + seq_len
            data_i = dataset[(dataset['ts'] >= t_start) & (dataset['ts'] < t_end)]
            if len(data_i) < min_len:
                break
            target_i = []
            detect_i = []
            for j in range(len(pred_len)):
                pred_len_j = pred_len[j]
                t_pred_end_j = t_end + pred_len_j
                target_i_j = controller_data[
                    (controller_data['ts'] < t_pred_end_j) & (controller_data['ts'] >= t_end)]
                detect_i_j = 1 if not target_i_j.empty else 0
                target_i.append(target_i_j)
                detect_i.append(detect_i_j)
            data['data'].append(data_i)
            data['target'].append(target_i)
            data['detect'].append(detect_i)
        return data

    def batchify(self, dataset):
        seq_len = pd.Timedelta(seconds=self.seq_len)
        hop_len = pd.Timedelta(seconds=self.hop_len)
        min_len = self.min_len
        pred_len = [pd.Timedelta(seconds=p) for p in self.pred_len]
        start_times = pd.date_range(start=dataset.iloc[0]['ts'], end=dataset.iloc[-1]['ts'] - seq_len, freq=hop_len)
        controller_data = dataset[dataset['d_type'] == 'controller']

        # Split start_times into chunks
        n_chunks = 4  # Number of chunks, can be adjusted
        chunks = np.array_split(start_times, n_chunks)
        args = [(chunk, seq_len, min_len, pred_len, controller_data, dataset) for chunk in chunks]

        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_chunk, args), total=len(chunks)))

        # Combine results
        data = {'data': [], 'target': [], 'detect': []}
        for result in results:
            data['data'].extend(result['data'])
            data['target'].extend(result['target'])
            data['detect'].extend(result['detect'])

        # import matplotlib.pyplot as plt
        #
        # Assuming data['data'] is a list of lists, and you want to plot the lengths of the inner lists
        # from collections import Counter
        #
        # # Assuming data['data'] is a list of lists, and you want to print the counts of unique lengths of the inner lists
        # lengths = [len(element) for element in data['data']]
        # unique_length_counts = Counter(lengths)
        # print("Unique Lengths and Their Counts:")
        # for length, count in unique_length_counts.items():
        #     print(f"Length: {length}, Count: {count}")
        #
        # plt.hist(lengths, bins='auto')  # You can specify the number of bins or leave it as 'auto'
        # plt.title('Histogram of Lengths')
        # plt.xlabel('Length')
        # plt.ylabel('Frequency')
        # plt.show()
        return data, start_times