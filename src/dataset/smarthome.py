import bisect
import numpy as np
import os
import pandas as pd
import torch
from multiprocessing import Pool
from tqdm import tqdm
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from scipy.interpolate import interp1d


class SmartHome(Dataset):
    data_name = 'SmartHome'

    def __init__(self, root, split, subset, seq_len=1200, hop_len=300, min_len=1, reprocess=False):
        self.root = os.path.expanduser(root)  # 替换root中的~为当前系统的用户目录***
        self.split = split
        self.transform = None
        self.subset = self.parse_subset(subset)

        self.seq_len = seq_len
        self.hop_len = hop_len
        self.min_len = min_len
        self.reprocess = reprocess
        self.process()
        self.other = {}

    def parse_subset(self, subset):
        parsed_subset = []
        preprocessed_folder = self.preprocessed_folder
        subset_list = subset.split('-')
        data_info_set = ['all', 'all', 'all']
        for i in range(len(subset_list)):
            data_info_set[i] = subset_list[i]
        room_info, year_info, device_info = data_info_set
        if room_info == 'all':
            room_set = os.listdir(preprocessed_folder)
        else:
            room_set = room_info.split('~')

        for room in room_set:
            if year_info == 'all':
                year_set = []
                filenames = os.listdir(os.path.join(preprocessed_folder, room, 'year'))
                print(filenames)
                for filename in filenames:
                    if filename.startswith('data_') and filename.endswith('.csv'):
                        year_set_i = os.path.splitext(filename)[0].split('_')[1]
                        year_set.append(year_set_i)
            else:
                year_set = year_info.split('~')

            if device_info == 'all':
                device_set = []
                filenames = os.listdir(os.path.join(preprocessed_folder, room, 'year', 'device'))
                for filename in filenames:
                    if filename.startswith('data_') and filename.endswith('.csv'):
                        device_set_i = os.path.splitext(filename)[0].split('_')[2]
                        device_set.append(device_set_i)
            else:
                device_set = device_info.split('~')

            for year in year_set:
                for device in device_set:
                    parsed_subset.append('{}~{}~{}'.format(room, year, device))

        seen = set()
        result = [x for i, x in enumerate(parsed_subset) if x not in seen and not seen.add(x)]
        return result

    def configure(self, subset=None, seq_len=None, hop_len=None, min_len=None,
                  transform=None):  # 用于更新参数，子集（subset），序列长度（seq_len），跳跃长度（hop_len），最小长度（min_len）
        if subset:
            self.subset = subset
        if seq_len:
            self.seq_len = seq_len
        if hop_len:
            self.hop_len = hop_len
        if min_len:
            self.min_len = min_len
        self.transform = transform
        self.configuration = '{}_{}_{}'.format(self.seq_len, self.hop_len, self.min_len)
        return

    def __getitem__(self, index):  # 覆盖Dataset中__getitem__
        subset_index = bisect.bisect_left(self.length, index + 1)
        index_ = index if subset_index == 0 else index - self.length[subset_index - 1]
        subset = self.subset[subset_index]
        input = {'data': self.data[subset]['data'][index_], 't_start': self.data[subset]['t_start'][index_]}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        if hasattr(self, 'length') and self.length:
            length = self.length[-1]
        else:
            length = 0
        return length

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def preprocessed_folder(self):
        return os.path.join(self.root, 'preprocessed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')


    def process(self):
        self.configure()
        # if not check_exists(self.preprocessed_folder):  # 当root\processed文件夹不存在时，抛出报错
        #     self.download()
        data_exists = True
        for subset in self.subset:
            room_set, year_set, device_set = subset.split('~')
            data_path = os.path.join(self.processed_folder, room_set, year_set, device_set, self.configuration)  # 数据子集路径
            print(f'data_path: {data_path}')

            if not check_exists(data_path) or self.reprocess:  # 如果子集路径不存在，建立子集路径
                makedir_exist_ok(data_path)
                self.make_data(room_set, year_set, device_set)
                data_exists = False  # 标记数据是新创建的
            else:
                # 检查具体的数据文件是否存在
                data_file = os.path.join(data_path, self.split)
                if not os.path.exists(data_file):
                    self.make_data(room_set, year_set, device_set)
                    data_exists = False

                # save(train_set, os.path.join(data_path, 'train'))
                # save(test_set, os.path.join(data_path, 'test'))
                # save(train_set, os.path.join(data_path, 'meta'))
        # self.data, self.meta = self.load_data()
        # 只在所有数据文件都存在时才加载数据
        if data_exists:
            self.data = self.load_data()
        else:
            # 如果是新创建的数据，初始化空数据结构
            self.data = {}
            self.length = [0]
            # 数据创建完成后，尝试重新加载
            try:
                self.data = self.load_data()
            except FileNotFoundError:
                # 如果还是找不到文件，保持空数据结构
                pass
        return

    def download(self):  # 抛出报错NotImplementedError
        raise NotImplementedError

    def load_data(self):
        data, meta = {}, {}
        self.length = []
        length = 0
        for subset in self.subset:
            room_set, year_set, device_set = subset.split('~')
            data[subset] = load(os.path.join(self.processed_folder, room_set, year_set, device_set,
                                             self.configuration, self.split))
            # meta[subset] = load(os.path.join(self.processed_folder, room_set, year_set, device_set,
            #                                  self.configuration, 'meta'))
            length += len(data[subset])
            self.length.append(length)
        # return data, meta
        return data

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(self.__class__.__name__, self.__len__(),
                                                                     self.root, self.split)
        return fmt_str

    def make_data(self, room_set, year_set, device_set):
        print(f'----------------make_data for: {room_set}-{year_set}-{device_set}-------------------')
        
        # 解析 'hh105-2015' 这样的子集名称
        parts = [room_set, year_set, device_set]
        if len(parts) < 2:
            print(f"Error: subset_name '{room_set}-{year_set}-{device_set}' is not in the expected format 'room-year-device'.")
            return {}
        
        room_set = parts[0]
        year_set = parts[1]
        
        # 从全局配置获取设备名称 ('all' 或 'LS005' 等)
        device_spec = parts[2]
        
        raw_data_path = os.path.join(self.raw_folder, room_set)
        
        devices_to_process = []
        if device_spec.lower() == 'all':
            print(f"Loading ALL devices for {room_set} - {year_set}...")
            # 假设每个设备的数据都在一个单独的文件夹里，并且文件夹名就是设备名
            # 我们需要扫描 raw_data_path/2015/ 类似这样的目录
            year_path = os.path.join(raw_data_path, year_set)
            if os.path.isdir(year_path):
                 devices_to_process = [d for d in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, d))]
            else:
                print(f"Warning: Directory for year {year_set} not found at {year_path}")
        else:
            print(f"Loading specific device: {device_spec} for {room_set} - {year_set}...")
            devices_to_process.append(device_spec)
            
        print(f"Devices to be processed: {devices_to_process}")

        all_device_data = []
        for device in devices_to_process:
            file_path = os.path.join(raw_data_path, year_set, device, 'data.csv')
            if os.path.exists(file_path):
                print(f"  - Loading from: {file_path}")
                df = pd.read_csv(file_path)
                all_device_data.append(df)
            else:
                print(f"  - Warning: File not found, skipping: {file_path}")

        if not all_device_data:
            print("Error: No data loaded. Please check paths and configurations.")
            return {}
            
        # 合并所有设备的数据
        data = pd.concat(all_device_data, ignore_index=True)
        data = data.sort_values(by='ts').reset_index(drop=True)

        data = data[['ts', 'd_name', 'd_value']]
        data['ts'] = pd.to_datetime(data['ts'])

        print(f'-----------Total data points from all devices: {len(data)}')

        # Batchify 和保存的逻辑需要针对合并后的数据进行
        # 注意：这里的 'device_set' 参数需要一个代表性的名称，比如 'all'
        data = self.batchify(data, room_set, year_set, device_spec)
        
        data_path = os.path.join(self.processed_folder, room_set, year_set, device_spec, self.configuration)
        save(data, os.path.join(data_path, self.split))
        
        return data

    def process_chunk(self, chunk_args):
        chunk, dataset, freq = chunk_args  # 读取元组数据
        data = {'data': [], 't_start': []}
        for t_start in tqdm(chunk, desc="Processing chunk", leave=False):  # 遍历chunk中的start_times
            t_end = t_start + self.seq_len // freq
            data_i = dataset[t_start:t_end]  # 以t_start为起点，t_end为终点，在dataset中获取data_i
            data['data'].append(data_i['d_value'].values)  # 每个序列数据
            data['t_start'].append(t_start)
        return data

    def batchify(self, dataset, room_set, year_set, device_set):
        interpolate_function = self.get_interpolate_function(dataset)
        dataset, freq = self.interpolate_data(dataset, interpolate_function, type='h', freq=300)
        # dataset.to_csv('./data/{}_{}_{}.csv'.format(room_set, year_set, device_set), index=False)  # 输出到TIMESNET的数据，之后手动放入TIMESNET处理
        # exit()
        # print(len(dataset))
        # s_r = 0.01
        # s_data_len = int(s_r * len(dataset))
        # dataset = dataset[:s_data_len]
        start_index = np.arange(0, len(dataset) - self.seq_len // freq, self.hop_len // freq)
        # print(len(dataset))
        # print(self.seq_len / freq)
        # print(self.hop_len / freq)
        # print(start_index)
        # exit()

        n_chunks = 8  # Number of chunks, can be adjusted
        chunks = np.array_split(start_index, n_chunks)  # 将start_timies平均切割为8份
        args = [(chunk, dataset, freq) for chunk in chunks]  # 将每份数据chunk、序列长度、最小长度、预测长度、全部控制器数据、全部数据组成元组，将各元组以列表形式保存到args中
        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_chunk, args),
                                total=len(chunks)))  # 将args传递给self.process_chunk函数在一个池中的独立进程上并行处理，处理结果保存到列表results中
        # Combine results
        data = {'data': [], 't_start': []}
        for result in results:
            data['data'].extend(result['data'])
            data['t_start'].extend(result['t_start'])

        return data

    @staticmethod
    def get_interpolate_function(dataset):
        data = dataset.copy()
        data['ts'] = data['ts'].astype(np.int64)
        x = data['ts'].values
        y = data['d_value'].values
        f = interp1d(x, y, kind='previous', bounds_error=False, fill_value=np.nan)
        return f

    @staticmethod
    def interpolate_data(df, f, type='s', freq=1):
        addon = 1
        df_ts_max = df['ts'].astype(np.int64).values.max()

        df_i = df['ts'].dt.floor(type)
        df_i = df_i.astype(np.int64)

        start_ts = df_i.values.min()
        end_ts = df_i.values.max()
        s_to_ns = 1_000_000_000  # 每秒的纳秒数
        step_ns = freq * s_to_ns

        addon += s_to_ns if end_ts != df_ts_max else 0

        ts_new = np.arange(start_ts, end_ts + addon, step_ns)
        # print(ts_new)
        y_new = f(ts_new)
        ts = pd.to_datetime(ts_new, unit='ns')
        df = pd.DataFrame({'ts': ts, 'd_value': y_new})
        df = df.dropna()

        return df, freq