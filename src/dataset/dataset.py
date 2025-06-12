import dataset
import numpy as np
import os
import torch
from collections import defaultdict
from datasets import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg
import pandas as pd

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, subset_name, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)  # 数据路径root: data\SmartHome
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SmartHome']:
        # dataset.Preprocess('data', data_name)  # 注释掉，因为我们已经有预处理好的数据
        dataset_['train'] = dataset.SmartHome(root=root, split='train', subset=subset_name)
        dataset_['test'] = dataset.SmartHome(root=root, split='test', subset=subset_name)
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    first = input[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in input])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in input]))
            else:
                batch[k] = torch.tensor([f[k] for f in input])
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset, tokenizer):
    max_length = cfg['max_length']
    window_length = dataset['train'].seq_len

    def preprocess_function(examples):
        data = tokenizer(examples, window_length=window_length, max_length=max_length, padding=True, truncation=True,
                         return_tensors='pt')
        model_inputs = {'data': data['data'], 'attention_mask': data['attention_mask'],
                        'control_mask': data['control_mask']}
        return model_inputs

    processed_dataset = {}
    # 分别对训练集和测试集进行处理
    for split in dataset:
        # SmartHome数据集的特殊处理
        if hasattr(dataset[split], 'data_name') and dataset[split].data_name == 'SmartHome':
            # 从数据集中收集所有数据
            all_data = []
            all_t_start = []
            
            for subset in dataset[split].subset:
                if subset in dataset[split].data:
                    subset_data = dataset[split].data[subset]
                    all_data.extend(subset_data['data'])
                    all_t_start.extend(subset_data['t_start'])
            
            # 创建一个简化的数据结构供tokenizer使用
            # 将简单的数值数组转换为tokenizer期望的格式
            formatted_data = []
            for i in range(len(all_data)):
                # 为LS005设备创建格式化数据
                # t_start是数组索引，需要转换为时间戳
                start_timestamp = pd.Timestamp('2015-01-01') + pd.Timedelta(seconds=all_t_start[i] * 300)
                data_dict = {
                    'd_value': all_data[i].tolist() if hasattr(all_data[i], 'tolist') else all_data[i],
                    'ts': pd.date_range(start=start_timestamp, periods=len(all_data[i]), freq='300s'),
                    'd_name': ['LS005'] * len(all_data[i]),
                    'd_func': ['light'] * len(all_data[i]),
                    'd_type': ['sensor'] * len(all_data[i])
                }
                formatted_data.append(data_dict)
            
            # 创建HuggingFace Dataset格式
            # 将t_start索引转换为时间戳
            formatted_t_start = []
            for t_idx in all_t_start:
                formatted_t_start.append(pd.Timestamp('2015-01-01') + pd.Timedelta(seconds=t_idx * 300))
            
            dataset_dict = {'data': formatted_data, 't_start': formatted_t_start}
            processed_dataset[split] = Dataset.from_dict(dataset_dict)
            processed_dataset[split] = processed_dataset[split].map(
                preprocess_function,
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                desc='Preprocess dataset',
                remove_columns=['t_start'],
                batch_size=50,
            )
        else:
            # 原有的处理逻辑
            data = defaultdict(list)
            for subset in dataset[split].subset:
                for k in dataset[split].data[subset]:
                    data[k].extend(dataset[split].data[subset][k])
            processed_dataset[split] = Dataset.from_dict(data)
            processed_dataset[split] = processed_dataset[split].map(
                preprocess_function,
                batched=True,
                num_proc=1,
                load_from_cache_file=False,
                desc='Preprocess dataset',
                remove_columns=['t_start'],
                batch_size=50,
            )

    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset
