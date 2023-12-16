import dataset
import numpy as np
import os
import torch
from collections import defaultdict
from datasets import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import default_data_collator
from config import cfg

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, tokenizer=None, verbose=True):
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
        dataset_['train'] = dataset.SmartHome(root=root, split='train', subset=['hh103'])
        dataset_['test'] = dataset.SmartHome(root=root, split='test', subset=['hh103'])
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(batch):
    return {key: [b[key] for b in batch] for key in batch[0]}


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    elif collate_mode == 'transformer':
        return default_data_collator
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


def process_dataset(dataset, tokenizer):
    max_length = cfg['max_length']

    def preprocess_(data, pad_width):
        pad_width = (0, pad_width)
        data = tokenizer(data)
        ts = np.array(data['ts_normalized'])
        d_value = np.array(data['d_value'])
        d_info = np.array(data['d_info'])
        ts = np.pad(ts, pad_width, mode='constant', constant_values=0)
        d_value = np.pad(d_value, pad_width, mode='constant', constant_values=0)
        d_info = np.pad(d_info, pad_width, mode='constant', constant_values=tokenizer.pad_token)
        ts = torch.tensor(ts, dtype=torch.float)
        d_value = torch.tensor(d_value, dtype=torch.float)
        d_info = torch.tensor(d_info, dtype=torch.float)
        processed_data = torch.stack([ts, d_value, d_info], dim=-1)
        return processed_data

    def preprocess_function(examples):
        batch_size = len(examples['data'])
        model_inputs = {'data': [], 'target': [], 'attention_mask': []}
        for i in range(batch_size):
            data = examples['data'][i]
            target = examples['target'][i]
            org_width = len(data['ts'])
            pad_width = max(0, max_length - org_width) # need truncation
            data = preprocess_(data, pad_width)
            target = preprocess_(target, 0)
            attention_mask = np.concatenate([np.ones(org_width, dtype=bool), np.zeros(pad_width, dtype=bool)], axis=0)
            attention_mask = torch.tensor(attention_mask)
            model_inputs['data'].append(data)
            model_inputs['target'].append(target)
            model_inputs['attention_mask'].append(attention_mask)
        return model_inputs

    processed_dataset = {}
    # 分别对训练集和测试集进行处理
    for split in dataset:
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
            desc="Preprocess dataset",
            batch_size=50,
        )

    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    return processed_dataset
