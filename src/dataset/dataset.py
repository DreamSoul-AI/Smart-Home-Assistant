import dataset
import numpy as np
import os
import torch
from collections import defaultdict
from datasets import Dataset
from functools import partial
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from config import cfg

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, encoder=None, verbose=True):
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
        dataset_['train'].transform = dataset.Compose([partial(tokenize, encoder)])
        dataset_['test'].transform = dataset.Compose([partial(tokenize, encoder)])
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
        if k in ['data']:
            input[k] = pad_sequence(input[k], batch_first=True, padding_value=0.0)
        elif k in ['target', 'detect']:
            input[k] = torch.stack(input[k], 0)
    input['mask'] = input['data'].sum(dim=-1) != 0
    return input


# sentence bert  average across vectors 变成一维矢量
def process_dataset(dataset):
    processed_dataset = dataset
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    return processed_dataset


def tokenize(encoder, input):
    encoded_input = {'data': [], 'target': [], 'detect': []}
    data = input['data']
    ts = np.array(data['ts_normalized']).reshape(-1, 1)
    d_value = np.array(data['d_value']).reshape(-1, 1)
    d_name, d_func, d_type = data['d_name'].tolist(), data['d_func'].tolist(), data['d_type'].tolist()
    d_info = []
    for j in range(len(d_name)):
        d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
        d_info.append(d_info_i)
    d_vec = encoder.encode(d_info)
    d_vec = np.concatenate([ts, d_vec, d_value], axis=-1)
    encoded_input['data'] = torch.tensor(d_vec).float()
    encoded_input['detect'] = torch.tensor(input['detect'])
    if not input['target'].empty:
        target = input['target']
        ts = np.array(data['ts_normalized']).reshape(-1, 1)
        d_value = np.array(data['d_value']).reshape(-1, 1)
        d_name, d_func, d_type = target['d_name'].tolist(), target['d_func'].tolist(), target['d_type'].tolist()
        d_info = []
        for j in range(len(d_name)):
            d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
            d_info.append(d_info_i)
        d_vec = encoder.encode(d_info)
        d_vec = np.concatenate([ts, d_vec, d_value], axis=-1)
        encoded_input['target'] = torch.tensor(d_vec).float()
    else:
        encoded_input['target'] = torch.tensor(input['target'])
    return encoded_input
