import dataset
import numpy as np
import os
import torch
from collections import defaultdict
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, verbose=True):
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


# sentence bert  average across vectors 变成一维矢量
def process_dataset(dataset):
    # def arr_split(arr, split_name, split_size):
    #     splits = np.split(arr, np.arange(split_size, arr.shape[0], split_size))
    #     if splits[-1].shape[0] != split_size:
    #         # print(f'last element has been dropped, which shade is {splits[-1].shape}')
    #         splits.pop()
    #     for i, split in enumerate(splits):
    #         np.save(f'{split_name}_{i}.npy', split)
    #     return splits

    # def preprocess_function_token(examples):
    #     sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
    #                                       cache_folder=os.path.join('output', 'model'))
    #     batch_size = len(examples['data'])
    #     print('-------------batch size:{}'.format(batch_size))
    #     pad_len = 1000
    #
    #     stacked_data = np.empty((0, pad_len, 770))  # var
    #     for j in range(batch_size):
    #         print('({}/{})'.format(j + 1, batch_size))
    #         data = examples['data'][j]
    #         ts = data['ts1']
    #         d_name, d_func, d_type = data['d_name'], data['d_func'], data['d_type']
    #         d_value = data['d_value']
    #         vect = np.empty((0, 770))
    #         for i in range(len(d_name)):
    #             # print('({}/{})'.format(i, len(d_name)-1))
    #             device_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[i], d_func[i], d_type[i])
    #             vec = sbert_model.encode([device_i])
    #             device_arr = np.array(vec)
    #             comb_arr = np.column_stack((ts[i], device_arr, d_value[i]))
    #             vect = np.vstack((vect, comb_arr))
    #         if vect.shape[0] > pad_len:
    #             vect = vect[-pad_len:]
    #
    #         pad_width = ((0, pad_len - vect.shape[0]), (0, 0))
    #         # print(vect.shape)
    #         # print(pad_len - vect.shape[0])
    #         padded_data = np.pad(vect, pad_width, mode='constant', constant_values=0)
    #         padded_data = padded_data.reshape(1, pad_len, 770)
    #         # padded_data = np.pad(vect, pad_width=((0, pad_len - len(vect[:, 0])), (0, 0)), mode='constant')
    #         stacked_data = np.vstack((stacked_data, padded_data))
    #
    #     # (50000, 1000, 770)    770输入 770输出
    #
    #     print(stacked_data.shape)
    #     return stacked_data

    def preprocess_function(examples):
        max_length = 128
        batch_size = len(examples['data'])
        model_inputs = {'data': [], 'attention_mask': [], 'target': []}
        for i in range(batch_size):
            data = examples['data'][i]
            ts = np.array(data['ts_normalized']).reshape(-1, 1)
            d_name, d_func, d_type = data['d_name'], data['d_func'], data['d_type']
            d_value = np.array(data['d_value']).reshape(-1, 1)
            d_str = []
            for j in range(len(d_name)):
                d_str_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
                d_str.append(d_str_i)
            d_vec = sbert_model.encode(d_str)
            d_vec = np.concatenate([ts, d_vec, d_value], axis=-1)
            if d_vec.shape[0] >= max_length:
                d_vec = d_vec[-max_length:]
                attention_mask = np.ones(max_length)
            else:
                org_width = d_vec.shape[0]
                pad_width_ = max_length - org_width
                pad_width = ((pad_width_, 0), (0, 0))
                d_vec = np.pad(d_vec, pad_width, mode='constant', constant_values=0)
                attention_mask = np.concatenate([np.zeros(pad_width_), np.ones(org_width)], axis=0)
            d_vec = torch.tensor(d_vec).float()
            attention_mask = torch.tensor(attention_mask).bool()
            model_inputs['data'].append(d_vec)
            model_inputs['attention_mask'].append(attention_mask)
            model_inputs['target'].append(d_vec)
        return model_inputs

    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                      cache_folder=os.path.join('output', 'model'))
    processed_dataset = {}
    # 分别对训练集和测试集进行处理
    for split in dataset:
        dataset[split].configure(['hh103'], 1200, 300, 2, (300,))
        data = defaultdict(list)
        for subset in dataset[split].subset:
            for k in dataset[split].data[subset]:
                data[k].extend(dataset[split].data[subset][k])
        processed_dataset[split] = Dataset.from_dict(data)

        processed_dataset[split] = processed_dataset[split].map(
            preprocess_function,
            batched=True,
            num_proc=2,
            load_from_cache_file=False,
            desc="Preprocess dataset",
            batch_size=50,
        )

    exit()
    # cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    # cfg['target_size'] = processed_dataset['train'].target_size
    return processed_dataset
