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
    def arr_split(arr, split_name, split_size):
        splits = np.split(arr, np.arange(split_size, arr.shape[0], split_size))
        if splits[-1].shape[0] != split_size:
            # print(f'last element has been dropped, which shade is {splits[-1].shape}')
            splits.pop()
        for i, split in enumerate(splits):
            np.save(f'{split_name}_{i}.npy', split)
        return splits

    def preprocess_function_token(examples):
        sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                          cache_folder=os.path.join('output', 'model'))
        batch_size = len(examples['data'])
        print('-------------batch size:{}'.format(batch_size))
        pad_len = 1000

        stacked_data = np.empty((0, pad_len, 770))  # var
        for j in range(batch_size):
            print('({}/{})'.format(j + 1, batch_size))
            data = examples['data'][j]
            ts = data['ts1']
            d_name, d_func, d_type = data['d_name'], data['d_func'], data['d_type']
            d_value = data['d_value']
            vect = np.empty((0, 770))
            for i in range(len(d_name)):
                # print('({}/{})'.format(i, len(d_name)-1))
                device_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[i], d_func[i], d_type[i])
                vec = sbert_model.encode([device_i])
                device_arr = np.array(vec)
                comb_arr = np.column_stack((ts[i], device_arr, d_value[i]))
                vect = np.vstack((vect, comb_arr))
            if vect.shape[0] > pad_len:
                vect = vect[-pad_len:]

            pad_width = ((0, pad_len - vect.shape[0]), (0, 0))
            # print(vect.shape)
            # print(pad_len - vect.shape[0])
            padded_data = np.pad(vect, pad_width, mode='constant', constant_values=0)
            padded_data = padded_data.reshape(1, pad_len, 770)
            # padded_data = np.pad(vect, pad_width=((0, pad_len - len(vect[:, 0])), (0, 0)), mode='constant')
            stacked_data = np.vstack((stacked_data, padded_data))

        # (50000, 1000, 770)    770输入 770输出

        print(stacked_data.shape)
        return stacked_data

    processed_dataset = {}
    # 分别对训练集和测试集进行处理
    for split in dataset:
        dataset[split].configure(['hh103'], 1200, 300, 2, (300,))  # 这里的参数有实际意义吗？  更换其他subset
        data = defaultdict(list)  # 访问不存在的键时，输出一个空列表[]
        for subset in dataset[split].subset:  # 遍历训练集/测试集中的子集
            for k in dataset[split].data[subset]:  # 遍历子集中的字典的key
                data[k].extend(dataset[split].data[subset][k])  # 将子集中的键值对赋值给data
        processed_dataset[split] = Dataset.from_dict(data)  # 从字典创建数据集
        ar_data = preprocess_function_token(processed_dataset[split])
        arr_split(ar_data, split, 50)
        # dt_now = datetime.datetime.now().time().strftime('%H-%M-%S.%f')
        # np.save(f'{split}_{dt_now}.npy', ar_data)

        # processed_dataset[split] = processed_dataset[split].map(
        #     preprocess_function_token,
        #     batched=True,
        #     num_proc=1,            #并发处理量
        #     remove_columns=['data', 'target', 'detect'],
        #     load_from_cache_file=False,
        #     desc="Preprocess dataset",
        #     batch_size=2000
        # )

    # print(f'Processed_dataset:\n\n{processed_dataset}')
    # print(torch.tensor(processed_dataset['train']['input_ids']).size())
    # data_0 = torch.tensor(processed_dataset['train']['input_ids'][0])
    # print(data_0.size())
    # # text = tokenizer.batch_decode(data_0.t())
    # # print(text[-5])
    exit()
    #
    # cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    # cfg['target_size'] = processed_dataset['train'].target_size
    return processed_dataset
