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

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
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


def process_dataset(dataset, tokenizer):
    max_length = 12
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    def preprocess_function_token(examples):
        data = examples['data']
        ts = data['ts']
        d_name, d_func, d_type = data['d_name'], data['d_func'], data['d_type']
        d_value = data['d_value']
        time = []
        device = []
        value = []
        for i in range(len(d_name)):
            month_i, day_i, hour_i, minute_i, second_i = (ts[i].month, ts[i].day, ts[i].hour,
                                                          ts[i].minute, ts[i].second)

            time_i = 'Month: {}, Day: {}, Hour: {}, Minute: {}, Second: {}'.format(month_i, day_i,
                                                                                   hour_i, minute_i, second_i)
            device_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[i], d_func[i], d_type[i])
            value_i = str(d_value[i])
            time.append(time_i)
            device.append(device_i)
            value.append(value_i)
        token_time = tokenizer(time, truncation=True, padding='max_length', max_length=32)
        token_device = tokenizer(device, truncation=True, padding='max_length', max_length=16)
        token_value = tokenizer(value, truncation=True, padding='max_length', max_length=8)
        model_inputs = {}
        model_inputs['input_ids'] = torch.cat(
            [torch.tensor(token_time['input_ids']), torch.tensor(token_device['input_ids']),
             torch.tensor(token_value['input_ids'])], dim=-1).t()
        model_inputs['attention_mask'] = torch.cat(
            [torch.tensor(token_time['attention_mask']), torch.tensor(token_device['attention_mask']),
             torch.tensor(token_value['attention_mask'])], dim=-1).t()
        model_inputs = tokenizer.pad(model_inputs, max_length=max_length, padding='max_length', return_tensors="pt")
        model_inputs['input_ids'] = model_inputs['input_ids'][:, -max_length:]
        model_inputs['attention_mask'] = model_inputs['attention_mask'][:, -max_length:]
        print(model_inputs['attention_mask'])
        exit()
        print(model_inputs['input_ids'].size())
        exit()

        return model_inputs


    processed_dataset = {}
    for split in dataset:
        dataset[split].configure(['hh103'], 1200, 300, 2, (300,))
        data = defaultdict(list)
        for subset in dataset[split].subset:
            for k in dataset[split].data[subset]:
                data[k].extend(dataset[split].data[subset][k])
        processed_dataset[split] = Dataset.from_dict(data)



        # def preprocess_function_pad(examples):
        #     print(examples['input_ids'][0].size(), )
        #     # targets = examples[label_column]
        #     # model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True,
        #     #                          return_tensors="pt")
        #     # labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
        #     # labels = labels["input_ids"]
        #     # labels[labels == tokenizer.pad_token_id] = -100
        #     # model_inputs["labels"] = labels
        #     return examples

        processed_dataset[split] = processed_dataset[split].map(
            preprocess_function_token,
            batched=False,
            num_proc=1,
            remove_columns=['data', 'target', 'detect'],
            load_from_cache_file=False,
            desc="Preprocess dataset",
        )

        # processed_dataset[split] = processed_dataset[split].map(
        #     preprocess_function_pad,
        #     batched=False,
        #     num_proc=1,
        #     load_from_cache_file=False,
        #     desc="Preprocess dataset",
        # )
    print(processed_dataset)
    exit()

    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    # cfg['target_size'] = processed_dataset['train'].target_size
    return processed_dataset
