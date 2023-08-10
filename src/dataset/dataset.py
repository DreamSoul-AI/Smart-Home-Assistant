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
    max_length = 512
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function_token(examples):
        max_length_time = 32
        max_length_device = 16
        max_length_value = 8
        feature_size = max_length_time + max_length_device + max_length_value
        batch_size = len(examples['data'])
        model_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for i in range(batch_size):
            data = examples['data'][i]
            ts = data['ts']
            d_name, d_func, d_type = data['d_name'], data['d_func'], data['d_type']
            d_value = data['d_value']
            time = []
            device = []
            value = []
            for j in range(len(d_name)):
                month_j, day_j, hour_j, minute_j, second_j = (ts[j].month, ts[j].day, ts[j].hour,
                                                              ts[j].minute, ts[j].second)

                time_j = 'Month: {}, Day: {}, Hour: {}, Minute: {}, Second: {}'.format(month_j, day_j,
                                                                                       hour_j, minute_j, second_j)
                device_j = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
                value_i = str(d_value[j])
                time.append(time_j)
                device.append(device_j)
                value.append(value_i)
            token_time = tokenizer(time, truncation=True, padding='max_length', max_length=max_length_time)
            token_device = tokenizer(device, truncation=True, padding='max_length', max_length=max_length_device)
            token_value = tokenizer(value, truncation=True, padding='max_length', max_length=max_length_value)
            input_ids_i = torch.cat(
                [torch.tensor(token_time['input_ids']), torch.tensor(token_device['input_ids']),
                 torch.tensor(token_value['input_ids'])], dim=-1)
            attention_mask_i = torch.cat(
                [torch.tensor(token_time['attention_mask']), torch.tensor(token_device['attention_mask']),
                 torch.tensor(token_value['attention_mask'])], dim=-1)
            model_inputs['input_ids'].append(input_ids_i[:-1])
            model_inputs['attention_mask'].append(attention_mask_i)
            model_inputs['labels'].append(input_ids_i[[-1]])
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = torch.cat([model_inputs["labels"][i],
                                         torch.tensor(tokenizer.pad_token_id).repeat(feature_size).unsqueeze(0)], dim=0)
            model_inputs['input_ids'][i] = torch.cat([sample_input_ids, label_input_ids], dim=0)
            mask_out = torch.tensor([-100]).expand_as(sample_input_ids)
            model_inputs['labels'][i] = torch.cat([mask_out, label_input_ids], dim=0)
            model_inputs["attention_mask"][i] = torch.cat([model_inputs['attention_mask'][i],
                                                           torch.ones(1, feature_size)], dim=0)
            # if len(model_inputs['input_ids']) > max_length:
            #     print(len(model_inputs['input_ids']))
            model_inputs['input_ids'][i] = model_inputs['input_ids'][i][-max_length:]
            model_inputs['attention_mask'][i] = model_inputs['attention_mask'][i][-max_length:]
            model_inputs['labels'][i] = model_inputs['labels'][i][-max_length:]
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            if len(sample_input_ids) < max_length:
                pad_token = torch.tensor([tokenizer.pad_token_id]).expand(max_length -
                                                                          len(sample_input_ids), feature_size)
                model_inputs["input_ids"][i] = torch.cat([pad_token, model_inputs["input_ids"][i]], dim=0)
                pad_token = torch.tensor([0]).expand(max_length - len(sample_input_ids), feature_size)
                model_inputs["attention_mask"][i] = torch.cat([pad_token, model_inputs["attention_mask"][i]], dim=0)
                pad_token = torch.tensor([-100]).expand(max_length - len(sample_input_ids), feature_size)
                model_inputs["labels"][i] = torch.cat([pad_token, model_inputs["labels"][i]], dim=0)
            model_inputs["input_ids"][i] = model_inputs["input_ids"][i].t()
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i].t()
            model_inputs["labels"][i] = model_inputs["labels"][i].t()
        return model_inputs

    processed_dataset = {}
    for split in dataset:
        dataset[split].configure(['hh103'], 1200, 300, 2, (300,))
        data = defaultdict(list)
        for subset in dataset[split].subset:
            for k in dataset[split].data[subset]:
                data[k].extend(dataset[split].data[subset][k])
        processed_dataset[split] = Dataset.from_dict(data)

        processed_dataset[split] = processed_dataset[split].map(
            preprocess_function_token,
            batched=True,
            num_proc=1,
            remove_columns=['data', 'target', 'detect'],
            load_from_cache_file=False,
            desc="Preprocess dataset",
        )


    print(processed_dataset)
    print(torch.tensor(processed_dataset['train']['input_ids']).size())
    data_0 = torch.tensor(processed_dataset['train']['input_ids'][0])
    print(data_0.size())
    text = tokenizer.batch_decode(data_0.t())
    print(text[-5])
    exit()

    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    # cfg['target_size'] = processed_dataset['train'].target_size
    return processed_dataset
