import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, resume, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 2
    data_names = ['SmartHome']
    subset_names = ['hh105~2015']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tokenizer_path'] = os.path.join(cfg['path'], 'tokenizer')
    with torch.no_grad():
        for data_name in data_names:
            for subset_name in subset_names:
                cfg['control']['data_name'] = '-'.join([data_name, subset_name])
                process_control()
                dataset = make_dataset(cfg['data_name'], cfg['subset_name'])
                tokenizer = resume(os.path.join(cfg['tokenizer_path']))[cfg['data_name']]
                dataset = process_dataset(dataset, tokenizer)
                cfg['step'] = 0
                data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
                stats = Stats(dim=dim)
                for i, input in enumerate(data_loader['train']):
                    stats.update(input['data'])
                stats = (stats.mean.tolist(), stats.std.tolist())
                print(data_name, stats)
                makedir_exist_ok(stats_path)
                save(stats, os.path.join(stats_path, '{}'.format(data_name)))
