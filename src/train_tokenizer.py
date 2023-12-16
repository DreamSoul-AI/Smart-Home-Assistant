import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset
from model import make_tokenizer
from module import save, process_control, resume

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiment']))
    for i in range(cfg['num_experiment']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    model_path = os.path.join('output', 'model')
    model_tag_path = os.path.join(model_path, cfg['model_tag'])
    tokenizer_path = os.path.join(model_tag_path, 'tokenizer')
    tokenizer = make_tokenizer()
    dataset = make_dataset(cfg['data_name'])
    result = resume(os.path.join(tokenizer_path, 'model'), resume_mode=cfg['resume_mode'])
    if result is not None:
        tokenizer.load_state_dict(result['tokenizer_state_dict'])
    train(dataset['train'], tokenizer)
    result = {'cfg': cfg, 'tokenizer_state_dict': tokenizer.state_dict()}
    save(result, os.path.join(tokenizer_path, 'model'))
    return


def train(dataset, tokenizer):
    tokenizer.train(True)
    for i, input in enumerate(dataset):
        tokenizer(input['data'])
        tokenizer(input['target'])
    return


if __name__ == "__main__":
    main()
