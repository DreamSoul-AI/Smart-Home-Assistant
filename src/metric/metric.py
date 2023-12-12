import torch
import torch.nn.functional as F
from collections import defaultdict
from config import cfg
from module import recur


def make_metric(metric_name):
    if cfg['data_name'] in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        pivot = -float('inf')
        pivot_direction = 'up'
        pivot_name = 'Accuracy'
        for k in metric_name:
            metric_name[k].extend(['Accuracy'])
    if cfg['data_name'] in ['SmartHome']:
        pivot = float('inf')
        pivot_direction = 'down'
        pivot_name = 'Loss'
        for k in metric_name:
            metric_name[k].extend(['MAE-ar-time', 'Accuracy-ar-d~info', 'MSE-ar-d~value'])
    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name)
    return metric


def Accuracy(output, target, topk=1):
    with torch.no_grad():
        if target.dtype != torch.int64:
            target = (target.topk(1, -1, True, True)[1]).view(-1)
        batch_size = torch.numel(target)
        pred_k = output.topk(topk, -1, True, True)[1]
        correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


def mae_time(output, target, mask):
    mask = mask[:, 1:]
    output_ts = output[:, :-1, 0][mask]
    target_ts = target[:, 1:, 0][mask]
    mae = F.l1_loss(output_ts, target_ts, reduction='mean')
    mae = mae.item()
    return mae


def acc_d_info(output, target, mask, tokenizer):
    acc = 0
    return acc


def mse_d_value(output, target, mask):
    mse = 0
    return mse


class Metric:
    def __init__(self, metric_name, pivot, pivot_direction, pivot_name):
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        self.metric_name = metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'MAE-ar-time':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mae_time, output['target'], input['data'],
                                                                        input['mask']))}
                elif m == 'Accuracy-ar-d~info':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(acc_d_info, output['target'], input['data'],
                                                                        input['mask'], input['tokenizer']))}
                elif m == 'MSE-ar-d~value':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_d_value, output['target'], input['mask'],
                                                                        input['data']))}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input=None, output=None):
        evaluation = {}
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == mode:
                evaluation[metric_name] = self.metric[split][metric_name]['metric'](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}
