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
            metric_name[k].extend(['MSE-ar-d~value', 'MSE-ar-time', 'Accuracy-ar-d~info'])
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


def mse_ts(output, target, mask):
    output_ts = output[:, 0]
    target_ts = target[:, 1:, 1][mask[:, 1:]]
    mse = F.mse_loss(output_ts, target_ts, reduction='mean').item()
    return mse


def mse_d_value(output, target, mask):
    output_d_value = output[:, 1]
    target_d_value = target[:, 1:, 2][mask[:, 1:]]
    mse = F.mse_loss(output_d_value, target_d_value, reduction='mean').item()
    return mse


def acc_d_info(output, target, mask):
    output_d_info = output[:, 2]
    target_d_info = target[:, 1:, 3][mask[:, 1:]]
    acc = (output_d_info == target_d_info).float().mean().item()
    return acc


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
                elif m == 'MSE-ar-time':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_ts, output['target'], input['data'],
                                                                        input['attention_mask']))}
                elif m == 'MSE-ar-d~value':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_d_value, output['target'], input['data'],
                                                                        input['attention_mask']))}
                elif m == 'Accuracy-ar-d~info':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(acc_d_info, output['target'], input['data'],
                                                                        input['attention_mask']))}
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
