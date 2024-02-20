import torch
import torch.nn.functional as F
from collections import defaultdict
from module import recur


def make_metric(split, **kwargs):
    data_name = kwargs['data_name']
    metric_name = {k: [] for k in split}
    if data_name in ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
        best = -float('inf')
        best_direction = 'up'
        best_metric_name = 'Accuracy'
        for k in metric_name:
            metric_name[k].extend(['Loss', 'Accuracy'])
    elif data_name in ['SmartHome']:
        best = float('inf')
        best_direction = 'down'
        best_metric_name = 'Loss'
        for k in metric_name:
            metric_name[k].extend(
                ['Loss', 'MSE-ts', 'MSE-d~value', 'Accuracy-d~info', 'MSE-control-ts', 'MSE-control-d~value',
                 'Accuracy-control-d~info'])
    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, best, best_direction, best_metric_name)
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


def MSE(output, target):
    with torch.no_grad():
        mse = F.mse_loss(output, target).item()
    return mse


def mse_ts(output, target, attention_mask, control_mask=None):
    if control_mask is not None:
        mask = torch.logical_and(attention_mask, control_mask)
        if not torch.any(mask):
            return None
    else:
        mask = attention_mask
    output_ts = output[:, :, 0][mask[:, 1:]]
    target_ts = target[:, 1:, 0][mask[:, 1:]]
    mse = F.mse_loss(output_ts, target_ts, reduction='mean').item()
    return mse


def mse_d_value(output, target, attention_mask, control_mask=None):
    if control_mask is not None:
        mask = torch.logical_and(attention_mask, control_mask)
        if not torch.any(mask):
            return None
    else:
        mask = attention_mask
    output_d_value = output[:, :, 1][mask[:, 1:]]
    target_d_value = target[:, 1:, 1][mask[:, 1:]]
    mse = F.mse_loss(output_d_value, target_d_value, reduction='mean').item()
    return mse


def acc_d_info(output, target, attention_mask, control_mask=None):
    if control_mask is not None:
        mask = torch.logical_and(attention_mask, control_mask)
        if not torch.any(mask):
            return None
    else:
        mask = attention_mask
    output_d_info = output[:, :, 2][mask[:, 1:]]
    target_d_info = target[:, 1:, 2][mask[:, 1:]]
    acc = (output_d_info == target_d_info).float().mean().item()
    return acc


class Metric:
    def __init__(self, metric_name, best, best_direction, best_metric_name):
        self.metric_name = metric_name
        self.best, self.best_direction, self.best_metric_name = best, best_direction, best_metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'MSE-ts':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_ts, output['target'], input['data'],
                                                                        input['attention_mask']))}
                elif m == 'MSE-d~value':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_d_value, output['target'], input['data'],
                                                                        input['attention_mask']))}
                elif m == 'Accuracy-d~info':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(acc_d_info, output['target'], input['data'],
                                                                        input['attention_mask']))}
                elif m == 'MSE-control-ts':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_ts, output['target'], input['data'],
                                                                        input['attention_mask'],
                                                                        input['control_mask']))}
                elif m == 'MSE-control-d~value':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(mse_d_value, output['target'], input['data'],
                                                                        input['attention_mask'],
                                                                        input['control_mask']))}
                elif m == 'Accuracy-control-d~info':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (
                                            lambda input, output: recur(acc_d_info, output['target'], input['data'],
                                                                        input['attention_mask'],
                                                                        input['control_mask']))}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input, output, metric_name):
        evaluation = {}
        for metric_name_i in metric_name[split]:
            if self.metric[split][metric_name_i]['mode'] == mode:
                evaluation[metric_name_i] = self.metric[split][metric_name_i]['metric'](input, output)
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            raise ValueError('Not valid best direction')
        if if_update:
            self.best = val
        return compared

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        return

    def state_dict(self):
        return {'best': self.best, 'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}
