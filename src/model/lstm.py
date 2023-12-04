import torch
import torch.nn as nn
import math
from config import cfg
from .model import init_param, make_loss


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size):
        super().__init__()
        input_size = data_shape[1]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True,
                            dropout=0.0, bidirectional=False)
        self.linear = nn.Linear(hidden_size, target_size)

    def feature(self, x):
        x = self.lstm(x)
        return x

    def output(self, x):
        x = self.linear(x)
        return x

    def f(self, x):
        x = self.feature(x)
        x = self.output(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = self.f(x)
        output['target'] = x
        output['loss'] = make_loss(output, input)
        return output


def lstm():
    data_shape = cfg['data_shape']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    target_size = cfg['target_size']
    model = LSTM(data_shape, hidden_size, num_layers, target_size)
    model.apply(init_param)
    return model
