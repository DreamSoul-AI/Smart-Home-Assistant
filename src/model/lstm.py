import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size):
        super().__init__()
        input_size = data_shape[0]
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True,
                            dropout=0.0, bidirectional=False)
        self.decoder = nn.Linear(hidden_size, target_size)

    def feature(self, x):
        x, _ = self.encoder(x)
        return x

    def output(self, x):
        x = self.decoder(x)
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
        mask = input['mask'][:, 1:]
        output_target = output['target'][:, :-1][mask]
        input_target = input['data'][:, 1:][mask]
        loss = F.mse_loss(output_target, input_target, reduction='mean')
        output['loss'] = loss
        return output


def lstm(cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    target_size = cfg['target_size']
    model = LSTM(data_shape, hidden_size, num_layers, target_size)
    model.apply(init_param)
    return model
