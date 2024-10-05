import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import init_param


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size, seq_len, label_len, pred_len):
        super().__init__()
        input_size = data_shape[-1]
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True,
                            dropout=0.0, bidirectional=False)
        self.decoder = nn.Linear(hidden_size, target_size)

        # 加载各种len
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def feature(self, x):
        x, _ = self.encoder(x)
        return x

    def output(self, x):
        x = self.decoder(x)
        return x

    def f(self, x):
        x = self.feature(x)
        print(f"encoder-x.shape:{x.shape}")
        x = self.output(x[:, -self.pred_len:, :])
        print(f"decoder-x.shape:{x.shape}")
        return x

    def forward(self, input):
        print(f"seq_len:{self.seq_len}")
        print(f"label_len:{self.label_len}")
        print(f"pred_len:{self.pred_len}")
        output = {}
        print(f'lstm.input:{input}')
        input['data'] = input['data'].float()
        x = input['data'][:, 0:self.seq_len, :]
        print(f"x.shape1: {x.shape}")  # 打印原始输入形状
        x = x.float()  # 确保输入数据为float类型
        x = self.f(x)
        print( f"x.shape2: {x.shape}")  # 打印经过f函数后的形状

        # 目标步骤是输入序列的后50个时间步
        # target_steps = input['data'][:, 600:, :]
        # print(f"target_steps.shape: {target_steps.shape}")
        # predicted_steps = x[:, 600:, :]
        target_steps = input['data'][:, self.seq_len+self.label_len:, :]
        print(f"target_steps.shape: {target_steps.shape}")
        predicted_steps = x[:, :, :]
        print(f"predicted_steps.shape: {predicted_steps.shape}")
        # 计算损失
        loss = F.mse_loss(predicted_steps, target_steps, reduction='mean')
        output['loss'] = loss
        output['predicted_steps'] = predicted_steps

        output['target'] = x
        # mask = input['mask'][:, 1:]
        # output_target = output['target'][:, :-1][mask]
        # input_target = input['data'][:, 1:][mask]
        # loss = F.mse_loss(output_target, input_target, reduction='mean')
        # output['loss'] = loss
        return output


def lstm(cfg):
    data_shape = cfg['data_shape']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    seq_len = cfg['lstm']['seq_len']
    label_len = cfg['lstm']['label_len']
    pred_len = cfg['lstm']['pred_len']
    target_size = cfg['target_size']
    target_size = 1
    print(f"hidden_size: {hidden_size}")
    print(f"target_size: {target_size}")
    model = LSTM(data_shape, hidden_size, num_layers, target_size, seq_len, label_len, pred_len)
    model.apply(init_param)
    return model
