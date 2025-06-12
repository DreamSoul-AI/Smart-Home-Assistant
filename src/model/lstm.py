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
        # print(f'lstm.input:{input}')  # 注释掉这行，避免打印整个数据张量
        input['data'] = input['data'].float()
        
        # 提取输入序列的前seq_len个时间步
        x = input['data'][:, 0:self.seq_len, :]
        print(f"x.shape1: {x.shape}")  # 打印原始输入形状
        x = x.float()  # 确保输入数据为float类型
        
        # 通过模型得到预测
        x = self.f(x)
        print(f"x.shape2: {x.shape}")  # 打印经过f函数后的形状

        # 提取目标值：从输入的第seq_len个时间步开始，取pred_len个时间步的第二个特征（传感器值）
        if input['data'].shape[1] >= self.seq_len + self.pred_len:
            # 只取第二个特征（索引1）作为目标，因为我们只预测传感器值
            target_steps = input['data'][:, self.seq_len:self.seq_len+self.pred_len, 1:2]
        else:
            # 如果数据长度不够，从可用的数据中提取
            available_steps = input['data'].shape[1] - self.seq_len
            target_steps = input['data'][:, self.seq_len:, 1:2]
            # 如果目标步数少于预测步数，截断预测
            if available_steps < self.pred_len:
                x = x[:, :available_steps, :]
        
        print(f"target_steps.shape: {target_steps.shape}")
        predicted_steps = x
        print(f"predicted_steps.shape: {predicted_steps.shape}")
        
        # 计算损失
        loss = F.mse_loss(predicted_steps, target_steps, reduction='mean')
        output['loss'] = loss
        output['predicted_steps'] = predicted_steps
        output['target'] = target_steps  # 保存目标值用于评估

        return output


def lstm(cfg):
    data_shape = cfg['data_shape']
    print(f"Debug - data_shape in lstm(): {data_shape}")
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    seq_len = cfg['lstm']['seq_len']
    label_len = cfg['lstm']['label_len']
    pred_len = cfg['lstm']['pred_len']
    target_size = cfg['target_size']
    target_size = 1
    print(f"hidden_size: {hidden_size}")
    print(f"target_size: {target_size}")
    print(f"Creating LSTM with input_size: {data_shape[-1]}")
    model = LSTM(data_shape, hidden_size, num_layers, target_size, seq_len, label_len, pred_len)
    model.apply(init_param)
    return model
