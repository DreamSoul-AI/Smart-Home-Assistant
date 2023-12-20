import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from model import LSTM, init_param
from sentence_transformers import SentenceTransformer


class Base(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.text_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                                cache_folder=os.path.join('output', 'model'))
        self.embedding = self.make_embedding()
        self.encoder = nn.Linear(embedding_size, hidden_size)
        self.core = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True,
                            dropout=0.0, bidirectional=False)
        self.decoder = nn.Linear(hidden_size, embedding_size)

    def make_embedding(self):
        vocab = list(self.tokenizer.vocab.keys())
        info_embedding = torch.tensor(self.text_encoder.encode(vocab))
        num_embedding, embedding_dim = info_embedding.shape
        embedding = nn.Embedding(num_embedding, embedding_dim,
                                 padding_idx=self.tokenizer.convert_token_to_id(self.tokenizer.pad_token))
        embedding.weight.data.copy_(info_embedding.data)
        embedding.weight.requires_grad = False
        return embedding

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def f(self, x):
        x = self.encode(x)
        x, _ = self.core(x)
        x = self.decoder(x)
        return x

    def forward(self, input):
        output = {}
        x_target = input['data'][..., :2]
        x_info_target = input['data'][..., 2].long()
        mask = input['attention_mask'][:, 1:]
        x_info = self.embedding(x_info_target)
        x = torch.cat([x_target, x_info], dim=-1)
        x = self.f(x)

        x = x[:, :-1]
        x_ts, x_value, x_info = x[..., 0],  x[..., 1], x[..., 2:]
        x_info = x_info @ self.embedding.weight.t()

        x_target = x_target[:, 1:]
        x_ts_target = x_target[..., 0]
        x_value_target = x_target[..., 1]
        x_info_target = x_info_target[:, 1:]

        x_ts = x_ts[mask]
        x_ts_target = x_ts_target[mask]
        x_value = x_value[mask]
        x_value_target = x_value_target[mask]

        x_info = x_info.transpose(1, 2)
        x_info_target[~mask] = -100
        num_loss = mask.float().sum()
        mse_loss_ts = F.mse_loss(x_ts, x_ts_target, reduction='sum') / num_loss
        mse_loss_value = F.mse_loss(x_value, x_value_target, reduction='sum') / num_loss
        ce_loss_info = F.cross_entropy(x_info, x_info_target, reduction='sum') / num_loss
        loss = mse_loss_ts + mse_loss_value + ce_loss_info
        output['loss'] = loss

        x_info = torch.argmax(x_info, dim=1)[mask].float()
        output['target'] = torch.stack([x_ts, x_value, x_info], dim=-1)
        return output


def base(tokenizer):
    embedding_size = cfg['embedding_size']
    hidden_size = cfg[cfg['model_name']]['hidden_size']
    num_layers = cfg[cfg['model_name']]['num_layers']
    model = Base(tokenizer, embedding_size, hidden_size, num_layers)
    model.apply(init_param)
    return model
