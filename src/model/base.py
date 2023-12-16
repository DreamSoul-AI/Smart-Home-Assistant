import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .model import make_model, init_param
from .tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self, num_embedding, embedding_size):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_size = embedding_size
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                           cache_folder=os.path.join('output', 'model'))

    def encode(self, input):
        with torch.no_grad():
            tokenized_input = {'info': None, 'data': None, 'target': None, 'detect': None}
            data = input['data']
            ts = np.array(data['ts_normalized']).reshape(-1, 1)
            d_value = np.array(data['d_value']).reshape(-1, 1)
            d_name, d_func, d_type = data['d_name'].tolist(), data['d_func'].tolist(), data['d_type'].tolist()
            d_info = []
            for j in range(len(d_name)):
                d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
                d_info.append(d_info_i)
            tokenized_input['info'] = d_info
            d_vec = self.encoder.encode(d_info)
            for i in range(len(d_info)):
                self.update(d_info[i], d_vec[i])
            d_vec = np.concatenate([d_vec, d_value, ts], axis=-1)
            tokenized_input['data'] = torch.tensor(d_vec).float()
            tokenized_input['detect'] = torch.tensor(input['detect'])
            if not input['target'].empty:
                target = input['target']
                ts = np.array(data['ts_normalized']).reshape(-1, 1)
                d_value = np.array(data['d_value']).reshape(-1, 1)
                d_name, d_func, d_type = target['d_name'].tolist(), target['d_func'].tolist(), target['d_type'].tolist()
                d_info = []
                for j in range(len(d_name)):
                    d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
                    d_info.append(d_info_i)
                d_vec = self.encoder.encode(d_info)
                d_vec = np.concatenate([d_vec, d_value, ts], axis=-1)
                tokenized_input['target'] = torch.tensor(d_vec).float()
            else:
                tokenized_input['target'] = torch.tensor(input['target'])
        return tokenized_input

    def state_dict(self):
        return {'dict': self.dict}

    def load_state_dict(self, state_dict):
        self.dict = state_dict['dict']
        return


class Base(nn.Module):
    def __init__(self, num_embedding, hidden_size, embedding_size):
        super().__init__()
        self.encoder = Encoder(num_embedding, embedding_size)
        self.core = make_model(cfg['model_name'])
        self.decoder = nn.Linear(hidden_size, embedding_size)

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def f(self, x):
        x = self.encode(x)
        x = self.core(x)
        x = self.decoder(x)
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


def base():
    num_embedding = cfg['num_embedding']
    embedding_size = cfg['embedding_size']
    hidden_size = cfg[cfg['model_name']]['hidden_size']
    model = Base(num_embedding, embedding_size, hidden_size)
    model.apply(init_param)
    return model
