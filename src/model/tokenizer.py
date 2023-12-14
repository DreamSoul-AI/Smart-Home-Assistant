import numpy as np
import os
import torch
import torch.nn as nn
from config import cfg
from collections import OrderedDict
from sentence_transformers import SentenceTransformer


class Tokenizer:
    def __init__(self):
        super().__init__()
        self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2',
                                           cache_folder=os.path.join('output', 'model'))
        self.embedding_size = cfg['embedding_size']
        self.dict = OrderedDict()

    def update(self, d_info, d_vec):
        if d_info not in self.dict:
            self.dict[d_info] = d_vec
        return

    def tokenize(self, input):
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
