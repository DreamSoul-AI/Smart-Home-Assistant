import torch
from collections import OrderedDict


class Tokenizer:
    def __init__(self, if_train=False):
        super().__init__()
        self.vocab = OrderedDict()
        self.if_train = if_train
        self.pad_token = 0
        self.vocab['[PAD]'] = self.pad_token

    def update(self, d_info):
        if d_info not in self.vocab:
            self.vocab[d_info] = len(self.vocab)
        return

    def train(self, if_train):
        self.if_train = if_train
        return

    def __call__(self, data):
        d_name, d_func, d_type = list(data['d_name']), list(data['d_func']), list(data['d_type'])
        d_info = []
        for j in range(len(d_name)):
            d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
            d_info.append(d_info_i)
        for i in range(len(d_info)):
            if self.if_train:
                self.update(d_info[i])
            d_info[i] = self.vocab[d_info[i]]
        data['d_info'] = d_info
        return data

    def state_dict(self):
        return {'dict': self.vocab}

    def load_state_dict(self, state_dict):
        self.vocab = state_dict['dict']
        return

