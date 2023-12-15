import torch
from collections import OrderedDict


class Tokenizer:
    def __init__(self, if_train=True):
        super().__init__()
        self.dict = OrderedDict()
        self.if_train = if_train

    def update(self, d_info):
        if d_info not in self.dict:
            self.dict[d_info] = len(self.dict)
        return

    def tokenize_(self, data):
        d_name, d_func, d_type = data['d_name'].tolist(), data['d_func'].tolist(), data['d_type'].tolist()
        d_info = []
        for j in range(len(d_name)):
            d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
            d_info.append(d_info_i)
        for i in range(len(d_info)):
            if self.if_train:
                self.update(d_info[i])
            d_info[i] = self.dict[d_info[i]]
        data['d_info'] = d_info
        return data

    def tokenize(self, input):
        self.tokenize_(input['data'])
        self.tokenize_(input['target'])
        return input

    def train(self, if_train):
        self.if_train = if_train
        return

    def state_dict(self):
        return {'dict': self.dict}

    def load_state_dict(self, state_dict):
        self.dict = state_dict['dict']
        return
