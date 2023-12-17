import numpy as np
import torch

class Tokenizer:
    def __init__(self, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', if_train=False):
        super().__init__()
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.if_train = if_train
        self.pad_direction = 'right'
        self.special_token = [self.unk_token, self.pad_token, self.cls_token, self.sep_token]
        for i in range(len(self.special_token)):
            self.update(self.special_token[i])

    def update(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[self.vocab[token]] = token
        return

    def train(self, if_train):
        self.if_train = if_train
        return

    def __call__(self, data, padding=False, truncation=False, max_length=None, return_tensors='np'):
        ts = data['ts_normalized']
        d_value = data['d_value']
        d_info = self.tokenize(data)
        seq_len = len(d_info)
        d_info = [self.convert_token_to_id(d_info[i]) for i in range(len(d_info))]
        if truncation and max_length is not None and max_length < seq_len:
            ts = ts[:max_length]
            d_value = d_value[:max_length]
            d_info = d_info[:max_length]
        seq_len = len(d_info)
        if padding and max_length is not None and max_length > seq_len:
            pad_width = max_length - seq_len
            ts = np.pad(ts, (0, pad_width), mode='constant', constant_values=0)
            d_value = np.pad(d_value, (0, pad_width), mode='constant', constant_values=0)







        ts = torch.tensor(ts, dtype=torch.float)
        d_value = torch.tensor(d_value, dtype=torch.float)
        d_info = torch.tensor(d_info, dtype=torch.float)
        data = torch.stack([ts, d_value, d_info], dim=-1)
        return data

    def tokenize(self, data):
        d_name, d_func, d_type = list(data['d_name']), list(data['d_func']), list(data['d_type'])
        d_info = []
        for j in range(len(d_name)):
            d_info_i = 'Name: {}, Function: {}, Info: {}'.format(d_name[j], d_func[j], d_type[j])
            if self.if_train:
                self.update(d_info_i)
            d_info.append(d_info_i)
        return d_info

    def convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)
