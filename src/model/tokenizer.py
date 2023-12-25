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
        self.padding_direction = 'right'
        self.special_token = [self.unk_token, self.pad_token, self.cls_token, self.sep_token]
        for i in range(len(self.special_token)):
            self.update(self.special_token[i])

    @property
    def normalization(self):
        # Constants
        days_per_year = 365
        hours_per_day = 24
        minutes_per_hour = 60
        seconds_per_minute = 60

        # Calculation
        seconds_per_year = days_per_year * hours_per_day * minutes_per_hour * seconds_per_minute
        return seconds_per_year

    def update(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[self.vocab[token]] = token
        return

    def train(self, if_train):
        self.if_train = if_train
        return

    def __call__(self, input, window_length, max_length=None, padding=False, truncation=False, return_tensors='pt'):
        seq_len = [len(input[i]['d_value']) for i in range(len(input))]
        window_length = window_length / self.normalization
        if max_length == 'longest':
            max_length = max(seq_len)
        data = []
        attention_mask = []
        for i in range(len(input)):
            input_i = input[i]
            ts_i = input_i['ts_normalized']
            if len(ts_i) > 0:
                ts_start_i = ts_i[0]
                ts_i = [0] + (np.diff(ts_i) / window_length).tolist()
                ts_start_i = [ts_start_i] * len(ts_i)
            else:
                ts_start_i = []
            d_value_i = input_i['d_value']
            d_info_i = self.tokenize(input_i)
            d_info_i = [self.convert_token_to_id(d_info_i[j]) for j in range(len(d_info_i))]
            if truncation and max_length is not None and max_length < seq_len[i]:
                ts_start_i = ts_start_i[:max_length]
                ts_i = ts_i[:max_length]
                d_value_i = d_value_i[:max_length]
                d_info_i = d_info_i[:max_length]
                seq_len[i] = len(d_value_i)
            if padding and max_length is not None and max_length > seq_len[i]:
                pad_width = max_length - seq_len[i]
                if self.padding_direction == 'right':
                    ts_start_i = np.pad(ts_start_i, (0, pad_width), mode='constant', constant_values=0).tolist()
                    ts_i = np.pad(ts_i, (0, pad_width), mode='constant', constant_values=0).tolist()
                    d_value_i = np.pad(d_value_i, (0, pad_width), mode='constant', constant_values=0).tolist()
                    d_info_i = np.pad(d_info_i, (0, pad_width), mode='constant',
                                      constant_values=self.convert_token_to_id(self.pad_token)).tolist()
                    attention_mask_i = [1] * seq_len[i] + [0] * pad_width
                elif self.padding_direction == 'left':
                    ts_start_i = np.pad(ts_start_i, (pad_width, 0), mode='constant', constant_values=0).tolist()
                    ts_i = np.pad(ts_i, (pad_width, 0), mode='constant', constant_values=0).tolist()
                    d_value_i = np.pad(d_value_i, (pad_width, 0), mode='constant', constant_values=0).tolist()
                    d_info_i = np.pad(d_info_i, (pad_width, 0), mode='constant',
                                      constant_values=self.convert_token_to_id(self.pad_token)).tolist()
                    attention_mask_i = [0] * pad_width + [1] * seq_len[i]
                else:
                    raise ValueError('Not valid padding direction')
            else:
                attention_mask_i = [1] * seq_len[i]
            data_i = np.array([ts_start_i, ts_i, d_value_i, d_info_i]).transpose().tolist()
            data.append(data_i)
            attention_mask.append(attention_mask_i)
        if return_tensors == 'np':
            data = np.array(data, dtype=np.float32)
            attention_mask = np.array(attention_mask, dtype=bool)
        if return_tensors == 'pt':
            data = torch.tensor(data, dtype=torch.float32)
            attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        output = {'data': data, 'attention_mask': attention_mask}
        return output

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
