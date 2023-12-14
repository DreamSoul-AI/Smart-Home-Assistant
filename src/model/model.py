import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from config import cfg

# ------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd


def make_vec_lib(data_path, tokenizer):

    env_data = pd.read_csv(data_path, delimiter=',')
    env_data = env_data[(env_data['d_type'] != 'room') & (env_data['d_type'] != 'house')]
    env_data = env_data[['d_name', 'd_func', 'd_type']]
    env_data = env_data.reset_index()
    d_name, d_func, d_type = env_data['d_name'], env_data['d_func'], env_data['d_type']
    env_str_list = []
    for i in range(len(env_data)):
        device_j = 'Name: {}, Function: {}, Info: {}'.format(d_name[i], d_func[i], d_type[i])
        env_str_list.append(device_j)
    vec_lib = tokenizer.encode(env_str_list)  # vectors_lib
    env_str_array = np.array(env_str_list)
    if not os.path.exists(os.path.join('output', 'vec_lib')):
        os.makedirs(os.path.join('output', 'vec_lib'))
    np.save(os.path.join('output', 'vec_lib', 'vec_lib.npy'), vec_lib)
    np.save(os.path.join('output', 'vec_lib', 'env_str_array.npy'), env_str_array)
    return


def query_vec_lib(query_vector):
    vec_lib = np.load(os.path.join('output', 'vec_lib', 'vec_lib.npy'))
    env_str_list = np.load(os.path.join('output', 'vec_lib', 'env_str_array.npy')).tolist()
    similarities = np.dot(query_vector, np.transpose(vec_lib))
    most_similar_index = np.argmax(similarities)
    most_similar_sentence = env_str_list[most_similar_index]
    return most_similar_sentence


# ------------------------------------------------------------------------------------------------------



def make_model(model_name):
    model_ = eval('model.{}()'.format(model_name))
    tokenizer = model.Tokenizer()
    return model_, tokenizer


def make_loss(output, input):
    if 'target' in input:
        loss = loss_fn(output['target'], input['target'])
    else:
        return
    return loss


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = kld_loss(output, target, reduction=reduction)
    return loss


def cross_entropy_loss(output, target, reduction='mean'):
    if target.dtype != torch.int64:
        target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction=reduction)
    return ce


def kld_loss(output, target, reduction='batchmean'):
    kld = F.kl_div(F.log_softmax(output, dim=-1), target, reduction=reduction)
    return kld


def init_param(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=cfg['device']))
            m.register_buffer('running_var', torch.ones(m.num_features, device=cfg['device']))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=cfg['device']))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def make_tokenizer(data_loader):
    for i, input in enumerate(data_loader):
        pass
    return


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                                weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler
