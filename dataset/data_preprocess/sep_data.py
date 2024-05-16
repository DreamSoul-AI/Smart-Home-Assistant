# -*- coding: utf-8 -*-
import os
import pandas as pd


def sep_data_by_d_name(path, time_range, extension='csv'):
    time_range_data_path = os.path.join(path, time_range)
    env_path = os.path.join(path, 'env.csv')
    env = pd.read_csv(env_path)
    env = env[env['level'] != 1]
    d_name_list = list(env['d_name'])

    filenames = os.listdir(time_range_data_path)
    for filename in filenames:
        if filename.split('.')[-1] == extension:
            data = pd.read_csv(os.path.join(time_range_data_path, filename))
            for d_name in d_name_list:
                data_i = data[data['d_name'] == d_name]
                if not data_i.empty:
                    save_path_i = os.path.join(time_range_data_path, 'device', '{}.csv'.format(d_name))
                    data_i.to_csv(save_path_i, index=False)
    return

if __name__ == '__main__':
    sep_data_by_d_name('hh105', 'year')
