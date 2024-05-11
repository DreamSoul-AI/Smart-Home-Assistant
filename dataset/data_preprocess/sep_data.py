# -*- coding: utf-8 -*-
import os
import pandas as pd


def sep_data_by_d_name(directory, extension):
    processed_data_path = os.path.join(directory, 'processed_data')
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    env_path = next((str(i) for i in file_paths if 'env' in i), None)
    file_paths.remove(env_path)
    env = pd.read_csv(env_path)
    env = env[env['level'] != 1]
    d_name_list = list(env['d_name'])

    for data_path_i in file_paths:
        data = pd.read_csv(str(data_path_i))
        for d_name in d_name_list:
            data_i = data[data['d_name'] == d_name]
            if not data_i.empty:
                save_path = os.path.join(processed_data_path, os.path.basename(data_path_i).replace('.csv', '') + '-' + d_name)
                data_i.to_csv(save_path, index=False)


if __name__ == '__main__':
    sep_data_by_d_name('hh105', 'csv')







