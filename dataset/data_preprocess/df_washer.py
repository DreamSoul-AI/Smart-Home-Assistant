import re
import pandas as pd


class DataframeWasher:
    def __init__(self, wash_dict):
        self.wash_dict = wash_dict
        self.wash_step_len = len(wash_dict)

    def wash(self, df):
        for step in range(self.wash_step_len):
            task_info = self.wash_dict[step]
            action_type = task_info['action_type']
            sub_action_type = task_info['sub_action_type']
            column_name = task_info['column_name']
            condition = task_info['condition']
            input_value = task_info['input_value']
            method_name = 'action_{}_{}'.format(action_type, sub_action_type)
            if 'c_' in sub_action_type:
                method_to_call = getattr(self, method_name, None)
                if callable(method_to_call):
                    df = method_to_call(df, column_name, input_value, condition)
                    # print('============={}============='.format(step))
                    # print(df)
            else:
                method_to_call = getattr(self, method_name, None)
                if callable(method_to_call):
                    df = method_to_call(df, column_name, input_value)
                    # print('============={}============='.format(step))
                    # print(df)
        return df

    def action_drop_equal(self, df, column_name, input_value: list):
        return df[~df[column_name].isin(input_value)]

    def action_drop_contain(self, df, column_name, input_value: list):
        pattern = '|'.join(map(re.escape, input_value))
        return df[~df[column_name].astype(str).str.contains(pattern)]

    def action_replace_equal(self, df, column_name, input_value: dict):
        df.loc[:, column_name] = df.loc[:, column_name].replace(input_value)
        return df

    def action_replace_contain(self, df, column_name, input_value: dict):
        for k, v in input_value.items():
            df.loc[:, column_name] = df.loc[:, column_name].astype(str).str.replace(k, v)
        return df

    def action_replace_c_equal(self, df, column_name, input_value, condition: dict):
        c_column_name = condition['c_column_name']
        c_type = condition['c_type']
        c_value1 = condition['c_value1']
        # c_value2 = condition['c_value2']

        if c_type == 'contain':
            pattern = '|'.join(map(re.escape, c_value1))
            df.loc[df[c_column_name].astype(str).str.contains(pattern), column_name] = input_value
            return df

        if c_type == 'equal':
            df.loc[df[c_column_name].isin(c_value1), column_name] = input_value
            return df

    def action_trans_value_type(self, df, column_name, input_value):
        df[column_name] = df[column_name].astype(input_value)
        return df

    def action_drop_cant_trans_to_num(self, df, column_name, input_value):
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        df.dropna(subset=[column_name], inplace=True)
        return df


if __name__ == '__main__':
    wash_dict_example = {
        0: {
            'action_type': 'drop',
            'sub_action_type': 'contain',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': ['TAP_COUNT', 'HOLD_DEPRESS', 'HOLD_RELEASE', 'RELEASE']
        },
        1: {
            'action_type': 'replace',
            'sub_action_type': 'equal',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': {"ON": 1.0, "OFF": 0.0, "OPEN": 1.0, "CLOSE": 0.0}
        },
        2: {
            'action_type': 'drop',
            'sub_action_type': 'contain',
            'column_name': 'd_name',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': ['BATP', 'ZB', 'HOME']
        },
        3: {
            'action_type': 'drop',
            'sub_action_type': 'equal',
            'column_name': 'd_name',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': ['c'],
            'target_value': []
        },
        4: {
            'action_type': 'replace',
            'sub_action_type': 'c_equal',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': 'd_name',
                'c_type': 'contain',
                'c_value1': ['ButtonDown'],
                'c_value2': None
            },
            'input_value': 0.0
        },
        5: {
            'action_type': 'replace',
            'sub_action_type': 'c_equal',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': 'd_name',
                'c_type': 'contain',
                'c_value1': ['ButtonUp'],
                'c_value2': None
            },
            'input_value': 1.0
        },
        6: {
            'action_type': 'replace',
            'sub_action_type': 'contain',
            'column_name': 'd_name',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': {'ButtonUp': 'Button', 'ButtonDown': 'Button'}
        },
        7: {
            'action_type': 'drop',
            'sub_action_type': 'cant_trans_to_num',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': None
        },
        8: {
            'action_type': 'trans',
            'sub_action_type': 'value_type',
            'column_name': 'd_name',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': 'str'
        },
        9: {
            'action_type': 'trans',
            'sub_action_type': 'value_type',
            'column_name': 'd_value',
            'condition': {
                'c_column_name': None,
                'c_type': None,
                'c_value1': None,
                'c_value2': None
            },
            'input_value': float
        }
    }

    test_data0 = {
        'd_value': ['TAP_COUNTAAA', 'HOLD_DEPRESSBBB', 'HOLD_RELEASE', 'RELEAS'],
        'd_name': ['A', 'B', 'C', 'D']
    }

    test_data1 = {
        'd_value': ['ON', 'OFF', 'OPEN', 'CLOSE'],
        'd_name': ['A', 'B', 'C', 'D']
    }

    test_data2 = {
        'd_value': ['A', 'B', 'C'],
        'd_name': ['BATP1', 'ZB22', 'HOME']
    }

    test_data3 = {
        'd_value': ['A', 'B', 'C'],
        'd_name': ['c', 'C', 'c_data']
    }

    test_data4 = {
        'd_value': ['A', 'B', 'C'],
        'd_name': ['ButtondownA', 'ButtonDown008', 'ButtonDown']
    }

    test_data5 = {
        'd_value': ['A', 'B', 'C'],
        'd_name': ['ButtonupA', 'ButtonUp007', 'ButtonUp']
    }

    test_data6 = {
        'd_value': ['A', 'B', 'C', 'D', 'E'],
        'd_name': ['ButtonupA', 'ButtonUp007', 'ButtonUp001', 'ButtonUp002', 'Buttonup']
    }

    test_data7 = {
        'd_value': [1, 2, 3.0],
        'd_name': ['A', 'B', 'C']
    }

    test_data8 = {
        'd_value': ['001', '0.25', '0.006', 'OFF5cc', 'OFF5c', 'ON5c'],
        'd_name': ['A', 'B', 'C', 'D', 'E', 'F']
    }

    # test_dataset = (test_data0, test_data1, test_data2, test_data3, test_data4, test_data5, test_data6)
    test_dataset = [test_data8]



    for i, test_data in enumerate(test_dataset):
        print('\n\n====={}====='.format(i))
        df_test = pd.DataFrame(test_data)
        print("Original DataFrame:")
        print(df_test)

        washer = DataframeWasher(wash_dict_example)
        df = washer.wash(df_test)

        print("\nWashed DataFrame:")
        print(df)

        a =input('pause')