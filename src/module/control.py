from config import cfg


def process_control():
    if cfg['control_name'] is None:
        return
    control = cfg['control_name'].split('-')
    cfg['data_name'] = control[0]
    if cfg['data_name'] == 'SmartHome':
        cfg['subset_name'] = control[1]
        # New logic: Check for device name specification
        # e.g., SmartHome-hh105-all will use all devices
        if len(control) > 2:
            cfg['device_name'] = control[2] # 应该是第3个元素，如 'all' 或 'LS005'
        else:
            cfg['device_name'] = 'all' # 默认加载所有
    elif cfg['data_name'] == '2011LS017':
        cfg['subset_name'] = control[1]
    return 