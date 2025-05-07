import torch
import numpy as np
import random
import os
import scipy.io as scio

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_params(params):
    path = 'E:/datasets/'
    if os.name == 'posix':  # Linux or macOS
        path = '/home/zwr/dataset/'

    params['time'] = 384
    data = f"{params['dataset']}_{params['feature']}_Preprocessed_{params['time']}"

    if params['dataset'] == 'DEAP':
        params['num_electrodes'] = 32           #DEAP有62给电极通道
        params['num_classes'] = 2               #DEAP是二分类:high, low
        params['subjects'] = 32
        params['trial'] = 40
        params['KFold'] = 10
    elif params['dataset'] == 'DREAMER':
        params['num_electrodes'] = 14
        params['num_classes'] = 2
        params['subjects'] = 23
        params['trial'] = 18
        params['KFold'] = 6

    elif params['dataset'] == 'SEED' or params['dataset'] == 'SEEDIV':
        params['num_electrodes'] = 62           #SEED有62给电极通道
        params['subjects'] = 15
        params['trial'] = 15
        params['KFold'] = 5
        if params['dataset'] == 'SEED':
            params['num_classes'] = 3           #SEED是三分类:positive, negative, neutral
        elif params['dataset'] == 'SEEDIV':
            params['num_classes'] = 4           #SEEDIV是三分类:sad, fear, happy, neutral
    elif params['dataset'] == 'CEED':
        params['num_electrodes'] = 63
        params['num_classes'] = 3
        params['subjects'] = 20
        params['trial'] = 15
        params['KFold'] = 5

    params['data_dir'] = os.path.join(path, data)
    params['device'] = torch.device("cuda:0")  # training device
    return params


def save_results(params, results):
    # 设置文件名的初始后缀数字
    suffix_number = 1
    # 构建文件名
    file_name = f"./results/{params['feature']}/{params['val']}/{params['net']}-{params['dataset']}-{params['session']}-{suffix_number}.mat"
    # 检查文件是否存在，如果存在，则增加后缀数字
    while os.path.exists(file_name):
        suffix_number += 1
        file_name = f"./results/{params['feature']}/{params['val']}/{params['net']}-{params['dataset']}-{params['session']}-{suffix_number}.mat"
    # 执行保存操作
    scio.savemat(file_name, results)