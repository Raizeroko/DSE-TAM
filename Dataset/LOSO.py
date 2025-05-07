import os
import torch
import torch.utils.data as Data
import scipy.io as scio

def Dataset_LOSO(dataset, input_dir, session, subjects, trial, target_id):
    data_dir = None
    if dataset == 'SEED' or dataset == 'SEEDIV':
        # SEED
        data_dir = os.path.join(input_dir, f'Session{session}')
    elif dataset == 'DREAMER' or dataset == 'DEAP':
        # DEAP/DREAMER
        if session == 1:
            data_dir = os.path.join(input_dir, 'Arousal')
        elif session == 2:
            data_dir = os.path.join(input_dir, 'Valence')
        elif session == 3:
            data_dir = os.path.join(input_dir, 'Dominance')
    elif dataset == 'CEED':
        data_dir = input_dir

    # 初始话空列表，用于存储从数据文件中读取的特征和标签
    feature_list = []
    label_list = []
    for i in range(subjects):
        file_path = os.path.join(data_dir, f'subject{i+1}.mat')
        data = scio.loadmat(file_path)
        feature_trial = data['feature']
        label_trial = data['label']
        feature = None
        label = None

        # 提取subject i 的feature和label
        for i in range(trial):
            if feature == None:
                feature = torch.tensor(feature_trial[f'trial{i+1}'][0][0])
                label = torch.tensor(label_trial[f'trial{i+1}'][0][0]).reshape(-1)
            else:
                feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i+1}'][0][0])), dim=0)
                label = torch.cat((label, torch.tensor(label_trial[f'trial{i+1}'][0][0]).reshape(-1)), dim=0)

        feature = feature.permute(0, 2, 1).float()
        label = label.long()

        feature_list.append(feature)
        label_list.append(label)


    target_feature, target_label = feature_list[target_id-1], label_list[target_id-1]
    del feature_list[target_id-1]
    del label_list[target_id-1]
    source_feature = torch.cat(feature_list, dim=0).permute(0, 1, 2).float()
    source_label  = torch.cat(label_list, dim=0)
    # 构建目标域数据集和源域数据集并返回

    source_set = {'feature': source_feature.float(), 'label': source_label}
    target_set = {'feature': target_feature.float(), 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset


