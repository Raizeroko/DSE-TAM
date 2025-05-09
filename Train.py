from torch import nn
from Dataset.LOSO import *
from Dataset.KFold import *
from ChooseNet import *
from utils import *
from sklearn.metrics import confusion_matrix

def evaluate_model(model, dataloader, lossFunction, device):
    model.eval()  # 将模型设为评估模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = lossFunction(outputs, labels)
            # 统计损失
            running_loss += loss.item() * inputs.size(0)
            # 统计正确预测的样本数量
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # 保存真实标签和预测标签
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_preds / total_preds

    labels_range = list(range(3))
    cm = confusion_matrix(all_labels, all_predictions, labels=labels_range)

    return avg_loss, accuracy, cm


def train_and_validation(net, train_iter, test_iter, num_epochs, lr, weight_decay, device):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.99)
    net.to(device)
    new_lr = None
    state = None
    max_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []  # 用于存储每个 epoch 的训练损失
    val_cm = []


    for epoch in range(num_epochs):
        net.train()  # 将模型设为训练模式
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.reshape(32, 5, 62, 5)
            # labels = labels.reshape(32, 5)
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = lossFunction(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计损失
            running_loss += loss.item() * inputs.size(0)
            # 统计正确预测的样本数量
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_acc = correct_preds / total_preds
        epoch_loss = running_loss / len(train_iter.dataset)
        print(f'Train Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        val_loss, val_acc, cm = evaluate_model(net, test_iter, lossFunction, device)
        print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_cm.append(cm)

    val_cm = np.array(val_cm)
    return train_losses, val_losses, val_accs, val_cm


def train_by_LOSO(params):
    torch.autograd.set_detect_anomaly(True)
    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []
    sub_cm = []

    subjects = params['subjects']
    print(f"Starting LOSO training for {subjects} subjects.")
    for i in range(1, subjects + 1):
        print(f"Training for Subject {i}")
        net = choose_net(params)

        train_dataset, test_dataset = Dataset_LOSO(params['dataset'], params['data_dir'], params['session'],
                                                   params['subjects'], params['trial'], i)

        loader_train = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )
        loader_test = Data.DataLoader(
            dataset=test_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )

        print("============================================================================")
        print(f"Subject: {i}")
        train_losses, val_losses, val_accs, val_cm= train_and_validation(net,
                                                                          loader_train,
                                                                          loader_test,
                                                                          num_epochs=params['epoch'],
                                                                          lr=params['lr'],
                                                                          weight_decay=params['weight_decay'],
                                                                          device=params['device'])

        sub_train_loss.append(train_losses)
        sub_val_loss.append(val_losses)
        sub_val_acc.append(val_accs)
        sub_cm.append(val_cm)


    # 避免params中的device在保存时出错
    params_copy = params.copy()
    if 'device' in params_copy:
        del params_copy['device']

    results = {
        'train_loss': sub_train_loss,
        'val_loss': sub_val_loss,
        'val_acc': sub_val_acc,
        'params': params_copy,
        'cm': sub_cm
    }
    return results


def train_by_KFold(params):
    torch.autograd.set_detect_anomaly(True)

    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []

    sub_cm = []
    subjects = params['subjects']
    print(f"Starting KFold training for {subjects} subjects with {params['KFold']} folds.")

    for i in range(1, subjects + 1):
        fold_train_loss, fold_val_loss, fold_val_acc = [], [], []
        fold_cm = []

        print(f"Training for Subject {i}")

        for fold in range(params['KFold']):
            net = choose_net(params)
            if params['shuffle'] == 'Sample':
                train_dataset, test_dataset = Dataset_KFold_Sample(params['dataset'], params['data_dir'],
                                                                   params['session'], i, params['trial'],
                                                                   params['KFold'], fold)
            elif params['shuffle'] == 'Trial':
                train_dataset, test_dataset = Dataset_KFold_Trial(params['dataset'], params['data_dir'],
                                                                  params['session'], i, params['trial'],
                                                                  params['KFold'], fold)


            loader_train = Data.DataLoader(
                dataset=train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0
            )
            loader_test = Data.DataLoader(
                dataset=test_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0
            )
            print("============================================================================")
            print(f"Subject: {i}   Fold: {fold}")
            train_losses, val_losses, val_accs, val_cm = train_and_validation(net,
                                                                                  loader_train,
                                                                                  loader_test,
                                                                                  num_epochs=params['epoch'],
                                                                                  lr=params['lr'],
                                                                                  weight_decay=params['weight_decay'],
                                                                                  device=params['device'])

            fold_train_loss.append(train_losses)
            fold_val_loss.append(val_losses)
            fold_val_acc.append(val_accs)
            fold_cm.append(val_cm)


        sub_train_loss.append(fold_train_loss)
        sub_val_loss.append(fold_val_loss)
        sub_val_acc.append(fold_val_acc)
        sub_cm.append(fold_cm)


    # 避免params中的device在保存时出错
    params_copy = params.copy()
    if 'device' in params_copy:
        del params_copy['device']

    results = {
        'train_loss': sub_train_loss,
        'val_loss': sub_val_loss,
        'val_acc': sub_val_acc,
        'params': params_copy,
        'cm': sub_cm,
    }
    return results



# 代码运行开始-设置参数
params = {
        # -----------网络参数-------------------------------------------------------
        'emb_dim': 48,  # embedding dimension of Embedding, Self-Attention, Mamba
        'emb_kernel': 4,  # 2D-conv embedding length of Embedding
        'pool': 4,  # pool kernel size
        'd_state': 16,  # d_state of Mamba2
        'd_conv': 4,  # d_conv of Mamba2
        'expand': 8,  # expand of Mamba2
        'num_layers': 2,  # layer of MambaFormer
        'num_heads': 8,  # num head of Self-Attention
        'g_layers': 4,  # graph embedding layers
        'dropout': 0.5,  # dropout of Embedding, Self-Attention, Mamba
        'lr': 1e-4,  # learning rate
        'weight_decay': 1e-6,  # L2-norm
        'time': 384,  # window size of EEG
        # -----------训练参数-------------------------------------------------------
        'seed': 20, # random seed
        'epoch': 500,  # training epoch
        'batch_size': 256,  # training batch size
        'session': 1,  # dataset session: 1/2 (DEAP:Arousal/Valence, DREAMER:Arousal/Valence, CEED)
        'val': "KFold",  # experiment validation：LOSO/KFold
        'shuffle': 'Trial',  # validation shuffle way: Sample/Trial
        'net': "DSE-TAM",  # Choose net：ACRNN/DSE-TAM/Conformer/Deformer/DGCNN/LGGNet/TSception
        'dataset': 'DEAP',  # choose dataset: DEAP/DREAMER/CEED
        'feature': "Time",  # input feature: Time/DE
}

# 定义一个训练函数映射，根据 params['val'] 选择相应的训练方式
train_funcs = {
    "LOSO": train_by_LOSO,
    "KFold": train_by_KFold
}


if __name__ == '__main__':
    setup_seed(params['seed'])
    params = init_params(params)

    # 根据 params['val'] 选择对应的训练函数，并执行
    train_func = train_funcs.get(params['val'])
    if train_func:
        results = train_func(params)
        save_results(params, results)