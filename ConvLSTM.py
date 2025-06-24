
import csv

import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data #将数据分批次需要用到它
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    # return torch.device('cpu')
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # 拼接输入张量和隐藏状态
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# 定义 ConvLSTM 模型
class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_classes, num_layers=1, bias=True):
        super(ConvLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            self.cells.append(ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias))
            input_dim = hidden_dim

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, _ = x.size()

        # 将特征维度视为一个高度为1、宽度为特征维度的图像
        x = x.unsqueeze(3).unsqueeze(3)  # (batch_size, seq_len, 1, feature_dim)

        # 初始化隐藏状态和细胞状态
        h, c = self.init_hidden(batch_size, (1, 1))

        for t in range(seq_len):
            for layer_idx in range(self.num_layers):
                h, c = self.cells[layer_idx](x[:, t, :, :, :], (h, c))

        # 取最后一个时间步的隐藏状态
        output = self.fc(h.view(batch_size, -1))
        return output

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden(batch_size, image_size))
        return init_states[0]


if __name__ == '__main__':

    feature_dim = 35  # 每个时间步的特征维度
    # HIDDEN_DIM = 64  # 隐藏状态的维度
    # KERNEL_SIZE = 3  # 卷积核大小
    NUM_CLASSES = 2  # 分类数
    NUM_LAYERS = 2  # LSTM 层的数量
    BATCH_SIZE = 32  # 批次大小
    sequence_length = 10  # 序列长度

    n_result = 3

    model = ConvLSTMModel(input_dim=feature_dim, hidden_dim=feature_dim, kernel_size=7, num_classes=NUM_CLASSES, num_layers=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    # 如果GPU可用，将模型移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # device = torch.device("cpu")
# ----------------------------------------加载数据---------------------------------------------
    db_list = [2, 0, -2, -4, -6, -8, -10, -12]
    for db in db_list:
        root_path = r"/home/c220/Documents/xxx/data/MIMII_gear/noise/" + str(db) + r"db/"
        acc_array = []
        re_array = []
        F1_array = []

        csvWrite = csv.writer(open('test.csv', mode='a', encoding="utf-8-sig", newline=""))
        csvWrite.writerow([root_path, "accuracy", "precision", "recall", "f1", db])

        start = time.time()
        # **********************MIMII Gearbox*******************************

        normal_train = np.load(root_path + r'normal_train_50.npy')
        abnormal_train = np.load(root_path + r'abnormal_train_50.npy')

        normal_test = np.load(root_path + r'normal_test_50.npy')
        abnormal_test = np.load(root_path + r'abnormal_test_50.npy')

        normal_train = normal_train[0:abnormal_train.shape[0]]

        label0 = np.full((normal_train.shape[0], 1), 0)
        label1 = np.full((abnormal_train.shape[0], 1), 1)

        label0_test = np.full((normal_test.shape[0], 1), 0)
        label1_test = np.full((abnormal_test.shape[0], 1), 1)

        x_data = np.concatenate((normal_train, abnormal_train), axis=0)
        y_data = np.concatenate((label0, label1), axis=0)

        x_validation = np.concatenate((normal_test, abnormal_test), axis=0)
        y_validation = np.concatenate((label0_test, label1_test), axis=0)
        # **********************End*******************************

        for i in range(n_result):
            random_indices = np.random.permutation(x_data.shape[0])
            X_data = x_data[random_indices]
            Y_data = y_data[random_indices]

            line = round(len(X_data)*0.7)
            xtrain = X_data[0:line]
            ytrain = Y_data[0:line]

            xtest = X_data[line:]
            ytest = Y_data[line:]

            print(x_data.shape)
            print(y_data.shape)

            mean = np.mean(xtrain, axis=0)
            std = np.std(xtrain, axis=0)
            xtrain = (xtrain - mean) / std
            xtest = (xtest - mean) / std

            xtrain = torch.tensor(xtrain, dtype=torch.float32)
            ytrain = torch.tensor(ytrain, dtype=torch.float32)

            xtest = torch.tensor(xtest, dtype=torch.float32)
            ytest = torch.tensor(ytest, dtype=torch.float32)

            train_dataset = Data.TensorDataset(xtrain, ytrain)  # 将x,y读取，转换成Tensor格式
            test_dataset = Data.TensorDataset(xtest, ytest)  # 将x,y读取，转换成Tensor格式

            train_loader = Data.DataLoader(
                dataset=train_dataset,  # torch TensorDataset format
                batch_size=BATCH_SIZE,  # 最新批数据
                shuffle=True,  # 是否随机打乱数据
                num_workers=2,  # 用于加载数据的子进程
                drop_last=True
            )
            test_loader = Data.DataLoader(
                dataset=test_dataset,  # torch TensorDataset format
                batch_size=BATCH_SIZE,  # 最新批数据
                shuffle=True,  # 是否随机打乱数据
                num_workers=2,  # 用于加载数据的子进程
                drop_last=True
            )

            data_vali = (x_validation - mean) / std
            data_vali = torch.tensor(data_vali, dtype=torch.float32)
            label_vali = torch.tensor(y_validation, dtype=torch.float32)

            test1_dataset = Data.TensorDataset(data_vali, label_vali)

            test1_loader = Data.DataLoader(
                dataset=test1_dataset,  # torch TensorDataset format
                batch_size=BATCH_SIZE,  # 最新批数据
                shuffle=False,  # 是否随机打乱数据
                num_workers=2,  # 用于加载数据的子进程
                drop_last=True
            )

            # 训练和验证函数
            def accuracy(preds, y):
                preds = torch.argmax(preds, dim=1)
                correct = (preds == y).float()
                acc = correct.sum() / len(correct)
                return acc

            def train(model, iterator, optimizer, criterion):
                epoch_loss = 0
                model.train()
                for step, (batch_x, batch_y) in enumerate(iterator):  # 每个训练步骤
                    batch_x, batch_y = batch_x.to(try_gpu()), batch_y.to(try_gpu())
                    batch_y = np.squeeze(batch_y.long())  # 改变维度
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    # acc = binary_accuracy(outputs, batch_y)
                    print('db:','%02d' % (db), 'c:', '%02d' % (i),
                          'Epoch:', '%04d' % (epoch + 1), 'Step:', '%04d' % (step + 1), 'loss =', '{:.6f}'.format(loss))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                return epoch_loss / len(iterator)

            def evaluate(model, iterator, criterion):
                epoch_loss = 0
                for step, (batch_x, batch_y) in enumerate(iterator):  # 每个训练步骤
                    batch_x, batch_y = batch_x.to(try_gpu()), batch_y.to(try_gpu())
                    batch_y = np.squeeze(batch_y.long())  # 改变维度
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    # acc = binary_accuracy(outputs, batch_y)
                    print('Epoch:', '%04d' % (epoch + 1), 'Step:', '%04d' % (step + 1), 'loss =', '{:.6f}'.format(loss))
                    epoch_loss += loss.item()
                return epoch_loss / len(iterator)

            # 训练循环
            N_EPOCHS = 15

            for epoch in range(N_EPOCHS):
                train_loss = train(model, train_loader, optimizer, criterion)
                val_loss = evaluate(model, test_loader, criterion)
                print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}')

            # 验证集
            model.eval()    # 设置为评估模式，不进行梯度更新
            # 对待测信号进行分类
            predicted_test = []
            y_true = []
            with torch.no_grad():
                test_corr = 0
                for step, (batch_x3, batch_y3) in enumerate(test1_loader):  # 每个训练步骤
                    batch_x3, batch_y3 = batch_x3.to(try_gpu()), batch_y3.to(try_gpu())
                    outputs3 = model(batch_x3)
                    batch_y3 = np.squeeze(batch_y3.long())
                    _, predicted3 = torch.max(outputs3.data, 1)
                    y_true.extend(batch_y3.cpu().numpy())
                    predicted_test.extend(predicted3.cpu().numpy())
                    test_corr += torch.sum(predicted3 == batch_y3)
                test1_acc = 100.0 * test_corr / len(test1_dataset)
                # print('Accuracy: {:.2f}%'.format(test1_acc))
                accuracy = accuracy_score(y_true, predicted_test)
                precision = precision_score(y_true, predicted_test, average='macro')
                recall = recall_score(y_true, predicted_test, average='macro')
                f1 = f1_score(y_true, predicted_test, average='macro')

            print(accuracy)
            print(precision)
            print(recall)
            print(f1)
            csvWrite.writerow([time.strftime('%H:%M:%S', time.localtime(time.time())), accuracy, precision, recall, f1])


