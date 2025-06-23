import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data #将数据分批次需要用到它
class AdaptiveWindowController(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.size_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.step_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = x.mean(dim=1)  # [B, D]
        alpha = self.size_mlp(h)
        beta = self.step_mlp(h)
        return alpha, beta

class SwinBlock(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.window_size = window_size

    def forward(self, x):
        B, L, C = x.shape
        if L % self.window_size != 0:
            pad_len = self.window_size - (L % self.window_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            L = x.shape[1]

        windows = x.unfold(1, self.window_size, self.window_size)
        windows = windows.contiguous().view(-1, self.window_size, C)
        out, _ = self.attn(windows, windows, windows)
        out = out.reshape(B, -1, C)
        x = x[:, :out.shape[1], :]  # match output shape
        x = x + self.norm1(out)
        x = x + self.norm2(self.mlp(x))
        return x

class ATSwinTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim=64, depth=5, window_size=27):
        super().__init__()
        self.embed = nn.Linear(input_channels, embed_dim)
        self.adaptive = AdaptiveWindowController(embed_dim)
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, heads=4, window_size=window_size) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, T, C]
        x = self.embed(x)  # [B, T, D]
        alpha, beta = self.adaptive(x)  #
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


n_result = 50
db_list = [-10]
csvWrite = csv.writer(open('test.csv', mode='a', encoding="utf-8-sig", newline=""))

for db in db_list:
    root_path = r"/home/c220/Documents/xxx/data/MIMII_gear/noise/" + str(db) + "db/"
    csvWrite.writerow([root_path, "accuracy", "precision", "recall", "f1"])
    for i in range(n_result):

        data_normal = np.load(root_path + r'normal_train.npy')
        data_abnormal_25 = np.load(root_path + r'abnormal_25_train.npy')
        data_abnormal_50 = np.load(root_path + r'abnormal_50_train.npy')
        data_abnormal_75 = np.load(root_path + r'abnormal_75_train.npy')
        data_abnormal_25_75 = np.load(root_path + r'abnormal_25_75_train.npy')
        test_normal = np.load(root_path + r'normal_test.npy')
        test_abnormal_25 = np.load(root_path + r'abnormal_25_test.npy')
        test_abnormal_50 = np.load(root_path + r'abnormal_50_test.npy')
        test_abnormal_75 = np.load(root_path + r'abnormal_75_test.npy')
        test_abnormal_25_75 = np.load(root_path + r'abnormal_25_75_test.npy')
        label0 = np.full((data_normal.shape[0], 1), 0)
        label1 = np.full((data_abnormal_25.shape[0], 1), 1)
        label2 = np.full((data_abnormal_50.shape[0], 1), 2)
        label3 = np.full((data_abnormal_75.shape[0], 1), 3)
        label4 = np.full((data_abnormal_25_75.shape[0], 1), 4)

        test_label0 = np.full((test_normal.shape[0], 1), 0)
        test_label1 = np.full((test_abnormal_25.shape[0], 1), 1)
        test_label2 = np.full((test_abnormal_50.shape[0], 1), 2)
        test_label3 = np.full((test_abnormal_75.shape[0], 1), 3)
        test_label4 = np.full((test_abnormal_25_75.shape[0], 1), 4)

        x_data = np.concatenate(
            (data_normal, data_abnormal_25, data_abnormal_50, data_abnormal_75, data_abnormal_25_75), axis=0)
        y_data = np.concatenate((label0, label1, label2, label3, label4), axis=0)

        x_validation = np.concatenate(
            (test_normal, test_abnormal_25, test_abnormal_50, test_abnormal_75, test_abnormal_25_75), axis=0)
        y_validation = np.concatenate((test_label0, test_label1, test_label2, test_label3, test_label4), axis=0)

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

        BATCH_SIZE = 32
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

        # 模型训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ATSwinTransformer(35, 5).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        total_step = len(train_loader)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        model.train()
        for epoch in range(15):  # 可改为200
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                loss = criterion(out, batch_y.view(-1).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / total_step:.4f}")

        # 模型评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(batch_y)

        # 拼接并计算指标
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"\n✅ Evaluation:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        csvWrite.writerow([time.strftime('%H:%M:%S', time.localtime(time.time())), acc, prec, rec, f1])

