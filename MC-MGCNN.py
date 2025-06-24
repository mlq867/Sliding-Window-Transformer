import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.size()  # batch_size: 32, num_nodes: 10
        adj = torch.zeros((batch_size * num_nodes, batch_size * num_nodes), device=x.device)

        for i in range(batch_size):
            src = edge_index[0, :, i] + (num_nodes * i)  # 使用每个批次的节点偏移
            dst = edge_index[1, :, i] + (num_nodes * i)

            assert src.max() < batch_size * num_nodes, f"src index {src.max()} out of range for adj of size {adj.size()}"
            assert dst.max() < batch_size * num_nodes, f"dst index {dst.max()} out of range for adj of size {adj.size()}"

            adj[src, dst] = 1

        # 添加自环
        adj += torch.eye(adj.size(0), device=x.device)

        # 归一化邻接矩阵
        D = adj.sum(dim=1)
        D_inv_sqrt = D.pow(-0.5)  # D的平方根的倒数
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0  # 处理D为0的情况
        adj_normalized = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)  # D^-1/2 * A * D^-1/2

        # 通过线性层
        x = self.linear(adj_normalized @ x.view(-1, x.size(2)))  # 矩阵乘法与线性层结合
        return x.view(batch_size, num_nodes, -1)


# 2. 定义多通道多图卷积神经网络
class MC_MGCNN(nn.Module):
    def __init__(self):
        super(MC_MGCNN, self).__init__()
        self.conv1 = GraphConvLayer(35, batch_size)  # 第一层图卷积
        self.conv2 = GraphConvLayer(batch_size, batch_size)  # 第二层图卷积
        self.fc = nn.Linear(32, 2)  # 输出层

    def forward(self, x, edge_index):
        # 这里是图卷积的前向传播
        src, dst = edge_index[0], edge_index[1]

        # 打印索引以调试
        # print(f"src: {src}, dst: {dst}")

        num_nodes = x.size(0)  # 获取节点数
        assert src.max() < num_nodes, "src index out of range"
        assert dst.max() < num_nodes, "dst index out of range"

        adj = torch.zeros(num_nodes, num_nodes).to(x.device)  # 创建邻接矩阵
        adj[src, dst] = 1

        # 归一化邻接矩阵（如果有需要）
        # ...

        x = F.relu(self.conv1(x, edge_index))  # 通过第一个卷积层
        x = F.relu(self.conv2(x, edge_index))  # 通过第二个卷积层
        x = x.mean(dim=1)  # 对每个图的节点特征取平均
        x = self.fc(x)  # 输入到全连接层
        return x




# 3. 创建自定义数据集
class MyDataset(Data.Dataset):
    def __init__(self, npy_files):
        super(MyDataset, self).__init__()
        self.data_list = []

        for label, npy_file in enumerate(npy_files):
            data_array = np.load(npy_file)  # 加载数据
            for sample in data_array:
                # 假设sample形状为 (10, 35)，每个样本的特征
                node_features = torch.tensor(sample, dtype=torch.float)  # 转换为张量

                # 创建边索引，这里我们假设每个时间步都与其他时间步连接
                num_time_steps = sample.shape[0]  # 应为 10
                edge_index = []
                for i in range(num_time_steps):
                    for j in range(num_time_steps):
                        if i != j:  # 排除自环
                            edge_index.append([i, j])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置并连续

                labels = torch.tensor([label], dtype=torch.long)  # 使用文件索引作为标签
                self.data_list.append((node_features, edge_index, labels))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# 4. 训练过程
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for node_features, edge_index, labels in data_loader:
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device).view(-1)

        # print("Node features shape:", node_features.shape)  # [32, 10, 35]
        # print("Edge index shape:", edge_index.shape)  # [32, 2, 90]

        optimizer.zero_grad()
        output = model(node_features, edge_index)

        # labels 的形状应为 [32]，output 的形状应为 [32, num_classes]
        loss = criterion(output, labels)  # 这里可以直接使用，不需要 view 操作
        loss.backward()
        optimizer.step()
    return loss.item()

batch_size = 32
n_result = 3
db_list = [2, 0, -2, -4, -6, -8, -10, -12]
for db in db_list:
    root_path = r"/home/c220/Documents/xxx/data/MIMII_gear/noise/" + str(db) + r"db/"
    csvWrite = csv.writer(open('test.csv', mode='a', encoding="utf-8-sig", newline=""))
    csvWrite.writerow([root_path, "accuracy", "precision", "recall", "f1", "MC-MGCNN"])
    for i in range(n_result):
        # 5. 创建数据集和数据加载器
        npy_files = [root_path + r'normal_train_50.npy', root_path + r'abnormal_train_50.npy']  # 替换为你的.npy文件路径
        dataset = MyDataset(npy_files)
        # train_loader = Data.DataLoader(dataset, batch_size=32, shuffle=True)
        train_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.cat([item[1].unsqueeze(0) for item in batch], dim=0),  # 适当处理 edge_index
            torch.stack([item[2] for item in batch])
        ))

        npy_test = [root_path + r'normal_test_50.npy', root_path + r'abnormal_test_50.npy']  # 替换为你的.npy文件路径
        dataset_test = MyDataset(npy_test)
        # test_loader = Data.DataLoader(dataset_test, batch_size=32, shuffle=True)
        test_loader = Data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.cat([item[1].unsqueeze(0) for item in batch], dim=0),  # 适当处理 edge_index
            torch.stack([item[2] for item in batch])
        ))

        # 6. 初始化模型、优化器和损失函数
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MC_MGCNN().to(device)  # 五分类
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        # 7. 训练模型
        num_epochs = 15
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, criterion, device)
            print('db:', '%02d' % (db), 'c:', '%02d' % (i),
                  'Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))


        # 8. 模型评估
        def evaluate(model, data_loader, device):
            model.eval()
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for node_features, edge_index, labels in data_loader:
                    node_features, edge_index, labels = node_features.to(device), edge_index.to(device), labels.to(device)
                    output = model(node_features, edge_index)
                    _, predicted = torch.max(output, dim=1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            return accuracy, precision, recall, f1


        # 评估模型
        accuracy, precision, recall, f1 = evaluate(model, train_loader, device)
        print(f'Train Accuracy: {accuracy:.9f}, Precision: {precision:.9f}, Recall: {recall:.9f}, F1 Score: {f1:.9f}')
        csvWrite.writerow([time.strftime('%H:%M:%S', time.localtime(time.time())), accuracy, precision, recall, f1])
