
import csv
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import torch.utils.data as Data
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

        self.maskMat = torch.zeros(BATCH_SIZE, 1, n_sequence, n_sequence, dtype=torch.float)
        for i in range(n_sequence):
            for j in range(n_sequence):
                if abs(i - j) < n_window:
                    self.maskMat[:, :, i, j] = 1

    def forward(self, Q, K, V, mask=False):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        m = self.maskMat == 0
        if mask:
            scores = scores.masked_fill(m.cuda(), float('-inf'))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask= False):

        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class SWHeadAttention(nn.Module):
    def __init__(self):
        super(SWHeadAttention, self).__init__()
        self.heads = n_heads_sw
        self.W_Q = nn.Linear(d_model, d_k * self.heads)
        self.W_K = nn.Linear(d_model, d_k * self.heads)
        self.W_V = nn.Linear(d_model, d_v * self.heads)
        self.linear = nn.Linear(self.heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask= True):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]



## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if d_model % 2 != 0:
            d_model += 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return x


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.sw = SWHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs_sw, attn1 = self.sw(enc_inputs, enc_inputs, enc_inputs, True)
        enc_outputs = self.pos_ffn(enc_outputs + enc_outputs_sw) # enc_outputs: [batch_size x len_q x d_model]
        # enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(input_dim, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 input_dim * d_model
        self.conv1d = nn.Conv1d(n_sequence, n_sequence, kernel_size=1, stride=1, padding=0)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_inputs = self.conv1d(enc_inputs)    # (32, 10, 35)
        enc_outputs = self.pos_emb(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)#(32,10,35)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层
        self.linear1 = nn.Linear(d_model*n_sequence, 256, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, num_class)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)      #形状为[batch_size, src_len]
        x = self.relu1(self.linear1(enc_outputs.view(BATCH_SIZE, d_model*n_sequence)))
        x = self.linear2(x)     #x形状[64,512,2]
        return x


if __name__ == '__main__':
    windows_list = [27]
    db_list = [-10]
    # 23, 25, 7, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49
    for db in db_list:
        for wd in windows_list:
            input_dim = 35  # length of source
            # 模型参数
            d_model = input_dim  # Embedding Size，人工设置d_model的维度
            d_ff = 2048  # FeedForward dimension
            d_k = d_v = 64  # dimension of K(=Q), V
            n_layers = 3  # number of Encoder of Decoder Layer
            n_heads = 4  # number of heads in Multi-Head Attention
            n_sequence = 50
            n_result = 50        # TRM
            n_window = (wd + 1) // 2
            n_heads_sw = 4
            num_class = 2
            epochs = 15
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
        # ----------------------------------------加载数据---------------------------------------------
            root_path = r"/home/c220/Documents/xxx/data/MIMII_gear/noise/" + str(db) + r"db/"
            acc_array = []
            re_array = []
            F1_array = []

            csvWrite = csv.writer(open('test.csv', mode='a', encoding="utf-8-sig", newline=""))
            csvWrite.writerow([root_path, "accuracy", "precision", "recall", "f1", wd])

            start = time.time()
            # **********************MIMII Gearbox*******************************

            normal_train = np.load(root_path + r'normal_train_50.npy')
            abnormal_train = np.load(root_path + r'abnormal_train_50.npy')

            normal_test = np.load(root_path + r'normal_test_50.npy')
            abnormal_test = np.load(root_path + r'abnormal_test_50.npy')

            # normal_train = normal_train[:, :, 12:]
            # abnormal_train = abnormal_train[:, :, 12:]
            # normal_test = normal_test[:, :, 12:]
            # abnormal_test = abnormal_test[:, :, 12:]
            # normal_train = np.concatenate((normal_train[:, :, 0:12], normal_train[:, :, 24:]), axis=-1)
            # abnormal_train = np.concatenate((abnormal_train[:, :, 0:12], abnormal_train[:, :, 24:]), axis=-1)
            # normal_test = np.concatenate((normal_test[:, :, 0:12], normal_test[:, :, 24:]), axis=-1)
            # abnormal_test = np.concatenate((abnormal_test[:, :, 0:12], abnormal_test[:, :, 24:]), axis=-1)


            label0 = np.full((normal_train.shape[0], 1), 0)
            label1 = np.full((abnormal_train.shape[0], 1), 1)

            label0_test = np.full((normal_test.shape[0], 1), 0)
            label1_test = np.full((abnormal_test.shape[0], 1), 1)

            x_data = np.concatenate((normal_train, abnormal_train), axis=0)
            y_data = np.concatenate((label0, label1), axis=0)

            x_validation = np.concatenate((normal_test, abnormal_test), axis=0)
            y_validation = np.concatenate((label0_test, label1_test), axis=0)
            # **********************End***************************************


            # **********************泵（将大数据）*******************************
            # data_normal = np.load(root_path + r'normal_train.npy')
            # data_abnormal_25 = np.load(root_path + r'abnormal_25_train.npy')
            # data_abnormal_50 = np.load(root_path + r'abnormal_50_train.npy')
            # data_abnormal_75 = np.load(root_path + r'abnormal_75_train.npy')
            # data_abnormal_25_75 = np.load(root_path + r'abnormal_25_75_train.npy')
            # test_normal = np.load(root_path + r'normal_test.npy')
            # test_abnormal_25 = np.load(root_path + r'abnormal_25_test.npy')
            # test_abnormal_50 = np.load(root_path + r'abnormal_50_test.npy')
            # test_abnormal_75 = np.load(root_path + r'abnormal_75_test.npy')
            # test_abnormal_25_75 = np.load(root_path + r'abnormal_25_75_test.npy')
            #
            # data_normal = np.concatenate((data_normal[:, :, 0:12], data_normal[:, :, 24:]), axis=-1)
            # data_abnormal_25 = np.concatenate((data_abnormal_25[:, :, 0:12], data_abnormal_25[:, :, 24:]), axis=-1)
            # data_abnormal_50 = np.concatenate((data_abnormal_50[:, :, 0:12], data_abnormal_50[:, :, 24:]), axis=-1)
            # data_abnormal_75 = np.concatenate((data_abnormal_75[:, :, 0:12], data_abnormal_75[:, :, 24:]), axis=-1)
            # data_abnormal_25_75 = np.concatenate((data_abnormal_25_75[:, :, 0:12], data_abnormal_25_75[:, :, 24:]), axis=-1)
            # test_normal = np.concatenate((test_normal[:, :, 0:12], test_normal[:, :, 24:]), axis=-1)
            # test_abnormal_25 = np.concatenate((test_abnormal_25[:, :, 0:12], test_abnormal_25[:, :, 24:]), axis=-1)
            # test_abnormal_50 = np.concatenate((test_abnormal_50[:, :, 0:12], test_abnormal_50[:, :, 24:]), axis=-1)
            # test_abnormal_75 = np.concatenate((test_abnormal_75[:, :, 0:12], test_abnormal_75[:, :, 24:]), axis=-1)
            # test_abnormal_25_75 = np.concatenate((test_abnormal_25_75[:, :, 0:12], test_abnormal_25_75[:, :, 24:]), axis=-1)
            #
            # # data_abnormal_25 = data_abnormal_25[:, :, 12:]
            # # data_abnormal_50 = data_abnormal_50[:, :, 12:]
            # # data_abnormal_75 = data_abnormal_75[:, :, 12:]
            # # data_abnormal_25_75 = data_abnormal_25_75[:, :, 12:]
            # # test_normal = test_normal[:, :, 12:]
            # # test_abnormal_25 = test_abnormal_25[:, :, 12:]
            # # test_abnormal_50 = test_abnormal_50[:, :, 12:]
            # # test_abnormal_75 = test_abnormal_75[:, :, 12:]
            # # test_abnormal_25_75 = test_abnormal_25_75[:, :, 12:]
            #
            # label0 = np.full((data_normal.shape[0], 1), 0)
            # label1 = np.full((data_abnormal_25.shape[0], 1), 1)
            # label2 = np.full((data_abnormal_50.shape[0], 1), 2)
            # label3 = np.full((data_abnormal_75.shape[0], 1), 3)
            # label4 = np.full((data_abnormal_25_75.shape[0], 1), 4)
            #
            # test_label0 = np.full((test_normal.shape[0], 1), 0)
            # test_label1 = np.full((test_abnormal_25.shape[0], 1), 1)
            # test_label2 = np.full((test_abnormal_50.shape[0], 1), 2)
            # test_label3 = np.full((test_abnormal_75.shape[0], 1), 3)
            # test_label4 = np.full((test_abnormal_25_75.shape[0], 1), 4)
            #
            # x_data = np.concatenate((data_normal, data_abnormal_25, data_abnormal_50, data_abnormal_75, data_abnormal_25_75), axis=0)
            # y_data = np.concatenate((label0, label1, label2, label3, label4), axis=0)
            #
            # x_validation = np.concatenate((test_normal, test_abnormal_25, test_abnormal_50, test_abnormal_75, test_abnormal_25_75), axis=0)
            # y_validation = np.concatenate((test_label0, test_label1, test_label2, test_label3, test_label4), axis=0)
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

                def try_gpu(i=0):
                    """如果存在，则返回gpu(i)，否则返回cpu()"""
                    if torch.cuda.device_count() >= i + 1:
                        return torch.device(f'cuda:{i}')
                    # return torch.device('cpu')
            # --------------------------------------------定义参数-----------------------------------------------
                # 存储训练过程中的损失和准确度
                train_losses = []
                train_accs = []
                test_accs = []
                true_label = []
                pre_label = []

                # 实例化模型并将其移动到 GPU
                model = Transformer().to(device)
                criterion = nn.CrossEntropyLoss()#交叉熵损失函数
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                total = sum([param.nelement() for param in model.parameters()])  # 计算总参数量
                print("Number of parameter: %.6f" % (total))  # 输出
                #训练模型
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total trainable parameters: {total_params}")
                total_step = len(train_loader)

                for epoch in range(epochs):
                    epoch_loss = 0.0
                    epoch_correct = 0
                    epoch_correct2 = 0
                    for step, (batch_x, batch_y) in enumerate(train_loader):  # 每个训练步骤
                        batch_x, batch_y = batch_x.to(try_gpu()), batch_y.to(try_gpu())
                        batch_y = np.squeeze(batch_y.long()) #改变维度
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        print('db:', '%2d' % db, 'wd:', '%2d' % wd, 'num:', '%02d' % i, 'Epoch:', '%04d' % (epoch + 1),
                              'Step:', '%04d' % (step + 1), 'loss =', '{:.6f}'.format(loss))
                        loss.backward()
                        optimizer.step()

                        _, predicted = torch.max(outputs.data, 1)
                        epoch_correct += torch.sum(predicted == batch_y)
                        epoch_loss += loss.item()
                    epoch_loss /= total_step
                    epoch_acc = 100.0 * epoch_correct / len(train_dataset)
                    # 保存损失和准确度
                    epoch_acc = epoch_acc.item()
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)
                    # 打印当前 epoch 的训练损失和准确度
                    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, epochs, epoch_loss, epoch_acc))

                    # 测试集
                    true_label = []
                    pre_label = []
                    for step, (batch_x2, batch_y2) in enumerate(test_loader):  # 每个训练步骤
                        batch_x2, batch_y2 = batch_x2.to(try_gpu()), batch_y2.to(try_gpu())
                        outputs2 = model(batch_x2)
                        batch_y2 = np.squeeze(batch_y2.long())
                        true_label.extend(batch_y2)
                        _, predicted2 = torch.max(outputs2.data, 1)
                        pre_label.extend(predicted2)
                        epoch_correct2 += torch.sum(predicted2 == batch_y2)
                    epoch_acc2 = 100.0 * epoch_correct2 / len(test_dataset)
                    # 保存损失和准确度
                    epoch_acc2 = epoch_acc2.item()
                    test_accs.append(epoch_acc2)
                time_used = time.time() - start
                print(time_used)
                # csvWrite.writerow(train_losses)

                output_model_dir = ""
                # 如果一开始用了并行训练最好加上这句
                model_to_save = model.module if hasattr(model, 'module') else model
                # 这样保存的是模型参数，记得格式是.pt
                # torch.save(model_to_save.state_dict(), output_model_dir + r'\model\CRB_Model_mix.pt')
                torch.save(model_to_save.state_dict(),r'MIMII_gear_Model_.pt')

                model = Transformer().to(device)
                # model_static_dict = torch.load(output_model_dir + r'\model\CRB_Model_mix.pt')
                model_static_dict = torch.load(r'MIMII_gear_Model_.pt')
                # 把参数加载到模型中
                model.load_state_dict(model_static_dict)

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
                    print('Accuracy: {:.2f}%'.format(test1_acc))
                    accuracy = accuracy_score(y_true, predicted_test)
                    precision = precision_score(y_true, predicted_test, average='macro')
                    recall = recall_score(y_true, predicted_test, average='macro')
                    f1 = f1_score(y_true, predicted_test, average='macro')

                print(accuracy)
                print(precision)
                print(recall)
                print(f1)

                csvWrite.writerow([time.strftime('%H:%M:%S', time.localtime(time.time())), accuracy, precision, recall, f1])

