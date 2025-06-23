import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
from scipy.io import wavfile

# 假设输入为 [B, 1, 1024]
class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15 * k), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3 * k), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3 * k), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3 * k), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16)  # 输出 [B, 256, 16]
        )

    def forward(self, x):
        return self.features(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=False):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
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
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(input_dim, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 input_dim * d_model
        self.conv1d = nn.Conv1d(n_sequence, n_sequence, kernel_size=1, stride=1, padding=0)
        self.pos_emb = PositionalEncoding(d_model) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

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
        self.linear1 = nn.Linear(4096, 256, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, num_class)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)      #形状为[batch_size, src_len]
        # print(enc_outputs.shape)
        x = self.linear1(enc_outputs.reshape(BATCH_SIZE, -1))
        x = self.relu1(x)
        x = self.linear2(x)     #x形状[64,512,2]
        return x


BATCH_SIZE = 32
input_dim = 16  # length of source
# 模型参数
d_model = input_dim  # Embedding Size，人工设置d_model的维度
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 5  # number of heads in Multi-Head Attention
n_sequence = 256  # TRM 1 1 SWT 3 5
n_result = 50     
num_class = 2
epochs = 15

class CTrans(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = MultiScaleCNN()
        self.transformer = Transformer()

    def forward(self, x):
        x = self.cnn(x)               # [B, 256, 16]
        x = self.transformer(x)       # [B, 1, d_model]
        # x = self.classifier(x[:, 0])  # [B, num_classes]
        return x


# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = r"/home/c220/Documents/MaLiuqi/data/CLFormer/-2db"
k = 1
window_size = 4096*k
overlap = 0.9
step = int(window_size * (1 - overlap))
num_classes = 2

# 加载数据函数
def load_wav_data(path):
    X, y = [], []
    for fname in os.listdir(path):
        if not fname.endswith('.wav'):
            continue
        label = 1 if 'anomaly' in fname else 0
        full_path = os.path.join(path, fname)
        try:
            sr, audio = wavfile.read(full_path)
            audio = audio.astype(np.float32)
            if audio.ndim > 1:  # 多通道取单通道
                audio = audio[:, 0]
            # 滑动窗口处理
            for start in range(0, len(audio) - window_size + 1, step):
                segment = audio[start:start + window_size]
                X.append(segment[np.newaxis, :])  # [1, 1024]
                y.append(label)
        except Exception as e:
            print(f"⚠️ Error loading {fname}: {e}")
    return torch.tensor(X), torch.tensor(y)

# 加载数据
X, y = load_wav_data(data_path)
print(f"Total segments: {len(X)}, Positive: {y.sum().item()}, Negative: {(y == 0).sum().item()}")

# 划分训练/测试集
total_len = len(X)
train_len = int(0.8 * total_len)
test_len = total_len - train_len
dataset = TensorDataset(X, y)
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

# 模型训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CTrans(num_classes=num_classes).to(device)
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
        loss = criterion(out, batch_y)
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
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"\n✅ Evaluation:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
