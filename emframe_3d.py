import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from numpy.fft import fft
import os


def sliding_window(data, window_size):
    samples = (len(data) - window_size + 1)//5
    result = np.zeros((samples, window_size, data.shape[1]))
    # 生成3D数据
    for i in range(samples):
        result[i, :, :] = data[i*3:i*3+window_size, :]
    return result



path = r"C:\Users\86198\Desktop\study\MIMII_pump\6_dB_pump\id_00\normal_feature\\"
AllData = np.empty((0, 35))
for num, file in enumerate(os.listdir(path), 1):
    # print(file.split("_")[3])
    # if(file.split("_")[3] == "train"):
    #     continue
    temp = np.loadtxt(path+file, delimiter=',')
    AllData = np.r_[AllData, temp]

# 定义滑动窗口大小
window_size = 50

# 使用滑动窗口处理数据
processed_data1 = sliding_window(AllData[:, 0:35], window_size)
np.random.shuffle(processed_data1)
np.save(r'C:\Users\86198\Desktop\study\MIMII_pump\6_dB_pump\id_00\npy\S10_normal_test.npy', processed_data1[0:1000])
np.save(r'C:\Users\86198\Desktop\study\MIMII_pump\6_dB_pump\id_00\npy\S10_normal_train.npy', processed_data1[1000:])

print(f"{processed_data1.shape}")
print("over")
