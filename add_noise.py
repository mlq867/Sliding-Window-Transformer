import time

import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import math

from scipy.io import wavfile
from tqdm import tqdm
import wave
import random


def add_noise(the_clean_data, the_noise_data, save_file, SNR, fs):
    # data align
    if len(the_clean_data) > len(the_noise_data):
        times = math.ceil(len(the_clean_data)/len(noise_data)) #向上取整
        the_noise_data = list(the_noise_data)*times
        the_noise_data = np.array(the_noise_data)

    the_noise_data = the_noise_data[:len(the_clean_data)]

    #计算语音信号功率Ps和噪声功率Pn1
    Ps = np.sum(the_clean_data ** 2) / len(the_clean_data)
    Pn1 = np.sum(the_noise_data ** 2) / len(the_noise_data)

    # 计算k值
    k = math.sqrt(Ps/(10**(SNR/10)*Pn1))
    #将噪声数据乘以k,
    random_values_we_need = the_noise_data*k

    #将噪声数据叠加到纯净音频上去
    outdata = the_clean_data + random_values_we_need

    # 将叠加噪声的数据写入文件
    wavfile.write(save_file, fs, outdata)



if __name__ == '__main__':
    path = r'D:\声纹数据集\江大数据（泵）\pumpRAB\\'
    subDir = [r'normal\\', r'abnormal_25\\', r'abnormal_50\\', r'abnormal_75\\', r'abnormal_25_75\\']
    # noise_data, fs = librosa.load(r'C:\Users\86198\Desktop\噪声\factory1.wav', sr=48000, mono=False)
    # noise_data = np.array(noise_data, dtype=np.float32)
    SNR = -16
    save_path = r'D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise' + str(SNR) + r'db\\'
    # save_path = r'D:\声纹数据集\江大数据（泵）\pumpFNoise' + str(SNR) + r'db\\'
    for subdir in subDir:
        for num, file in enumerate(os.listdir(path + subdir), 1):
            # np.random.seed(time.time_ns() % (2**31 - 1))
            # index = random.randint(0, len(noise_data) - 48000 * 3)
            noise_data = np.random.normal(loc=0.0, scale=1.0, size=48000*3)
            clean_data, _ = librosa.load(path + subdir + file, sr=None, mono=False)
            clean_data = np.array(clean_data, dtype=np.float32)
            # add_noise(clean_data, noise_data[index:], save_path + subdir + file, SNR, fs)
            add_noise(clean_data, noise_data, save_path + subdir + file, SNR, 48000)
            print(file)
