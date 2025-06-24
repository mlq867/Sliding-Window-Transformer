import os
import librosa
import PreProcess
from Feature import Fea_Extra
import scipy.io.wavfile as wav
import wave
import Feature
import pandas as pd
import struct
import binascii
import numpy as np
def wav_read(filedir):

    #打开wav文件 ，open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据。
    f = wave.open(filedir, "rb")
    #读取格式信息
    #一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采样频率, 采样点数
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print(params[:4])
    #读取波形数据
    #读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_data  = f.readframes(nframes)
    f.close()
    #将波形数据转换成数组
    #需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_data = np.frombuffer(str_data,dtype = np.int16)
    #将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
    wave_data.shape = -1,2
    #转置数据
    wave_data = wave_data.T
    # print(wave_data.shape)
    #通过取样点数和取样频率计算出每个取样的时间。
    time=np.arange(0,nframes)/framerate
    wave_data = np.require(wave_data, dtype='f4', requirements=['O', 'W'])
    wave_data.flags.writeable = True
    return wave_data,framerate,time

def mean_std(x):
    x1 = np.empty((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        xmean = np.mean(x[i, :])
        xstd = np.std(x[i, :])
        for j in range(x.shape[1]):
            x1[i, j] = (x[i, j]-xmean)/xstd
    return x1

def MaxMinScale(x):
    x1 = np.empty((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        xmax = max(x[i,:])
        xmin = min(x[i,:])
        if xmax == xmin:
            x1 = np.ones((x.shape[0], x.shape[1]))
            continue
        for j in range(x.shape[1]):
            x1[i,j] = (x[i,j]-xmin)/(xmax-xmin)
    return x1

def wavhandle(path):
    data, sample_rate, _ = wav_read(path)
    count = 0
    for i in range(data.shape[1]):
        if int(data[0, i] * data[1, i]) != 0:
            break
        count = count + 1
    data = data[:, count:]
    standard_data = MaxMinScale(data)
    return standard_data, sample_rate

def csvhandle(path):
    file_rate = 48000
    df = pd.read_csv(path, header=None, dtype='str')
    rows = df.values.shape[0]
    columns = df.values.shape[1]
    datanp = np.empty((1, rows*(columns-1)))
    print(f"表格行数：{rows}  表格列数:{columns}")
    count = 0
    for row in range(rows):
        for column in range(columns)[1:]:
            # 读取为二进制编码
            # data = np.empty((8, 1))
            # for i in range(8):
            #     data1 = binascii.unhexlify(df.values[row][column][i*4:i*4+4])
            #     # 使用 struct.unpack 对二进制编码进行解包，有符号整形
            #     pcm_data = struct.unpack('<h', data1)
            #     data[i, 0] = pcm_data[0]
            # datanp[0, count] = float(sum(data)/8.0)
            data = binascii.unhexlify(df.values[row][column][0:4])
            pcm_data = struct.unpack('<h', data)
            datanp[0, count] = pcm_data[0]
            count = count + 1
        print(f"{row}/{rows}")
    print(path)
    return mean_std(datanp), file_rate
#MaxMinScale(mean_std(datanp))


def wav_mean_handle(path):
    dat, fs = librosa.load(path, sr=None, mono=False)
    # x = np.array([dat.T], dtype=np.float32)
    x = np.array(dat.T, dtype=np.float32)
    print(x.shape)
    # x = np.mean(x, axis=1)
    # print(x.shape)
    return x, fs


num = r"-12"

# path_list = [r"D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise" + num + r"db\normal\\",
#              r"D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise" + num + r"db\abnormal_25\\",
#              r"D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise" + num + r"db\abnormal_50\\",
#              r"D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise" + num + r"db\abnormal_75\\",
#              r"D:\声纹数据集\江大数据（泵）\pumpWNoise\pumpWNoise" + num + r"db\abnormal_25_75\\"]
#
# save_list = [r"D:\声纹数据集\江大数据（泵）\Wfeature\\" + num + r"db\normal\\",
#              r"D:\声纹数据集\江大数据（泵）\Wfeature\\" + num + r"db\abnormal_25\\",
#              r"D:\声纹数据集\江大数据（泵）\Wfeature\\" + num + r"db\abnormal_50\\",
#              r"D:\声纹数据集\江大数据（泵）\Wfeature\\" + num + r"db\abnormal_75\\",
#              r"D:\声纹数据集\江大数据（泵）\Wfeature\\" + num + r"db\abnormal_25_75\\"]

path_list = [r"E:\声纹数据集\江大数据（泵）\pumpFNoise" + num + r"db\normal\\",
             r"E:\声纹数据集\江大数据（泵）\pumpFNoise" + num + r"db\abnormal_25\\",
             r"E:\声纹数据集\江大数据（泵）\pumpFNoise" + num + r"db\abnormal_50\\",
             r"E:\声纹数据集\江大数据（泵）\pumpFNoise" + num + r"db\abnormal_75\\",
             r"E:\声纹数据集\江大数据（泵）\pumpFNoise" + num + r"db\abnormal_25_75\\"]

save_list = [r"E:\声纹数据集\江大数据（泵）\Noise+Length\-12db+4096\normal\\",
             r"E:\声纹数据集\江大数据（泵）\Noise+Length\-12db+4096\abnormal_25\\",
             r"E:\声纹数据集\江大数据（泵）\Noise+Length\-12db+4096\abnormal_50\\",
             r"E:\声纹数据集\江大数据（泵）\Noise+Length\-12db+4096\abnormal_75\\",
             r"E:\声纹数据集\江大数据（泵）\Noise+Length\-12db+4096\abnormal_25_75\\"]

if __name__ == '__main__':
    for c in range(len(path_list)):
        path = path_list[c]
        for num, file in enumerate(os.listdir(path), 1):
            file_data, file_rate = wav_mean_handle(path + file)
            # file_data_one = file_data[0, :]
            file_data_one = file_data
            file_data_one = file_data_one * 1.0 / (max(abs(file_data_one)))
            file_data_one = PreProcess.pre_fun(file_data_one)
            print(file_data_one.shape)
            frame_data, _, _ = PreProcess.frame(file_data_one, 2048, 1024)
            temp1 = np.empty((0, 35))  # 0-11时域，11-23频域，23-35MFCC
            for i in range(len(frame_data)):
                feature_voice = Feature.Fea_Extra(frame_data[i, :], file_rate)
                fea = feature_voice.Both_Fea()
                temp1 = np.vstack((temp1, fea))
            # temp_path = r'D:\声纹数据集\江大数据（泵）\feature\20db\abnormal_20db_25_75\\' + file.split('.wav')[0] + '.csv'
            temp_path = save_list[c] + file.split('.wav')[0] + '.csv'
            print(file)
            np.savetxt(temp_path, temp1, delimiter=',')
