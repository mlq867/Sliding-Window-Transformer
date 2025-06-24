import numpy as np

# ------------------------------------------预加重----------------------------------------
def pre_fun(x):  # 定义预加重函数
    signal_points=len(x)  # 获取语音信号的长度
    signal_points=int(signal_points)  # 把语音信号的长度转换为整型
    # s=x  # 把采样数组赋值给函数s方便下边计算
    for i in range(1, signal_points, 1):# 对采样数组进行for循环计算
        x[i] = x[i] - 0.98 * x[i - 1]  # 一阶FIR滤波器
    return x  # 返回预加重以后的采样数组
# -----------------------------------------分帧-------------------------------------------
def frame(x, lframe, mframe):  # 定义分帧函数
    signal_length = len(x)  # 获取语音信号的长度
    fn = (signal_length)/mframe  # 分成fn帧-lframe
    fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
    fn1 = int(fn1)  # 将帧数化为整数
    # 求出添加的0的个数
    numfillzero = (fn1*mframe+lframe)-signal_length
    # 生成填充序列
    fillzeros = np.zeros(numfillzero)
    # 填充以后的信号记作fillsignal
    fillsignal = np.concatenate((x,fillzeros))  # concatenate连接两个维度相同的矩阵
    # 对所有帧的时间点进行抽取，得到fn1*lframe长度的矩阵d
    d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1*mframe, mframe), (lframe, 1)).T
    # 将d转换为矩阵形式（数据类型为int类型）
    d = np.array(d, dtype=np.int32)
    signal = fillsignal[d]
    return(signal, fn1, numfillzero)



