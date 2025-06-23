
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')

if __name__ == '__main__':
    n_result = 50        # Adaboost
    epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
# ----------------------------------------加载数据---------------------------------------------
    db_list = [2, 0, -2, -4, -6, -8, -10, -12]
    for db in db_list:
        root_path = r"/home/c220/Documents/xxx/data/MIMII_gear/noise/" + str(db) + r"db/"
        acc_array = []
        re_array = []
        F1_array = []

        csvWrite = csv.writer(open('test.csv', mode='a', encoding="utf-8-sig", newline=""))
        csvWrite.writerow([root_path, "accuracy", "precision", "recall", "f1", "adaboost"])

        start = time.time()

        # **********************泵（将大数据）*******************************
        normal_train = np.load(root_path + r'normal_train_50.npy')
        abnormal_train = np.load(root_path + r'abnormal_train_50.npy')

        normal_test = np.load(root_path + r'normal_test_50.npy')
        abnormal_test = np.load(root_path + r'abnormal_test_50.npy')

        normal_train = normal_train[:, 0:5, :]
        abnormal_train = abnormal_train[:, 0:5, :]
        normal_test = normal_test[:, 0:5, :]
        abnormal_test = abnormal_test[:, 0:5, :]

        normal_train = normal_train.reshape(-1, normal_train.shape[2])
        abnormal_train = abnormal_train.reshape(-1, abnormal_train.shape[2])
        normal_test = normal_test.reshape(-1, normal_test.shape[2])
        abnormal_test = abnormal_test.reshape(-1, abnormal_test.shape[2])

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

            # 创建AdaBoost分类器实例
            # base_estimator参数指定基础分类器，默认为决策树桩（Decision Tree Stump）
            # n_estimators参数指定了要结合的弱分类器的数量
            ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)

            # 训练模型
            ada_clf.fit(xtrain, ytrain)

            # 预测测试集
            predicted_test = ada_clf.predict(xtest)

            accuracy = accuracy_score(ytest, predicted_test)
            precision = precision_score(ytest, predicted_test, average='macro')
            recall = recall_score(ytest, predicted_test, average='macro')
            f1 = f1_score(ytest, predicted_test, average='macro')

            print(accuracy)
            print(precision)
            print(recall)
            print(f1)

            csvWrite.writerow([time.strftime('%H:%M:%S', time.localtime(time.time())), accuracy, precision, recall, f1])






