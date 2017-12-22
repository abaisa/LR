from numpy import *
import pandas as pd

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def GetResult():
    ori_data = pd.read_csv('data/train.csv')
    train_data = ori_data.iloc[:, 2:]
    train_target = ori_data.iloc[:, 1]
    num_features_data = ori_data.iloc[:, 2:15]
    # 数值的部分用均值填补缺失值
    for i in range(num_features_data.shape[1]):
        num_features_data.iloc[:, i] = num_features_data.iloc[:, i].replace(nan, num_features_data.iloc[:, i].mean())

    train_data = num_features_data
    # 归一
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(train_data)
    train_data = sc.transform(train_data)

    from sklearn.model_selection import train_test_split
    #0.2的比例划分为训练集和验证集
    train_x,test_x,train_y,test_y = train_test_split(train_data,train_target,test_size=0.2,random_state=7)
    weights = gradAscent(train_x, train_y)
    print(weights)
    predict_res = test_x * weights

    from sklearn.preprocessing import MinMaxScaler
    mmsc = MinMaxScaler()
    predict_res = mmsc.fit_transform(predict_res)
    print(predict_res)

    # 计算AUC和准确率的score
    from sklearn import metrics
    test_auc = metrics.roc_auc_score(test_y, predict_res)
    print('auc value is {0}'.format(test_auc))

    for i in range(predict_res.size):
        if predict_res[i] > 0.5:
            predict_res[i] = 1
        else:
            predict_res[i] = 0
    predict_score = metrics.accuracy_score(test_y, predict_res)
    print('predict score is {0}'.format(predict_score))

if __name__ == '__main__':
    GetResult()