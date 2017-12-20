import pretreatMapping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import io

train_data = pd.read_csv('D://course/SVM/data/train.txt', header=None)
# 去除重复的数据，训练集数据删除数据中多余的空格，并将缺失数据的问号变为NaN形式，并删除
train_data = train_data.drop_duplicates(keep='first')
for i in range(15):
    train_data[i] = train_data[i].map(pretreatMapping.base_mapping)
train_data = train_data.dropna()
# 将字符串信息转换为数字
train_data[1] = train_data[1].map(pretreatMapping.mapping_2)
train_data[3] = train_data[3].map(pretreatMapping.mapping_4)
train_data[5] = train_data[5].map(pretreatMapping.mapping_6)
train_data[6] = train_data[6].map(pretreatMapping.mapping_7)
train_data[7] = train_data[7].map(pretreatMapping.mapping_8)
train_data[8] = train_data[8].map(pretreatMapping.mapping_9)
train_data[9] = train_data[9].map(pretreatMapping.mapping_10)
train_data[13] = train_data[13].map(pretreatMapping.mapping_14)
train_data[14] = train_data[14].map(pretreatMapping.mapping_15)

test_data = pd.read_csv('D://course/SVM/data/test.txt', header=None)
# 并将缺失数据的问号变为NaN形式，并删除
for i in range(14):
    test_data[i] = test_data[i].map(pretreatMapping.base_mapping)
#test_data = test_data.dropna()
# 将字符串信息转换为数字
test_data[1] = test_data[1].map(pretreatMapping.mapping_2)
test_data[3] = test_data[3].map(pretreatMapping.mapping_4)
test_data[5] = test_data[5].map(pretreatMapping.mapping_6)
test_data[6] = test_data[6].map(pretreatMapping.mapping_7)
test_data[7] = test_data[7].map(pretreatMapping.mapping_8)
test_data[8] = test_data[8].map(pretreatMapping.mapping_9)
test_data[9] = test_data[9].map(pretreatMapping.mapping_10)
test_data[13] = test_data[13].map(pretreatMapping.mapping_14)

test_data = test_data.fillna(method='ffill')
# 将数据进行归一化，训练集和测试集合使用相同的规则
for i in range(14):
    mms = MinMaxScaler()
    train_data[[i]] = mms.fit_transform(train_data[[i]])
    test_data[[i]] = mms.transform(test_data[[i]])
# 填充一下缺失的数字
# 训练并预测 拆分成X参数和y结果
# attention 国籍被我搞没了先不用了
X_train = np.array(train_data.iloc[:, :-2])
y_train = np.array(train_data.iloc[:, 14])

# 调用库测试
print('开始训练啦')
clf = svm.SVC()
clf.fit(X_train, y_train)
print('训练结束啦')

print('开始预测啦')
X_test = np.array(test_data.iloc[:, :-1])

res = clf.predict(X_test)
outDataStream = io.open('D://course/SVM/data/myres.txt', 'w', encoding='utf-8')
outDataStream.write(pd.DataFrame(res).to_csv())
print('预测结束啦')

#X_test = np.array(pd.concat([testPdObj.iloc[:, 0:2], testPdObj.iloc[:, 3:]], axis=1)) 之前不要的代码
