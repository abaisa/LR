import pandas as pd
import numpy as np

ori_data = pd.read_csv('data/train.csv')

train_data = ori_data.iloc[:, 2:]
train_target = ori_data.iloc[:, 1]
num_features_data = ori_data.iloc[:, 2:15]
digit_features_data = ori_data.iloc[:, 15:41]

# 非数值特征转二进制dummies后pca降维
digit_features_data = digit_features_data.drop(columns=[\
    'C3', 'C4', 'C7', 'C10', 'C11', 'C12', 'C13', 'C16', 'C19', 'C21', 'C22', 'C25', 'C26', 'C20'])
digit_features_data.fillna(method='ffill')
dummies = pd.get_dummies(digit_features_data)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)
# pca.fit(dummies)
# pca_digits_data = pca.transform(dummies)]
pca_digits_data = dummies
pca_digits_data = pd.DataFrame(pca_digits_data)

# 数值的部分用均值填补缺失值
for i in range(num_features_data.shape[1]):
    num_features_data.iloc[:, i] = num_features_data.iloc[:, i].replace(np.nan, num_features_data.iloc[:, i].mean())

# 这里随便写了一下，train_data需要拼接数值和非数值的部分
train_data = pd.concat([num_features_data, pca_digits_data], axis=1)
train_data = num_features_data
# print(train_data)
# 交叉验证

from sklearn.model_selection import train_test_split
#0.2的比例划分为训练集和验证集
train_x,test_x,train_y,test_y = train_test_split(train_data,train_target,test_size=0.2,random_state=7)


# 罗辑回归训练并预测
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 1000, random_state=0)
lr.fit(train_x, train_y)
prob_predict_res = lr.predict_proba(test_x)

#计算AUC和准确率的score
from sklearn import metrics
test_auc = metrics.roc_auc_score(test_y, prob_predict_res[:, 1])
print(prob_predict_res[:, 1])
print('auc value is {0}'.format(test_auc))

predict_res = lr.predict(test_x)
predict_score = metrics.accuracy_score(test_y, predict_res)
print('predict score is {0}'.format(predict_score))
