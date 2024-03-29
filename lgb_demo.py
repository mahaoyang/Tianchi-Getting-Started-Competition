#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=321)

# 加载你的数据
# print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
#
# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    # 'objective': 'regression',  # 目标函数
    # 'metric': {'l2', 'auc'},  # 评估函数
    'objective': 'multiclass',  # 目标函数
    'num_class': 3,
    'metric': {'multi_logloss'},  # 评估函数
    'num_leaves': 3,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
    'top_k': 20,
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=2, valid_sets=lgb_eval,
                early_stopping_rounds=1)

print('Save model...')
# 保存模型到文件
gbm.save_model('model.txt')

# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration, pred_leaf=True)
# print(y_pred)
print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# print(y_test)
# print(y_pred)


def multi_class_acc(y_test, y_pred):
    pred = y_pred.argmax(axis=1)
    acc = 1 - float(np.count_nonzero(pred - y_test)) / len(y_test)
    return acc


# 评估模型
print('The accuracy of prediction is: %s' % multi_class_acc(y_test, y_pred))
