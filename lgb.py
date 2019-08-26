#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# def time_slot(x):
#     x = int(x.strip()[-2:])
#     if 0 < x <= 7:
#         return 0
#     elif 7 < x <= 11:
#         return 1
#     elif 11 < x <= 14:
#         return 2
#     elif 14 < x <= 18:
#         return 3
#     elif 18 < x <= 24:
#         return 4
#
#
# def day_offset(x):
#     x = datetime.strptime(x[:10], '%Y-%m-%d')
#     y = datetime.strptime('2014-12-19', '%Y-%m-%d')
#     delta = y - x
#     return delta.days


label_encoder = {}


def label_encode(x):
    for i in x.columns:
        encoder = LabelEncoder()
        x[i] = encoder.fit_transform(x[i])
        label_encoder[i] = encoder
    return x


def label_inverse(x):
    for i in x.columns:
        x[i] = label_encoder[i].inverse_transform(x[i])
    return x


# def one_label_inverse(i, x):
#     x = label_encoder[i].inverse_transform(x)
#     return x

def multi_class_acc(y_test, y_pred):
    pred = y_pred.argmax(axis=1)
    acc = 1 - float(np.count_nonzero(pred - y_test)) / len(y_test)
    return acc


def model_train(debug=False):
    if debug:
        train = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')[:10000]
    else:
        train = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    # train['time_slot'] = train['time'].map(lambda x: time_slot(x))
    # train['day_offset'] = train['time'].map(lambda x: day_offset(x))
    train['hour'] = train['time'].map(lambda x: x[-2:])
    train['day'] = train['time'].map(lambda x: x[:10])
    for index, row in tqdm(train.iterrows()):
        row = row.values.tolist()
        neg = train.sample(n=5)[['item_id', 'item_category']].values.tolist()
        for neg_i in neg:
            row[1] = neg_i[0]
            row[4] = neg_i[1]
            row[2] = 0
            train.loc[train.shape[0] + 1] = row
    label = train['behavior_type']
    feature = train.drop(columns=['time', 'user_geohash', 'behavior_type'])
    feature = label_encode(feature)
    # last_day_label = label_encoder['day'].transform(['2014-12-12']).tolist()[0]
    # test = train[train['day'] == last_day_label]
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2,
                                                        random_state=321)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass',  # 目标函数
        'num_class': 5,
        'metric': {'multi_logloss'},  # 评估函数
        'num_leaves': 200,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
        'top_k': 20,
        'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    print('Start training...')
    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Save model...')
    # 保存模型到文件
    gbm.save_model('lgb_model.txt')

    print('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # print(y_test)
    # print(y_pred)

    # 评估模型
    print('The accuracy of prediction is: %s' % multi_class_acc(y_test, y_pred))
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    model_train(True)
