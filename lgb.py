#!/usr/bin/python3
# -*- encoding: utf-8 -*-
from datetime import datetime
from copy import deepcopy
import os

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
        train = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')[:1000000]
    else:
        train = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    train['behavior_type'] = 1
    for _ in tqdm(range(2)):
        shuffle_item = train[['item_id', 'item_category']].sample(frac=1)
        # shuffle_item = pd.concat([shuffle_item, shuffle_item, shuffle_item], ignore_index=True).sample(frac=1)[
        #                :len(train)]
        shuffle_train = deepcopy(train)
        shuffle_train['item_id'] = shuffle_item['item_id']
        shuffle_train['item_category'] = shuffle_item['item_category']
        shuffle_train['behavior_type'] = 0
        train = pd.concat([shuffle_train, train], ignore_index=True)
    # for index, row in tqdm(train.iterrows()):
    #     row = row.values.tolist()
    #     neg = train.sample(n=2)[['item_id', 'item_category']].values.tolist()
    #     for neg_i in neg:
    #         row[1] = neg_i[0]
    #         row[4] = neg_i[1]
    #         row[2] = 0
    #         train.loc[train.shape[0] + 1] = row
    # train['time_slot'] = train['time'].map(lambda x: time_slot(x))
    # train['day_offset'] = train['time'].map(lambda x: day_offset(x))
    # train['hour'] = train['time'].map(lambda x: x[-2:])
    train['day'] = train['time'].map(lambda x: x[:10])
    label = train['behavior_type']
    feature = train.drop(columns=['time', 'user_geohash', 'behavior_type'])
    feature = label_encode(feature)
    # last_day_label = label_encoder['day'].transform(['2014-12-12']).tolist()[0]
    # test = train[train['day'] == last_day_label]

    # 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.1,
                                                        random_state=54321)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass',  # 目标函数
        'num_class': 2,
        'metric': {'multi_logloss'},  # 评估函数
        'num_leaves': 100,  # 叶子节点数
        'learning_rate': 0.005,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 10,  # k 意味着每 k 次迭代执行bagging
        'top_k': 20,
        'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_eval, early_stopping_rounds=5)

    # 保存模型到文件
    print('Save model...')
    gbm.save_model('lgb_model.txt')

    # 预测数据集
    print('Start predicting...')
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # print(y_test)
    # print(y_pred)

    # 评估模型
    print('The accuracy of prediction is: %s' % multi_class_acc(y_test, y_pred))

    # 生成提交结果
    print('Creating submit.csv...')
    cf = pd.read_csv('cf_predict.csv')
    item_last = pd.read_csv('data/tianchi_fresh_comp_train_item.csv')['item_id'].values.tolist()
    cf = cf[(cf['item_id'].isin(item_last))]
    cf['day'] = '2014-12-18'
    cf = label_encode(cf)
    step = 5000000
    for i in range(0, len(cf), step):
        cf_temp = cf[i:i + step]
        lgb_predict = gbm.predict(cf_temp, num_iteration=gbm.best_iteration)
        lgb_predict = lgb_predict.argmax(axis=1)
        lgb_predict = cf_temp[lgb_predict == 1]
        lgb_predict = lgb_predict[['user_id', 'item_id']]
        name = 'pure_lgb_predict.csv'
        if not os.path.exists(name):
            lgb_predict.to_csv(name, index=None, encoding='utf-8')
        else:
            lgb_predict.to_csv(name, index=None, header=None, mode='a+', encoding='utf-8')


if __name__ == '__main__':
    model_train(True)
    # model_train()
