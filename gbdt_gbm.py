#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import time
import json
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV as lrcv


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H")))


def neg_sample(length):
    return random.randint(0, length)


user = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
item_last = pd.read_csv('data/tianchi_fresh_comp_train_item.csv')['item_id'].values.tolist()
user = user[(user['item_id'].isin(item_last))]
user['time'] = user['time'].apply(lambda x: timestamp(x))
item = user[['item_id', 'item_category']].drop_duplicates().values.tolist()
item_length = len(item)
print('item_length %s' % item_length)
for index, row in tqdm(user.iterrows()):
    row = row.values.tolist()
    neg = item[neg_sample(item_length)]
    row[1] = neg[0]
    row[4] = neg[1]
    user.loc[user.shape[0] + 1] = row
label = user['behavior_type']
feature = user[['user_id', 'item_id', 'item_category', 'time']]
print(len(user))
print(user.dtypes)
# label = label.values.tolist()
# feature = feature.values.tolist()
train_x, test_x, train_y, test_y = train_test_split(feature, label, test_size=0.2, random_state=0)

train = lgb.Dataset(train_x, train_y)
test = lgb.Dataset(test_x, test_y, reference=train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 100,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params, train, num_boost_round=300, valid_sets=test, early_stopping_rounds=10)
gbm.save_model('gbm.txt')
y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
print(y_pred)
# print('The roc of prediction is:', roc_auc_score(test_y, y_pred))

y_pred = gbm.predict(feature, num_iteration=gbm.best_iteration)
feature['leaf'] = y_pred
lr_cf = lrcv(Cs=[1], penalty="l2", tol=0.0001, max_iter=500, cv=5).fit(feature.as_matrix(),
                                                                       np.array(label.values.tolist()).reshape((-1, 1)))

item = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
item['time'] = item['time'].apply(lambda x: timestamp(x))
item = item[['item_id', 'item_category']].drop_duplicates(
    'item_id').astype('int32')
cfp = pd.read_csv('cf_predict.csv')
cfp = cfp.merge(item, on='item_id', how='left')
cfp = cfp[['user_id', 'item_id', 'item_category']]
cfp['time'] = int(time.time())
feature = cfp[['user_id', 'item_id', 'item_category', 'time']]
y_pred = gbm.predict(feature, num_iteration=gbm.best_iteration, pred_leaf=True)

y_pred = y_pred.tolist()
print('leaf %s' % y_pred)
lgb_predict = feature[['user_id', 'item_id', 'item_category', 'time']]
lgb_predict['leaf'] = y_pred

lr_predict = lr_cf.predict(lgb_predict.fillna(0).as_matrix()).tolist()
lgb_predict['predict'] = lr_predict
lgb_predict = lgb_predict.sort_values(['user_id', 'predict'], ascending=[1, 0])
lgb_predict = lgb_predict.groupby(['user_id']).head(30)
# lgb_predict = lgb_predict.drop_duplicates('user_id')
lgb_predict = lgb_predict[['user_id', 'item_id']]
lgb_predict.to_csv('lgb_predict.csv', index=None)

model_json = gbm.dump_model()
with open('gbm.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
