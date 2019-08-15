# -*-coding:utf8-*-
"""
author:david
date:2018****
lfm model train main function
"""
import time
import json
import random
import pickle
import threading
import multiprocessing

from redis import Redis
from tqdm import trange
import pandas as pd
import numpy as np
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer

from spark_session import spark

redis_cli = Redis()
user_b_input_filter = 'user_behavior_input_filter'
user_b_output_filter = 'user_behavior_output_filter'
user_b_output = 'user_behavior_output'


def task_filter(key, value):
    flag = redis_cli.sismember(key, value)
    redis_cli.sadd(key, value)
    return flag


def score(behavior):
    behavior_map = {'pv': 1, 'buy': 4, 'cart': 3, 'fav': 2}
    return behavior_map[behavior]


def read_user_behavior(path='data/ECommAI_EUIR_round1_testB_20190809/user_behavior.csv'):
    user_df_t = pd.read_csv(path, header=None)
    user_df_t.columns = ['user_id', 'item_id', 'behavior', 'timestamp']
    user_df = user_df_t[['user_id', 'item_id', 'behavior']]
    user_df['behavior'] = user_df['behavior'].map(lambda x: score(x))
    user_df = pd.read_csv(path, header=None)
    user_df.columns = ['user_id', 'item_id', 'behavior', 'timestamp']
    user_df = user_df[['user_id', 'item_id', 'behavior']]
    user_df['behavior'] = user_df['behavior'].map(lambda x: score(x))
    user_df = pd.concat([user_df, user_df_t])
    with open('user_behavior.pk', 'wb') as f:
        pickle.dump(user_df, f)
    return user_df
    # return spark.createDataFrame(user_df)


def read_user(path='data/ECommAI_EUIR_round1_testB_20190809/user.csv'):
    # user_df_t = pd.read_csv(path, header=None)
    # user_df_t.columns = ['user_id', 'gender', 'age', 'buy_power']
    # user_list_t = user_df_t['user_id'].values.tolist()
    user_df = pd.read_csv(path, header=None)
    user_df.columns = ['user_id', 'gender', 'age', 'buy_power']
    user_list = user_df['user_id'].values.tolist()
    # user_list.extend(user_list_t)
    user_list.sort()
    idx_user = dict(enumerate(user_list))
    user_idx = dict(zip(idx_user.values(), idx_user.keys()))
    with open('user.pk', 'wb') as f:
        pickle.dump((idx_user, user_idx), f)
    return idx_user, user_idx


def read_item(path='data/ECommAI_EUIR_round1_testB_20190809/item.csv'):
    item_df_t = pd.read_csv('data/ECommAI_EUIR_round1_testB_20190809/item.csv', header=None)
    item_df_t.columns = ['item_id', 'category_id', 'store_id', 'brand_id']
    item_list_t = item_df_t['item_id'].values.tolist()
    item_df = pd.read_csv(path, header=None)
    item_df.columns = ['item_id', 'category_id', 'store_id', 'brand_id']
    item_list = item_df['item_id'].values.tolist()
    item_list.extend(item_list_t)
    item_list.sort()
    idx_item = dict(enumerate(item_list))
    item_idx = dict(zip(idx_item.values(), idx_item.keys()))
    with open('item.pk', 'wb') as f:
        pickle.dump((idx_item, item_idx), f)
    return idx_item, item_idx


def read_data_input(path_b='data/ECommAI_EUIR_round1_testB_20190809/user_behavior.csv',
                    path_u='data/ECommAI_EUIR_round1_testB_20190809/user.csv',
                    path_i='data/ECommAI_EUIR_round1_testB_20190809/item.csv'):
    # idx_item, item_idx = read_item(path_i)
    # idx_user, user_idx = read_user(path_u)
    # user_b = read_user_behavior(path_b)
    # with open('item.pk', 'rb') as f:
    #     idx_item, item_idx = pickle.load(f)
    # item_len = len(idx_item)
    with open('user.pk', 'rb') as f:
        idx_user, user_idx = pickle.load(f)
    user_len = len(idx_user)
    with open('user_behavior.pk', 'rb') as f:
        user_b = pickle.load(f)
    del user_idx
    for i in trange(user_len):
        if task_filter(user_b_input_filter, i):
            continue
        user_b_new = user_b[user_b['user_id'] == idx_user[i]]
        # user_b = user_b[user_b['user_id'] != idx_user[i]]
        # user_b_new = user_b_new.toPandas()
        user_b_new = user_b_new.values.tolist()
        for ub in user_b_new:
            redis_cli.rpush('user_idx:%s' % i, json.dumps(ub))


def read_data_output():
    with open('item.pk', 'rb') as f:
        idx_item, item_idx = pickle.load(f)
    item_len = len(idx_item)
    with open('user.pk', 'rb') as f:
        idx_user, user_idx = pickle.load(f)
    user_len = len(idx_user)
    del idx_item, idx_user, user_idx
    for i in trange(user_len):
        if task_filter(user_b_output_filter, i):
            continue
        user_b_new = redis_cli.lpop('user_idx:%s' % i)
        if user_b_new:
            user_b_new = json.loads(user_b_new.decode('utf-8'))
            train_line = [0] * item_len
            label = 0
            for ub in user_b_new:
                train_line[item_idx[ub[1]]] = 1 if ub[2] else 0
                if label < 4:
                    label = max(train_line[item_idx[ub[1]]], ub[2])
            text = '%s ' % label
            for idx in range(item_len):
                if train_line[idx] == 1:
                    text += '%s:%s ' % (idx + 1, 1)
            text += '\n'
            redis_cli.rpush(user_b_output, text)


def data_input():
    read_data_input(path_b='data/ECommAI_EUIR_round1_train_20190701/user_behavior.csv',
                    path_i='data/ECommAI_EUIR_round1_train_20190701/item.csv',
                    path_u='data/ECommAI_EUIR_round1_train_20190701/user.csv')


def trans_libsvm():
    with open('train.libsvm', 'w') as f:
        text = redis_cli.lpop(user_b_output)
        if text:
            text = text.decode('utf-8')
            f.write(text)
        # while 1:
        #     text = redis_cli.lpop(user_b_output)
        #     if text:
        #         text = text.decode('utf-8')
        #         f.write(text)
        #     else:
        #         time.sleep(10)


def lda_train():
    # Loads data.
    dataset = spark.read.format("libsvm").load("train.libsvm")

    # Trains a LDA model.
    lda = LDA(k=10, maxIter=100)
    model = lda.fit(dataset)

    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    # Describe topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    # Shows the result
    transformed = model.transform(dataset)
    transformed.show(truncate=False)

    # Save & Stop
    model.save('lda.model')
    spark.stop()


if __name__ == "__main__":
    # read_data()
    # read_data(path_b='data/ECommAI_EUIR_round1_train_20190701/user_behavior.csv',
    #           path_i='data/ECommAI_EUIR_round1_train_20190701/item.csv',
    #           path_u='data/ECommAI_EUIR_round1_train_20190701/user.csv')
    # for i in range(16):
    #     t1 = threading.Thread(target=data_input, name=str(i), args=())
    #     t1.start()
    #     time.sleep(30)
    for i in range(16):
        t1 = threading.Thread(target=read_data_output, name=str(i), args=())
        t1.start()
        time.sleep(30)
    # lda_train()
