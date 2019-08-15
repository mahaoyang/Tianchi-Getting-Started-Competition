#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import time
import pickle
import pandas as pd
from collections import OrderedDict


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H")))


def read_user():
    df = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    df = df[['user_id', 'item_id', 'behavior_type']]
    df.sort_values(by=['user_id', 'item_id'], inplace=True)
    item = sorted(list(set(df['item_id'])))
    idx_item = dict(enumerate(item))
    item_idx = dict(zip(idx_item.values(), idx_item.keys()))
    user = sorted(list(set(df['user_id'])))
    idx_user = dict(enumerate(user))
    user_idx = dict(zip(idx_user.values(), idx_user.keys()))
    df = df.values.tolist()
    print('item size %s' % len(idx_item))
    label = {}
    for i in df:
        label.setdefault(i[0], {})
        label[i[0]].setdefault(i[1], 0)
        label[i[0]][i[1]] += i[2]
    print('build label complete')
    train = []
    for uid in sorted(label.keys()):
        line = '%s ' % user_idx[uid]
        for tid in sorted(label[uid].keys()):
            line += '%s:%s ' % (item_idx[tid] + 1, label[uid][tid])
        train.append(line.strip() + '\n')
    print('build train dataset complete')
    with open('train.libsvm', 'w') as f:
        f.write(''.join(train))
    with open('idx.pickle', 'wb') as f:
        pickle.dump((idx_item, item_idx, idx_user, user_idx, label, train), f)
    return idx_item, item_idx, idx_user, user_idx, label, train


if __name__ == '__main__':
    read_user()
