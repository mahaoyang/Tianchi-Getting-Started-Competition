import time
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, get_fixlen_feature_names


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H")))


def neg_sample(length):
    return random.randint(0, length)


def read(data, lbe_store):
    # data['time'] = data['time'].apply(lambda x: timestamp(x), convert_dtype='int32')
    sparse_features = ["user_id", "item_id", "item_category", "time"]
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        data[feat] = lbe_store[feat].transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
    data_model_input = [data[name].values for name in fixlen_feature_names]
    return data, data_model_input


if __name__ == "__main__":

    data = pd.read_csv("data/tianchi_fresh_comp_train_user.csv")
    item_last = pd.read_csv('data/tianchi_fresh_comp_train_item.csv')['item_id'].values.tolist()
    data = data[(data['item_id'].isin(item_last))]
    data['time'] = data['time'].apply(lambda x: timestamp(x), convert_dtype='int32')
    item = data[['item_id', 'item_category']].drop_duplicates().values.tolist()
    item_length = len(item)
    print('item_length %s' % item_length)
    for index, row in tqdm(data.iterrows()):
        row = row.values.tolist()
        neg = item[neg_sample(item_length)]
        row[1] = neg[0]
        row[4] = neg[1]
        data.loc[data.shape[0] + 1] = row
    sparse_features = ["user_id", "item_id", "item_category", "time"]
    target = ['behavior_type']
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    lbe_store = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        lbe_store[feat] = lbe
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    data = data.sample(frac=0.01)
    train, test = train_test_split(data, test_size=0.2)
    # train = train[:1000]
    # test = test[:200]
    train_model_input = [train[name].values for name in fixlen_feature_names]
    test_model_input = [test[name].values for name in fixlen_feature_names]
    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=100, verbose=2, validation_split=0.2, )

    pred_ans = model.predict(test_model_input, batch_size=256)
    print(pred_ans)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))

    test = pd.read_csv('lgb_predict.csv')
    item = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    item = item[['item_id', 'item_category']].drop_duplicates('item_id').astype('int32')
    # item = dict(zip(item['item_id'].values.tolist(), item['item_category'].values.tolist()))
    test = test.join(item.set_index('item_id'), on='item_id', how='left')
    test['time'] = timestamp(pd.read_csv('data/tianchi_fresh_comp_train_user.csv')['time'].max())
    _, test_model_input = read(test.astype('int32'), lbe_store)
    pred_ans = model.predict(test_model_input, batch_size=256)
    pred_ans = pred_ans.reshape((1, -1)).tolist()[0]
    test = pd.read_csv('lgb_predict.csv')
    test['predict'] = pred_ans
    test = test[['user_id', 'item_id', 'predict']]
    test = test.sort_values(['user_id', 'predict'], ascending=[1, 0])
    test = test.groupby(['user_id']).head(3)
    # test = test.drop_duplicates('user_id')
    test = test[['user_id', 'item_id']]
    test.to_csv('submit.csv', index=None)
    print(pred_ans)
