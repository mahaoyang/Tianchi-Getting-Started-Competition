from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from spark_session import spark
from tqdm import trange, tqdm
import pandas as pd
import time
import os


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H")))


def trans_item(item_res):
    return item_res[0]


udf_timestamp = udf(timestamp, IntegerType())
udf_trans_item = udf(trans_item, IntegerType())

df = pd.read_csv("data/tianchi_fresh_comp_train_user.csv", low_memory=False)[:10000]
df = df[['user_id', 'item_id', 'behavior_type', 'item_category', 'time']]
df['time'] = df['time'].apply(lambda x: timestamp(x), convert_dtype='int32')
print(len(df))
print(df.dtypes)
ratings = spark.createDataFrame(df)
print('create spark DateFrame')
# ratings = ratings.withColumn('time', udf_timestamp('time'))
# print('trans time by udf')
(training, test) = ratings.randomSplit([0.8, 0.2], seed=123)
print('build train data success')
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(
    numItemBlocks=16, rank=1, maxIter=1, regParam=1, implicitPrefs=False, alpha=1,
    nonnegative=False,
    userCol="user_id",
    itemCol="item_id",
    ratingCol="behavior_type",
    coldStartStrategy="drop")
model = als.fit(training)
# model.save('acie.model')

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="behavior_type",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
# userRecs = model.recommendForAllUsers(1)
# # Generate top 10 user recommendations for each movie
# movieRecs = model.recommendForAllItems(30)
# userRecs = userRecs.withColumn('recommendations', udf_trans_item('recommendations'))
# userRecs.write.csv('tianchi_mobile_recommendation_predict.csv', header=None)
# userRecs.show(truncate=False)
# userRecs = userRecs.collect()
# ur = []
# for i in userRecs:
#     for ii in i[1]:
#         ur.append({'user_id': i[0], 'item_id': ii[0]})
# ur = pd.DataFrame(ur)
# ur.to_csv('tianchi_mobile_recommendation_predict.csv', index=None, encoding='utf-8')
# userRecs.show()
# movieRecs.show()

# # Generate top 10 movie recommendations for a specified set of users
users = pd.DataFrame(list(set(df['user_id'].values.tolist())))[:5000]
users.columns = ['user_id']
batch_size = 1000
for step in trange(1, len(users) + 1, batch_size):
    user = users[-(step + batch_size): -step]
    flag = user.values.tolist()
    if flag:
        user = spark.createDataFrame(user)
        userSubsetRecs = model.recommendForUserSubset(user, 10)
        # Generate top 10 user recommendations for a specified set of movies
        # movies = ratings.select(als.getItemCol()).distinct().limit(3)
        # movieSubSetRecs = model.recommendForItemSubset(movies, 30)

        userSubsetRecs = userSubsetRecs.toPandas()
        userSubsetRecs = userSubsetRecs.values.tolist()
        ur = []
        for i in userSubsetRecs:
            for ii in i[1]:
                ur.append({'user_id': i[0], 'item_id': ii[0]})
        ur = pd.DataFrame(ur)
        # name = 'tianchi_mobile_recommendation_predict.csv'
        name = 'cf_predict.csv'
        if not os.path.exists(name):
            ur.to_csv(name, index=None, encoding='utf-8')
        else:
            ur.to_csv(name, index=None, header=None, mode='a+',
                      encoding='utf-8')

        # userSubsetRecs.show(truncate=False)
        # movieSubSetRecs.show(truncate=False)

spark.stop()
