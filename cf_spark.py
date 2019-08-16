from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from spark_session import spark
import pandas as pd
import time


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H")))


udf_timestamp = udf(timestamp, IntegerType())

df = pd.read_csv("data/tianchi_fresh_comp_train_user.csv", low_memory=False)
df = df[['user_id', 'item_id', 'behavior_type', 'item_category', 'time']]
df['time'] = df['time'].apply(lambda x: timestamp(x), convert_dtype='int64')
print(len(df))
print(df.dtypes)
ratings = spark.createDataFrame(df)
# ratings = ratings.withColumn('time', udf_timestamp('time'))
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(
    numItemBlocks=16, rank=100, maxIter=1000, regParam=1, implicitPrefs=False, alpha=1,
    nonnegative=False,
    userCol="user_id",
    itemCol="item_id",
    ratingCol="behavior_type",
    coldStartStrategy="drop")
model = als.fit(training)
model.save('acie.model')

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="behavior_type",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(30)
# # Generate top 10 user recommendations for each movie
# movieRecs = model.recommendForAllItems(30)

userRecs = userRecs.toPandas()
userRecs = userRecs.values.tolist()
ur = []
for i in userRecs:
    for ii in i[1]:
        ur.append({'user_id': i[0], 'item_id': ii[0]})
ur = pd.DataFrame(ur)
ur.to_csv('tianchi_mobile_recommendation_predict.csv', index=None, encoding='utf-8')
# userRecs.show()
# movieRecs.show()

# # Generate top 10 movie recommendations for a specified set of users
# users = ratings.select(als.getUserCol()).distinct().limit(3)
# userSubsetRecs = model.recommendForUserSubset(users, 30)
# Generate top 10 user recommendations for a specified set of movies
# movies = ratings.select(als.getItemCol()).distinct().limit(3)
# movieSubSetRecs = model.recommendForItemSubset(movies, 30)

# userSubsetRecs = userSubsetRecs.toPandas()
# userSubsetRecs = userSubsetRecs.values.tolist()
# ur = []
# for i in userSubsetRecs:
#     for ii in i[1]:
#         ur.append({'user_id': i[0], 'item_id': ii[0]})
# ur = pd.DataFrame(ur)
# ur.to_csv('tianchi_mobile_recommendation_predict.csv', index=None, encoding='utf-8')

# userSubsetRecs.show(truncate=False)
# movieSubSetRecs.show(truncate=False)


spark.stop()
