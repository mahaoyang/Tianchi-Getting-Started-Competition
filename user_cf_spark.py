from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from spark_session import spark
import pandas as pd
import time


def timestamp(t):
    return int(time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")))


df = pd.read_csv("data/Antai_AE_round1_train_20190626.csv", low_memory=False)
df['buyer_country_id'] = df['buyer_country_id'].map(lambda x: 0 if 'xx' in str(x) else 1)
df['create_order_time'] = df['create_order_time'].map(lambda x: timestamp(x))
print(len(df))
ratings = spark.createDataFrame(df)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(maxIter=4, regParam=0.01, userCol="buyer_admin_id", itemCol="item_id", ratingCol="irank",
          coldStartStrategy="drop")
model = als.fit(training)
model.save('acie.model')

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="irank",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(30)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(30)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 30)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 30)

spark.stop()
