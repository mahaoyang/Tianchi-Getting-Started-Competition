# -*-coding:utf8-*-
"""
author:david
date:2018****
lfm model train main function
"""

import pickle
import numpy as np
import pandas as pd
from pyspark.ml.clustering import LDA
from spark_session import spark


def lda_train():
    # Loads data.
    dataset = spark.read.format("libsvm").load("train.libsvm", numFeatures=4758484)

    # Trains a LDA model.
    lda = LDA(k=20, maxIter=200)
    model = lda.fit(dataset)

    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

    # Describe topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

    topics = model.describeTopics()
    topics_array = topics.select('termWeights').collect()
    topics_array = np.array([i[0] for i in topics_array])

    # Shows the result
    transformed = model.transform(dataset)
    transformed.show(truncate=False)

    user_vector = transformed.select('topicDistribution').collect()
    with open('idx.pickle', 'rb') as f:
        idx_item, item_idx, idx_user, user_idx, label, train = pickle.load(f)
    user_test = list(user_idx.keys())

    submit = []
    user_test_idx = []
    for uid in user_test:
        user_test_idx.append(user_idx.get(uid))
    for i in user_test_idx:
        item_rec = [idx_user[i]]
        user_vector[i] = i[0].toArray()
        sim = i.dot(topics_array) / (np.linalg.norm(i) * np.linalg.norm(topics_array))
        sim = np.argsort(-sim).tolist()
        [item_rec.append(idx_item[i]) for i in sim]
        submit.append(item_rec)
    df = pd.DataFrame(submit)
    df.to_csv('submit.csv', header=None, index=None)

    # Save
    model.save('lda.model')

    # Stop
    spark.stop()


if __name__ == "__main__":
    lda_train()
