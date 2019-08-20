#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import os
from pyspark import SparkConf
from pyspark.sql import SparkSession

if os.environ.get('OS') == 'windows':
    os.environ['SPARK_HOME'] = 'E:/spark-2.4.3-bin-hadoop2.7'
    os.environ[
        "PYSPARK_PYTHON"] = "C:/Users/99263/PycharmProjects/ML-With-PySpark/venv/Scripts/python.exe"
else:
    os.environ['SPARK_HOME'] = '/usr/local/Cellar/apache-spark/spark-2.4.3-bin-hadoop2.7'
    os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.7"


def conf():
    sc_conf = SparkConf()
    # sc_conf.setMaster('spark://master:7077')
    sc_conf.setAppName('my-app')
    sc_conf.set('spark.executor.memory', '64g')  # executor memory是每个节点上占用的内存。每一个节点可使用内存
    sc_conf.set("spark.executor.cores",
                '16')  # spark.executor.cores：顾名思义这个参数是用来指定executor的cpu内核个数，分配更多的内核意味着executor并发能力越强，能够同时执行更多的task
    sc_conf.set('spark.cores.max',
                '16')  # spark.cores.max：为一个application分配的最大cpu核心数，如果没有设置这个值默认为spark.deploy.defaultCores
    # sc_conf.set('spark.logConf', 'True')  # 当SparkContext启动时，将有效的SparkConf记录为INFO。
    return sc_conf


spark = SparkSession \
    .builder \
    .appName("pysparkpro") \
    .config(conf=conf()) \
    .getOrCreate()

from pyspark.context import SparkContext

sc = SparkContext.getOrCreate(conf())
if os.environ.get('OS') == 'windows':
    sc.setCheckpointDir('C:/Users/99263/tmp/checkpoints')
else:
    sc.setCheckpointDir('/tmp/checkpoints')
