
#%%
import pyspark
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
import sys
import time
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

#%%

spark = SparkSession.builder.appName("mlflow_example").getOrCreate()

filename = "bank-full.csv"
target_variable_name = "y"

df = spark.read.csv(filename, header=True, inferSchema=True, sep=";")
df = df.withColumn("label", F.when(F.col("y")=="yes", 1).otherwise(0))
df = df.drop("y")
train, test = df.randomSplit([0.7, 0.3], seed=12345)

for k, v in df.dtypes:
    if v not in ["string"]:
        print(k)
        
df  = df.select(['age', 'balance', 'day', 'duration', 'campaign', 'pdays',
                 'previous', 'label']
                )












