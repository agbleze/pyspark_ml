
#%% ######  Latent Dirichlet Allocation #####
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.mllib.clustering import LDA, LDAModel
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords



spark = SparkSession.builder.appName("LDA").getOrCreate()

# print pyspark and python version

import sys
print("Python version: " + sys.version)
print("Spark version: " + spark.version)

# read data

file_location = "lda_data.csv"
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"

df = (spark.read.format(file_type)
      .option("inferSchema", infer_schema)
      .option("header", first_row_is_header)
      .load(file_location)
      )

# print metadata

df.printSchema()

# count data
df.count()


#%% preprocess text data








