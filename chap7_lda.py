
#%% ######  Latent Dirichlet Allocation #####
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LDA").getOrCreate()

# print pyspark and python version

import sys
print("Python version: " + sys.version)
print("Spark version: " + spark.version)

# read data

file_location = "lda_data.csv"





