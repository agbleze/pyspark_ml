
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

reviews = df.rdd.map(lambda x: x['Review Text']).filter(lambda x: x is not None)

StopWords = stopwords.words("english")

tokens = (reviews.map(lambda document: document.strip().lower())
          .map(lambda document: re.split(" ", document))
          .map(lambda word: [x for x in word if x.isalpha()])
          .map(lambda word: [x for x in word if len(x) > 3])
          .map(lambda word: [x for x in word if x not in StopWords])
          .zipWithIndex()
          )


# convert rdd to dataframe
df_txts = spark.createDataFrame(tokens, ["list_of_words", "index"])

# TF
cv = CountVectorizer(inputCol="raw_features", outputCol="features",
                     vocabSize=5000, minDF=10
                     )
cvmodel = cv.fit(df_txts)
result_cv = cvmodel.transform(df_txts)


# IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idfModel = idf.fit(result_cv)
result_tfidf = idfModel.transform(result_cv)


num_topics = 10
max_iterations = 100
lda_model = LDA.train(result_tfidf.select("index", "features")
                      .rdd.mapValues(Vectors.fromML).map(list), k=num_topics,
                      maxIterations=max_iterations
                      )


wordNumbers = 5
data_topics = lda_model.describeTopics(maxTermsPerTopic=wordNumbers)
vocabArray = cvmodel.vocabulary
topicIndices = spark.sparkContext.parallelize(data_tp)

def topic_render(topic):
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()
for topic in range(len(topics_final)):
    print("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print(term)
    print('\n')











