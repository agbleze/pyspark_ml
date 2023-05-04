
#%%
from pyspark.sql.types import *
from pyspark.sql import SparkSession



#%% ########  Bisecting kmeans  ########

file_location = "cluster_data.csv"
file_type = "csv"
infer_schema = False
first_row_is_header = "true"


spark = SparkSession.builder.getOrCreate()

df = (spark.read.format(file_type)
      .option("inferSchema", infer_schema)
      .option("header", first_row_is_header)
      .load(file_location)
      )


# print metadata
df.printSchema()

# casting multiple variables
## Identifying and assigning lists of variables

float_vars = list(set(df.columns) - set(['CUST_ID']))

for column in float_vars:
    df = df.withColumn(column, df[column].cast(FloatType()))
    
# imputing data

from pyspark.ml.feature import Imputer

input_cols = list(set(df.columns) - set(['CUST_ID']))

imputer = Imputer(inputCols=input_cols, 
        outputCol=["{}_imputed".format(c) for c in input_cols]
    )

# Apply the transformation
df_imputed = imputer.fit(df).transform(df)

# Dropping the original columns as we created the _imputed columns
df_imputed = df_imputed.drop(*input_cols)

# rename input cols
new_column_name_list = list(map(lambda x: x.replace("_imputed", ""),
                                df.columns
                                )
                            )

df_imputed = df_imputed.toDF(*new_column_name_list)


#%% Data preparation

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline

# variable not rquired in segmentation analysis
ignore = ['CUST_ID']

assembler = VectorAssembler(inputCols=[x for x in df.columns if x not in ignore],
                            outputCol='features'
                            )

normalizer = Normalizer(inputCol="features", outputCol="normFeatures",
                        p=1.0
                        )

# defining the pipeline

pipeline = Pipeline(stages= [assembler, normalizer])

transformations = pipeline.fit(df_imputed)
df_updated = transformations.transform(df_imputed)

# Building the model

from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

bkm = BisectingKMeans().setK(2).setSeed(1)
model = (bkm.fit(df_updated.select("normFeatures")
                 .withColumnRenamed("normFeatures", "features"))
         )

# make predictions

predictions = model.transform(df_updated.select("normFeatures")
                              .withColumnRenamed("normFeatures", "features")
                              )

evaluator = ClusteringEvaluator()


silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

print("Cluster Centers: ")
centers = model.clusterCenters()

for center in centers:
    print(center)
    

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sil_coeff = []
num_clusters = []

for iter in range(2, 8):
    bkm = BisectingKMeans().setK(2).setSeed(1)
    model = (bkm.fit(df_updated.select("normFeatures")
                    .withColumnRenamed("normFeatures", "features"))
            )

# make predictions
    predictions = model.transform(df_updated.select("normFeatures")
                                .withColumnRenamed("normFeatures", "features")
                                )
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    sil_coeff.append(silhouette)
    num_clusters.append(iter)
    print("Silhouette with squared euclidean distance for "+str(iter)
          + "cluster solution = " + str(silhouette))
    

df_viz = pd.DataFrame(zip(num_clusters, sil_coeff), columns=["num_clusters", "silhouette_score"])
sns.lineplot(x = "num_clusters", y = "silhouette_score", data=df_viz)
plt.title("Bisecting k-means : Silhouette scores")
plt.xticks(range(2, 8))
plt.show()



#%% #######  K Means ########
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Train a k-means model

kmeans = KMeans().setK(2).setSeed(1003)
model = kmeans.fit(df_updated.select("normFeatures").withColumnRenamed("normFeatures", "features"))

predictions = model.transform(df_updated.select("normFeatures").withColumnRenamed("normFeatures", "features"))

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

centers = model.clusterCenters()
print("Cluster Centers: ")

for center in centers:
    print(center)

sil_coeff = []
num_clusters = []

for iter in range(2, 8):
    kmeans = KMeans().setK(iter).setSeed(1003)
    model = kmeans.fit(df_updated.select("normFeatures").withColumnRenamed("normFeatures", "features"))
    predictions = model.transform(df_updated.select("normFeatures").withColumnRenamed("normFeatures", "features"))
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    sil_coeff.append(silhouette)
    num_clusters.append(iter)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

df_viz=pd.DataFrame(zip(num_clusters,sil_coeff), columns=['num_clusters','silhouette_score'])
sns.lineplot(x = "num_clusters", y = "silhouette_score", data=df_viz)
plt.title('k-means : Silhouette scores')
plt.xticks(range(2, 8))
plt.show()
    










    