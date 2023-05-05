
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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

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
        
df  = df.select(['age', 'balance', 'day', 'duration', 
                 'campaign', 'pdays',
                 'previous', 'label'
                 ]
                )

def AssembleVectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list,
    outputCol='features')
    stages = [assembler]
    #select all the columns + target + newly created 'features' column
    selectedCols = [target_variable_name, 'features']
    #use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    #assembler model
    assembleModel = pipeline.fit(df)
    #apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)
    return df

#exclude target variable and select all other feature vectors
features_list = df.columns
#features_list = char_vars #this option is used only for ChiSqselector
features_list.remove('label')
# apply the function on our DataFrame
assembled_train_df = AssembleVectors(train, features_list, 'label')
assembled_test_df = AssembleVectors(test, features_list, 'label')


print(sys.argv[1])
print(sys.argv[2])

maxBinsVal = float(sys.argv[1]) if len(sys.argv) > 3 else 20
maxDepthVal = float(sys.argv[2]) if len(sys.argv) > 3 else 3

with mlflow.start_run():
    stages_tree = []
    classifier = RandomForestClassifier(labelCol='label', 
                                        featuresCol='features',
                                        maxBins=maxBinsVal, 
                                        maxDepth=maxDepthVal
                                        )
    stages_tree = [classifier]
    pipeline_tree = Pipeline(stages=stages_tree)
    print("Running RFModel")
    RFmodel = pipeline_tree.fit(assembled_train_df)
    print("Completed training RFModel")
    predictions = RFmodel.transform(assembled_test_df)
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    mlflow.log_param("maxBins", maxBinsVal)
    mlflow.log_param("maxDepth", maxDepthVal)
    mlflow.log_metric("ROC", evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"}))
    mlflow.spark.log_model(RFmodel, "spark-model")
    
    
    
    



