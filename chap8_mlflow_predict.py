
#%%
import mlflow
import mlflow.spark
import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("mlflow_predict").getOrCreate()
filename = "bank-full.csv"
target_variable_name = "y"
from pyspark.sql import functions as F
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df = df.withColumn('label', F.when(F.col("y") == 'yes', 1).otherwise(0))
df = df.drop('y')
train, test = df.randomSplit([0.7, 0.3], seed=12345)

for k, v in df.dtypes:
    if v not in ['string']:
        print(k)
        
df = df.select(['age', 'balance', 'day', 'duration',
                'campaign', 'pdays', 'previous', 'label'
                ])


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
# assemble individual columns to one column - 'features'
def assemble_vectors(df, features_list, target_variable_name):
    stages = []
    #assemble vectors
    assembler = VectorAssembler(inputCols=features_list, outputCol='features')
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
# apply the function on our dataframe
assembled_test_df = assemble_vectors(test, features_list, 'label')
print(sys.argv[1])

# model info from arg
model_uri = sys.argv[1]
print("model_uri:", model_uri)
model = mlflow.spark.load_model(model_uri)
print("model.type:", type(model))
predictions = model.transform(assembled_test_df)
print("predictions.type:", type(predictions))
predictions.printSchema()
df = predictions.select("rawPrediction", "probability", "label", "features")
df.show(5, False)


#%%  ##### Automated Machine Learning Pipelines #####
## model requirements
# 1. data manipulation

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import IndexToString
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

# 1. missing value
def missing_value_calculation(X, miss_per=0.75):
    missing = ([X.select(F.count(F.when(F.isnan(c) | F.col(c).isNull(), c))
                        .alias(c) for c in X.columns)
                ]
               )
    missing_len = X.count()
    final_missing = missing.toPandas().transpose()
    final_missing.reset_index(inplace=True)
    final_missing.rename(columns={0:'missing_count'}, inplace=True)
    final_missing['missing_percentage'] = final_missing['missing_count']/missing_len
    vars_selected = final_missing['index'][final_missing['missing_percentage']<=miss_per]
    return vars_selected


# 2. Metadata categorization

def identify_variable_type(X):
    l = X.dtypes
    char_vars = []
    num_vars = []
    for i in l:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars


# 3. categorical to numerical using label encoders

def categorical_to_index(X, char_vars):
    chars = X.select(char_vars)
    indexers = [StringIndexer(inputCol=column, 
                              outputCol=column+"_index",
                              handleInvalid="keep"
                              ) 
                for column in chars.columns
                ]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(chars)
    X = char_labels.transform(X)
    return X, char_labels

# 4. umpute numerical columns with specific value

def numerical_imputation(X, num_vars, impute_with=0):
    X = X.fillna(impute_with, subset=num_vars)
    return X


    




