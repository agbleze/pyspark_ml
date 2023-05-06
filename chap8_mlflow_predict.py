
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


# 5. Rename categorical columns

def rename_columns(X, char_vars):
    mapping = dict(zip([i + '_index' for i in char_vars], char_vars))
    X = X.select([col(c).alias(mapping.get(c, c))
                  for c in X.columns
                  ]
                )
    return X


# 6. combining features and labels

def join_features_and_target(X, Y):
    X = X.withColumn('id', F.monotonically_increasing_id())
    Y = Y.withColumn('id', F.monotonically_increasing_id())
    joinedDF = X.join(Y, 'id', 'inner')
    joinedDF = joinedDF.drop('id')
    return joinedDF


# 7. Data splitting to training, testing, and validation

def train_valid_test_split(df, train_size=0.4, valid_size=0.3, seed=12345):
    train, valid, test = df.randomSplit([train_size, valid_size, 1-train_size-valid_size], seed=12345)
    return train, valid, test

# 8. Assembling vectors

def assembled_vectors(train,list_of_features_to_scale,target_column_name):
#
    stages = []
    assembler = VectorAssembler(inputCols=list_of_features_to_scale,
                                outputCol='features')
    stages=[assembler]
    selectedCols = [target_column_name,'features'] + list_of_features_to_scale
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(train)
    train = assembleModel.transform(train).select(selectedCols)
    return train 

# 9. scaling input variables

def scaled_dataframes(train, valid, test, list_of_features_to_scale,
                      target_column_name
                      ):
    assembler = VectorAssembler(inputCols=list_of_features_to_scale,
                                outputCols='assembled_features'
                                )
    scaler = StandardScaler(inputCol=assembler.getOutputCol(),
                            outputCol='features'
                            )
    stages = [assembler, scaler]
    selectedCols = [target_column_name, 'features'] + list_of_features_to_scale
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train)
    
    train = pipelineModel.transform(train).select(selectedCols)
    valid = pipelineModel.transform(valid).select(selectedCols)
    test = pipelineModel.transform(test).select(selectedCols)
    return train, valid, test, pipelineModel


#%% ####### FEATURE SELECTION #####
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw_feature_importance(user_id, mdl_ltrl, importance_df):
    importance_df = importance_df.sort_values('Importance_Score')
    plt.figure(figsize=(15,15))
    plt.title('Feature Importance')
    plt.barh(range(len(importance_df['Importance_Score'])), 
             importance_df['Importance_Score'],
             align='center'
             )
    plt.yticks(range(len(importance_df['Importance_Score'])),
               importance_df['name']
               )
    plt.ylabel('Variable Importance')
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' +
                'Features selected for modeling.png', bbox_inches='tight'
                )
    plt.close()
    return None


def save_feature_importance(user_id, mdl_ltrl, importance_df):
    importance_df.drop('idx', axis=1, inplace=True)
    importance_df = importance_df[0:30]
    importance_df.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl +
                            '/' + 'feature_importance.xlsx'
                            )
    draw_feature_importance(user_id, mdl_ltrl, importance_df)
    return None


def ExtractFeatureImp(featureImp, dataset, featuresCol="features"):
    list_extract = []
    for i in dataset.schema[featuresCol].metadat["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['Importance_Score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return (varlist.sort_values('Importance_Score', ascending=False))


#%% ####### MODEL BUILDING #######

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
import joblib
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel

def logistic_model(train, x, y):
    lr = LogisticRegression(featuresCol=x, labelCol=y, maxIter=10)
    lrModel = lr.fit(train)
    return lrModel


def randomForest_model(train, x, y):
    rf = RandomForestClassifier(featuresCol=x, labelCol=y, numTrees=10)
    rfModel = rf.fit(train)
    return rfModel


from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import GBTClassificationModel


def gradientBoosting_model(train, x, y):
    gb = GBTClassifier(featuresCol=x, labelCol=y, maxIter=10)
    gbModel = gb.fit(train)
    return gbModel


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassificationModel

def decisionTree_model(train, x, y):
    dt = DecisionTreeClassifier(featuresCol=x, labelCol=y, maxDepth=5)
    dtModel = dt.fit(train)
    return dtModel










# %%
