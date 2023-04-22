
#%%

from pyspark.sql import SparkSession


#%%
filename = 'bank-full.csv'
target_variable_name = 'y'

spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df.show()

#%% ### length of data ###

df.count()

#%% describe data ###

df.describe().toPandas()  # missing and cardinality followup

#%% type of each variable ###

df.dtypes
df.printSchema()

#%%  ### 
df.groupBy('education').count().show()

#%% ### target count ###
df.groupBy(target_variable_name).count().show()

#%% column aggregations
from pyspark.sql.functions import *

df.groupBy(target_variable_name).agg({'balance': 'avg', 'age': 'avg'}).show()

#%%  #####  cardinality check ####

def CardinalityCalculation(df, cut_off=1):
    cardinality = df.select(*[approxCountDistinct(c).alias(c)
                              for c in df.columns
                              ]
                            )
    ## convert to pandas for efficient calculations
    final_cardinality_df = cardinality.toPandas().transpose()
    final_cardinality_df.reset_index(inplace=True)
    final_cardinality_df.rename(columns={0:'Cardinality'}, inplace=True)

    # select variables with cardinality of 1
    vars_selected = final_cardinality_df['index']
    [final_cardinality_df['Cardinality'] <= cut_off]
    
    return final_cardinality_df, vars_selected

# %%

cardinality_df, cardinality_vars_selected = CardinalityCalculation(df)

#%% ##### Missing values check #####

def MissingCalculation(df, miss_percentage=0.80):
    missing = df.select(*[count(when(isnan(c) | col(c).isNull(), c)).alias(c)
                          for c in df.columns]
                        )
    length_df = df.count()
    
    final_missing_df = missing.toPandas().transpose()
    final_missing_df.reset_index(inplace=True)
    final_missing_df.rename(columns={0: 'missing_count'}, inplace=True)
    final_missing_df['missing_percentage'] = final_missing_df['missing_count']/length_df
    
    # select variables with cardinality of 1
    vars_selected = final_missing_df['index'][final_missing_df['missing_percentage']>=miss_percentage]
    return final_missing_df, vars_selected


#%% implement missing 
missing_df, missing_var_selected = MissingCalculation(df)
    
#%% ##### identify variable types #####

def VariableType(df):
    vars_list = df.dtypes
    char_vars = []
    num_vars = []
    
    for i in vars_list:
        if i[1] in ('string'):
            char_vars.append(i[0])
        else:
            num_vars.append(i[0])
    return char_vars, num_vars


#%% impplement varaible type func
char_vars, num_vars = VariableType(df)
    
 #%%  ##### Apply StringIndexer to character columns #####
 
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# convert each category column to index

def CategoryToIndex(df, char_vars):
    
    char_df = df.select(char_vars)
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_index",
                              handleInvalid="keep")
                for c in char_df.columns
                ]
    pipeline = Pipeline(stages=indexers)
    char_labels = pipeline.fit(char_df)
    df = char_labels.transform(df)
    return df, char_labels


#%%  ### apply category to index func  ###

df, char_labels = CategoryToIndex(df, char_vars)

#%%
df = df.select([c for c in df.columns if c not in char_vars]) 
 
#%%

def RenameColumns(df, char_vars):
    mapping = dict(zip([i + '_index' for i in char_vars], char_vars))
    df = df.select([col(c).alias(mapping.get(c, c)) for c in df.columns])
    return df

#%% apply RenameColumns
df = RenameColumns(df, char_vars)   

#%%  ######  Assemble Features ######

from pyspark.ml.feature import VectorAssembler

# assemble individual cols to 1 col - features
def AssembleVectors(df, feature_list, target_variable_name):
    stages = []
    
    assembler = VectorAssembler(inputCols=feature_list, outputCol='features')
    stages = [assembler]
    selectedCols = [target_variable_name, 'features'] + feature_list
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    
    # apply assemble model on data
    df = assembleModel.transform(df).select(selectedCols)
    return df

#%%

features_list = df.columns
features_list.remove(target_variable_name)

df = AssembleVectors(df, features_list, target_variable_name)

#%%
import pandas as pd

for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    features_df = pd.DataFrame(v)

#%% ###### PCA  ########

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
no_of_components = 3
pca = PCA(k=no_of_components, inputCol="features", outputCol="pcaFeatures")

model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)

#%% 

model.pc.toArray()

#%%
model.explainedVariance

#%% plot PCA ###
import numpy as np
import matplotlib.pyplot as plt
x = []

for i in range(0, len(model.explainedVariance)):
    x.append('PC' + str(i +1))
y = np.array(model.explainedVariance)
z = np.cumsum(model.explainedVariance)
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.bar(x, y)
plt.plot(x, z)


#%% ###### assemble and scale individual columns to 1 col

from pyspark.ml.feature import StandardScaler

def ScaledAssembleVectors(df, features_list, target_variable_name):
    #stages = []
    
    assembler = VectorAssembler(inputCols=features_list,
                                outputCol='assembled_features'
                                )
    scaler = StandardScaler(inputCol=assembler.getOutputCol(),
                            outputCol='features2'
                            )
    stages = [assembler, scaler]
    
    # select all columns + target + newly created 'features' col
    selectedCols = [target_variable_name, 'features2'] + features_list
    
    pipeline = Pipeline(stages=stages)
    scaledAssembleModel = pipeline.fit(df)
    
    df = scaledAssembleModel.transform(df).select(selectedCols)
    return df


#%% 

features_list = df.columns
features_list.remove(target_variable_name)
df = ScaledAssembleVectors(df, features_list, target_variable_name)

#%%

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

pca = PCA(k=3, inputCol="features2", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
# %%

model.explainedVariance

# %%  #####  Singular Value Decomposition  #####

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix


#%% convert DF TO RDD

df_svd_vector = df.rdd.map(lambda x: x['features'].toArray())

mat = RowMatrix(df_svd_vector)

# Compute top 5 singular values and corresponding vectors
svd = mat.computeSVD(5, computeU=True)

U = svd.U # U factor in rowmatrix
s = svd.s # singular values stored on local dense vector
V = svd.V # V factor is a local dense matrix


#%%  #### Built-in Variable Selection Process: with target ########

### chi-square selection

from pyspark.ml.feature import ChiSqSelector

features_list = char_vars

selector = ChiSqSelector(numTopFeatures=6, featuresCol='features',
                         outputCol='selectedFeatures', labelCol='y'
                         )

chi_selector = selector.fit(df)

result = chi_selector.transform(df)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())

print("Selected Indices: ", chi_selector.selectedFeatures)

features_df['chiq_importance'] = features_df['idx'].apply(lambda x: 
    1 if x in chi_selector.selectedFeatures else 0)

print(features_df)


#%% #######  Model-based feature selection ######















# %%
