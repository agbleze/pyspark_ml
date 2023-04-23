
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

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol=target_variable_name)

rf_model = rf.fit(df)
rf_model.featureImportances

#%%

import pandas as pd

for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    features_df = pd.DataFrame(v)
    
# temp output rf_output
rf_output = rf_model.featureImportances
features_df["Importance"] = features_df['idx'].apply(lambda x: rf_output[x]
                                                     if x in rf_output.indices else 0
                                                     )
# sort values based on descending importnace feature
features_df.sort_values("Importance", ascending=False, inplace= True)


features_df

#%% plot feature importance

features_df.sort_values(by="Importance", ascending=False, inplace=True)
plt.barh(features_df['name'], features_df['Importance'])
plt.title("Feature Importance plot")
plt.xlabel("Importance Score")
plt.ylabel("Variable Importance")


#%% ### custom-built variable selection process ###

## Weight of evidence (WOE) and Information Value(IV)

# default params

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

dataset = 2 # 1 OR 2
# Load Dataset

from pyspark.sql import functions as F

# load bank dataset if 1
if dataset == 1:
    filename = "bank-full.csv"
    target_variable_name = 'y'
    df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
    df = (df.withColumn(target_variable_name, 
                        F.when(df[target_variable_name]=='no', 0)
                        .otherwise(1)
                        )
          )
# load housing dataset if not 1
else:
    filename = "melb_data.csv"
    target_variable_name="type"
    df = spark.read.csv(filename, header=True, inferSchema=True, sep=',')
    df = (df.withColumn(target_variable_name, 
                        F.when(df[target_variable_name]=='h', 0)
                        .otherwise(1)
                        )
          )

df.show()

# target check
df.groupBy(target_variable_name).count().show()

# identify variable types and perform some operations

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

char_vars, num_vars = VariableType(df)

if dataset != 1:
    char_vars.remove('Address')
    char_vars.remove('SellerG')
    char_vars.remove('Date')
    char_vars.remove('Suburb')
        
num_vars.remove(target_variable_name)
final_vars = char_vars + num_vars

# WOE & IV 
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import VectorAssembler
import scipy.stats.stats as stats

# default values
# rho value for spearman correlation
custom_rho = 1

# max bins to start with and keep decreasing from this

max_bin = 20

# cal WOE and IV based on Pyspark output

def calculate_woe(count_df, event_df, min_value, max_value, feature):
   woe_df = pd.merge(left=count_df, right=event_df)
   woe_df['min_value'] = min_value
   woe_df['max_value'] = max_value
   woe_df['non_event'] = woe_df['count'] - woe_df['event']
   woe_df['event_rate'] = woe_df['event'] / woe_df['count']
   woe_df['nonevent_rate'] = woe_df['non_event'] / woe_df['count']
   woe_df['dist_event'] = woe_df['event'] / woe_df['event'].sum()
   woe_df['dist_nonevent'] = woe_df['non_event'] / woe_df['non_event'].sum()
   woe_df['woe'] = np.log(woe_df['dist_event'] / woe_df['dist_nonevent'])
   woe_df['iv'] = (woe_df['dist_event'] - woe_df['dist_nonevent'])*woe_df['woe']
   woe_df['varname'] = [feature] * len(woe_df)
   woe_df = woe_df[['varname', 'min_value', 'max_value', 'count',
                    'event', 'non_event', 'event_rate', 'nonevent_rate',
                    'dist_event', 'dist_nonevent', 'woe', 'iv'
                    ]]
   woe_df = woe_df.replace([np.inf, -np.inf], 0)
   woe_df['iv'] = woe_df['iv'].sum()
   return woe_df


#%% monotonic binning func implemented along with Spearman correlation 

def mono_bin(temp_df, feature, target, n = max_bin):
    r = 0
    while np.abs(r) < custom_rho and n > 1:
        try:
            # Quantile discretizer cut data into equal num
            qds = QuantileDiscretizer(numBuckets=n,
                                      inputCol=feature,
                                      outputCol='buckets',
                                      relativeError=0.01
                                      )
            bucketizer = qds.fit(temp_df)
            temp_df = bucketizer.transform(temp_df)
            
            # create correlation
            corr_df = (temp_df.groupby('buckets')
                       .agg({feature: 'avg', target: 'avg'})
                       .toPandas()
                       )
            corr_df.columns = ['buckets', feature, target]
            r, p = stats.spearmanr(corr_df[feature], corr_df[target])
            # spearman correlation
            n = n - 1
        except Exception as e:
            n = n - 1
        return temp_df
    
    
#%%  ####  execute WOE for all the variables

def execute_woe(df, target):
    count = -1
    for feature in final_vars:
        # excute if not target 
        if feature != target:
            count = count + 1
            temp_df = df.select([feature, target])
            
            # monotonic binning of numeric vars
            if feature in num_vars:
                temp_df = mono_bin(temp_df, feature, target,
                                   n = max_bin
                                   )
                grouped = temp_df.groupby('buckets')
            else:
                # grp cat vars
                grouped = temp_df.groupby(feature) 
                
            # count and event value for each group
            count_df = grouped.agg(F.count(target).alias('count')).toPandas()
            event_df = grouped.agg(F.sum(target).alias('event')).toPandas()
            
            # store min and max values of vars
            if feature in num_vars:
                min_value = grouped.agg(F.min(feature).alias('min')).toPandas()['min']
                max_value = grouped.agg(F.max(feature).alias('max')).toPandas()['max']
            else:
                min_value = count_df[feature]
                max_value = count_df[feature]
                
            # calculate WOE and IV
            temp_woe_df = calculate_woe(count_df, event_df, min_value, max_value, feature)
            
            # final dataset creation
            if count == 0:
                final_woe_df = temp_woe_df
            else:
                final_woe_df = final_woe_df.append(temp_woe_df, ignore_index=True)
                
        # separate IV dataset creation
        iv = pd.DataFrame({'IV': final_woe_df.groupby('varname').iv.max()})
        iv = iv.reset_index()
    return final_woe_df, iv

#%% invoke WOE and IV
output, iv = execute_woe(df, target_variable_name)


#%% ### CUSTOM TRANSFORMERS  ####

## CORRELATION ANALYSIS

from pyspark.mllib.stat import Statistics

correlation_type = 'pearson' # 'pearson', 'spearman'

# transform function

for k, v in df.schema['features'].metadata['ml_attr']['attrs'].items():
    features_df = pd.DataFrame(v)
column_names = list(features_df['name'])
df_vector = df.rdd.map(lambda x: x['features'].toArray())
matrix = Statistics.corr(df_vector, method=correlation_type)
corr_df = pd.DataFrame(matrix, columns=column_names, index=column_names)

final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort')).reset_index()
final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_vale'},
                     axis=1, inplace=True
                     )
final_corr_df = final_corr_df[final_corr_df['col1'] != final_corr_df['col2']]
correlation_cutoff = 0.65
final_corr_df[final_corr_df['correlation_value'] > correlation_cutoff]


#%%  ### write custom class ###########

from chap4_custom_correlation import CustomCorrelation
from pyspark.ml import Pipeline

features_list = df.columns
features_list.remove(target_variable_name)
stages = []

# assemble vectors
assembler = VectorAssembler(inputCols=features_list, outputCol='features')
custom_corr = CustomCorrelation(inputCol=assembler.getOutputCol())
stages = [assembler, custom_corr]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)

output, shortlisted_output = pipelineModel.transform(df)

#%%  ###### voting based selection #####
# in this case various feature selection method are used
# var selected my most methods is selected for modelling

#%%  ### decision tree ####

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol='features', labelCol=target_variable_name)
dt_model = dt.fit(df)

dt_output = dt_model.featureImportances
features_df['Decision_Tree'] = features_df['idx'].apply(lambda x: dt_output[x] if x in dt_output.indices else 0)


#%%  ### Gradient boosting  #####
from pyspark.ml.classification import GBTClassifier

gb = GBTClassifier(featuresCol='features', labelCol=target_variable_name)
gb_model = gb.fit(df)
gb_output = gb_model.featureImportances
features_df['Gradient Boosting'] = features_df['idx'].apply(lambda x: gb_output[x] if x in gb_output.indices else 0)


#% ### logistic regression ####

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol='features', labelCol=target_variable_name)
lr_model = lr.fit(df)

lr_output = lr_model.coefficients

# absolute value is used to convert neg coef
features_df['Logistic Regression'] = features_df['idx'].apply(lambda x: abs(lr_output[x]))


#%%  ###### random forest #####
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol='features', labelCol=target_variable_name)

rf_model = rf.fit(df)

rf_output = rf_model.featureImportances
features_df['Random Forest'] = features_df['idx'].apply(lambda x: rf_output[x] if x in rf_output.indices else 0)


#%% ###### voting-based selection ##### 

features_df.drop('idx', axis=1, inplace=True)
num_top_features = 7  # top n features from each algo
columns = ['Decision_Tree', 'Gradient_Boosting', 'Logistic Regression', 
           'Random Forest'
           ]
score_table = pd.DataFrame({}, [])
for i in columns:
    score_table[i] = features_df['name'].isin(list(features_df.nlargest(num_top_features, i)['name'])).astype(int)
    score_table['final_score'] = score_table.sum(axis=1)
    score_table.sort_values('final_score', ascending=0)








# %%
