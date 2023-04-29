
#%%

filename = "bank-full.csv"

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

#%%
spark = SparkSession.builder.getOrCreate()
data = spark.read.csv(filename, header=True, inferSchema=True, 
                      sep=';')

data.show()

#%% assemble feature vectors

def AssembleVectors(df, features_list, target_variable_name):
    assembler = VectorAssembler(inputCols=features_list,
                                outputCol='features'
                                )
    stages = [assembler]
    
    selectedCols = [target_variable_name, 'features'] + features_list
    
    # use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    
    # apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)
    return df


#%%  ####### Linear regression  #######

linear_df = data.select(['age', 'balance', 'day', 
                        'duration', 'campaign',
                        'pdays', 'previous'
                        ]
                       )
target_variable_name = 'balance'

features_list = linear_df.columns
features_list.remove(target_variable_name)

#%% apply the function on df

df = AssembleVectors(linear_df, features_list, target_variable_name)

#%% fot the regression model
from pyspark.ml.regression import LinearRegression

reg = LinearRegression(featuresCol='features', labelCol='balance')
reg_model = reg.fit(df)

# view coefficients and intercepts of each variable

import pandas as pd

for k, v in df.schema["features"].metadata["ml_attr"]["attrs"].items():
    features_df = pd.DataFrame(v)
    
#%% print coeffcient and intercept
print(reg_model.coefficients, reg_model.intercept)
features_df['coefficients'] = reg_model.coefficients

#%% prediction results
pred_results = reg_model.transform(df)

#%% ######  Variance Inflation Factor (VIF) ######

def vif_calculator(df, features_list):
    vif_list = []
    for i in features_list:
        temp_features_list = features_list.copy()
        temp_features_list.remove(i)
        temp_target = i
        assembler = VectorAssembler(inputCols=temp_features_list,
                                    outputCol='features'
                                    )
        temp_df = assembler.transform(df)
        reg = LinearRegression(featuresCol='features', 
                               labelCol=i
                               )
        reg_model = reg.fit(temp_df)
        temp_vif = 1/(1 - reg_model.summary.r2)
        vif_list.append(temp_vif)
    return vif_list


#%%
features_df['vif'] = vif_calculator(linear_df, features_list)
print(features_df)


#%% ######## Logistic Regression ########

target_variable_name = "y"
logistic_df = data.select(["age",
                           "balance",
                           "day", "duration", "campaign", "pdays",
                           "previous", "y"
                           ]
                          ) 
features_list = logistic_df.columns
features_list.remove(target_variable_name)

df = AssembleVectors(logistic_df, features_list, target_variable_name)

#%%

import numpy as np
from pyspark.ml.classification import LogisticRegression


binary_clf = LogisticRegression(featuresCol='features', labelCol='y',
                                family='binomial')

multinomial_clf = LogisticRegression(featuresCol='features',
                                     labelCol='y',
                                     family='multinomial'
                                     )

binary_clf_model = binary_clf.fit(df)
np.set_printoptions(precision=3, suppress=True)

print(binary_clf_model.coefficients)
print(binary_clf_model.intercept)

#%% 
multinomial_clf_model = multinomial_clf.fit(df)

np.set_printoptions(precision=4, suppress=True)
print(multinomial_clf_model.coefficentMatrix)
print(multinomial_clf_model.interceptVector)




# %%  ######### Decision tree algorithm ######

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer


#%%

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


#%% ##########
binary_df, y_label = CategoryToIndex(df=df, char_vars='y')



#%%

clf = DecisionTreeClassifier(featuresCol='features', 
                             labelCol='y_index', 
                             impurity='gini'
                             )


clf_model = clf.fit(binary_df)


clf2 = DecisionTreeClassifier(featuresCol='features',
                              labelCol='y_index',
                              impurity='entropy'
                              )
clf_model2 = clf2.fit(binary_df)

#%%

clf_model.transform(binary_df)  # predictions

#%%  ### gini feature importance ###

print(clf_model.featureImportances)

print(clf_model2.featureImportances)


#%% ### Decision Tree Regression ###

from pyspark.ml.regression import DecisionTreeRegressor

reg = DecisionTreeRegressor(featuresCol='features', 
                            labelCol='balance',
                            impurity='variance'
                            )

continuous_df = binary_df

#%%
reg_model = reg.fit(continuous_df)

print(reg_model.featureImportances)

#%%

clf_model.toDebugString

#%%
reg_model.toDebugString






# %%
