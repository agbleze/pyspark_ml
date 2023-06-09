
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
import time

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


from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import MultilayerPerceptronClassificationModel

def neuralNetwork_model(train, x, y, feature_count):
    layers = [feature_count, feature_count*3, feature_count*2, 2]
    mlp = MultilayerPerceptronClassifier(featuresCol=x, labelCol=y,
                                         maxIter=100, layers=layers, blockSize=512,
                                         seed=12345
                                         )
    mlpModel = mlp.fit(train)
    return mlpModel



#%% ###### Metric Calculation #######
from pyspark.sql.types import DoubleType
from pyspark.sql import *
from pyspark.sql.functions import desc, udf
from pyspark.sql import functions as F
import sys
import builtins
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
import numpy as np
from pyspark import (SparkContext, HiveContext, 
                     Row, SparkConf
                     )




spark = (SparkSession.builder.appName("MLA_metric_calculator")
         .enableHiveSupport().getOrCreate()
         ) 
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

def highlight_max(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''), index=data.index,
                            columns=data.columns
                            )


def calculate_metrics(predictions, y, data_type):
    start_time4 = time.time()
    evaluator = BinaryClassificationEvaluator(labelCol=y,
                                              rawPredictionCol='probability'
                                              )
    auroc = evaluator.evaluate(predictions, {evaluator.metricName: "aureaUnderROC"})
    print(auroc)
    
    selectedCols = (predictions.select(F.col("probability"),
                                      F.col('prediction'), F.col(y)
                                      )
                    .rdd.map(lambda row: (float(row['probability'][1]),
                                          float(row['prediction']), float(row[y])))
                    .collect()

                     )
    y_score, y_pred, y_true = zip(*selectedCols)
    
    # calculate accuracy
    accuracydf = predictions.withColumn('acc', F.when(predictions.prediction==predictions[y], 1)
                                        .otherwise(0))
    accuracydf.createOrReplaceTempView("accuracyTable")
    RFaccuracy = spark.sql("select sum(acc)/count(1) as accuracy from accuracyTable").collect()[0][0]
    print('Accuracy calculated', RFaccuracy)
    
    # Build KS Table
    split1_udf = udf(lambda value: value[1].item(), DoubleType())
    
    if data_type in ['train', 'valid', 'test', 'oot1', 'oot2']:
        decileDF = predictions.select(y, split1_udf('probability').alias('probability'))
    else:
        decileDF = predictions.select(y, 'probability')
        
    decileDF = decileDF.withColumn('non_target', 1-decileDF[y])

    window = Window.orderBy(desc("probability"))
    decileDF = decileDF.withColumn('rownum', F.row_number().over(window))
    decileDF.cache()
    
    decileDF = decileDF.withColumn('rownum', decileDF['rownum'].cast('double'))
    
    window2 = Window.orderBy('rownum')
    RFbucketedData = decileDF.withColumn('deciles', F.ntile(10).over(window2))
    RFbucketedData = RFbucketedData.withColumn('deciles', RFbucketedData['deciles'].cast('int'))
    RFbucketedData.cache()
    
    print('KS calculation start')
    target_cnt=RFbucketedData.groupBy('deciles').agg(F.sum(y).alias('target')).toPandas()
    non_target_cnt = RFbucketedData.groupBy('deciles').agg(F.sum('non_target').alias('non_target')).toPandas()
    overall_cnt = RFbucketedData.groupBy('deciles').count().alias('Total').toPandas()
    overall_cnt = overall_cnt.merge(target_cnt, on='deciles', how='inner').merge(
        non_target_cnt, on='deciles', how='inner')
    overall_cnt = overall_cnt.sort_values(by='deciles', ascending=True)
    overall_cnt['Pct_target'] = (overall_cnt['target']/overall_cnt['count']) * 100
    overall_cnt['cum_target'] = overall_cnt.target.cumsum()
    overall_cnt['cum_non_target'] = overall_cnt.non_target.cumsum()
    overall_cnt['%Dist_Target'] = (overall_cnt['cum_target'] / overall_cnt.
                                    target.sum()
                                    )*100
    overall_cnt['%Dist_non_Target'] = (overall_cnt['cum_non_target'] /
                                       overall_cnt.non_target.sum()
                                    )*100
    overall_cnt['spread'] = builtins.abs(overall_cnt['%Dist_Target']-
    overall_cnt['%Dist_non_Target'])
    decile_table=overall_cnt.round(2)
    print("KS_Value =", builtins.round(overall_cnt.spread.max(),2))
    decileDF.unpersist()
    
    RFbucketedData.unpersist()
    print("Metrics calculation process Completed in : "+ " %s seconds" %
    (time.time() - start_time4))
    return auroc,RFaccuracy,builtins.round(overall_cnt.spread.max(),2), y_score, y_pred, y_true, overall_cnt



#%%  ######### validation and plot generation #########

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
import os
import pandas as pd
import seaborn as sns
from pandas import ExcelWriter
from metrics_calculator import *


def draw_roc_plot(user_id, mdl_ltrl, y_score, y_true, model_type, data_type):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label = 1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(str(model_type) + ' Model - ROC for ' + str(data_type) + ' data')
    plt.plot([0,1], [0,1], 'r--')
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Snesitivity)')
    plt.legend(loc = 'lower right')
    print('/home/' + user_id + '/' + 'mla_' + mdl_ltrl 
          + '/' + str(model_type) + '/' + str(model_type) + ' Model - ROC for ' + str(data_type) +
         'data.png'
         )
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' +
                str(model_type) + '/' + str(model_type) + ' Model - ROC for ' +
                str(data_type) + ' data.png', bbox_inches='tight'
                )
    plt.close()
    
    
def draw_ks_plot(user_id, mdl_ltrl, model_type):
    writer = ExcelWriter('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/'
                         + str(model_type) + '/KS_Charts.xlsx'
                        )
    
    for filename in glob.glob('/home/' + user_id + '/' + 'mla_' + mdl_ltrl
                                + '/' + str(model_type) + '/KS ' + str(model_type) + ' Model*.xlsx'
                                ):
        excel_file = pd.ExcelFile(filename)
        (_, f_name) = os.path.split(filename)
        (f_short_name, _) = os.path.splitext(f_name)
        for sheet_name in excel_file.sheet_names:
            df_excel = pd.read_excel(filename, sheet_name=sheet_name)
            df_excel = df_excel.style.apply(highlight_max, subset=['spread'], color='#e6b71e')
            df_excel.to_excel(writer, f_short_name, index=False)
            worksheet = writer.sheets[f_short_name]
            worksheet.condiditonal_format('C2:C11', {'type': 'data_bar',
                                                     'bar_color': '#34b5d9'}
                                          )
            worksheet.conditional_format('E2:E11', {'type': 'data_bar', 'bar_color': '#366fff'})
        os.remove(filename)
    writer.save()
    
    
#%% #### confusion matrix

def draw_confusion_matrix(user_id, mdl_ltrl, y_pred, y_true, model_type, data_type):
    AccuracyValue = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    PrecisionValue = metrics.precision_score(y_pred=y_pred, y_true=y_true)
    RecallValue = metrics.recall_score(y_pred=y_pred, y_true=y_true)
    F1Value = metrics.f1_score(y_pred=y_pred, y_true=y_true)
    
    plt.title(f'''{str(model_type)} Model - Confusion Matrix for 
              {str(data_type)} data Accuracy: {AccuracyValue} 
              Precision: {PrecisionValue} Recall: {RecallValue} F1 Score: {F1Value}'''
              )
    
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    
    
    print('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/' + str(model_type) + ' Model - Confusion Matrix for ' +
            str(data_type) + ' data.png'
        )
    
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' +
                str(model_type) + '/' + str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png', bbox_inches='tight'
                )
    plt.close()



#%% #### model validation ###

def model_validation(user_id, mdl_ltrl, data, y, model, model_type, data_type):
    start_time = time.time()
    
    pred_data = model.transform(data)
    print('model output predicted')
    
    roc_data, accuracy_data, ks_data, y_score, y_pred, y_true, decile_table = calculate_metrics(pred_data, data_type)
    draw_roc_plot(user_id, mdl_ltrl, y_score, y_true, model_type, data_type)
    decile_table.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl
                            + '/' + str(model_type) + '/KS ' + str(model_type) + ' Model ' +
                            str(data_type) + '.xlsx',index=False
                        )
    
    draw_confusion_matrix(user_id, mdl_ltrl, y_pred, y_true, model_type, data_type)
    print('Metric computed')
    l = [roc_data, accuracy_data, ks_data]
    end_time = time.time()
    print("Model validation process completed in :  %s seconds" % (end_time-start_time))
    return l



#%%  ###### model selection #####    

def select_model(user_id, mdl_ltrl, model_selection_criteria, dataset_to_use):
    df = pd.DataFrame({}, 
                      columns=['roc_train', 'accuracy_train', 'ks_train', 'roc_valid', 'accuracy_valid', 'ks_valid', 'roc_test',
                                'accuracy_test', 'ks_test', 'roc_oot1', 'accuracy_oot1', 'ks_oot1',
                                   'roc_oot2', 'accuracy_oot2', 'ks_oot2']
                      )
    current_dir = os.getcwd()
    os.chdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl)
    for file in glob.glob('*metrics.z'):
        l = joblib.load(file)
        df.loc[str(file.split('_')[0])] = l
        
    for file in glob.glob('*metrics.z'):
        os.remove(file)
        
    os.chdir(current_dir)
    df.index = df.index.set_names(['model_type'])
    df = df.reset_index()
    model_selection_criteria = model_selection_criteria.lower()
    column_to_sort = model_selection_criteria + '_' + dataset_to_use.lower()
    checker_value = 0.03
    
    if model_selection_criteria == 'ks':
        checker_value = checker_value * 100
        
    df['counter'] = (np.abs(df[column_to_sort] - df[model_selection_criteria + '_train']) > checker_value).astype(int)
                    + (np.abs(df[column_to_sort] - df[model_selection_criteria + '_valid']) > checker_value).astype(int)
                    + (np.abs(df[column_to_sort] - df[model_selection_criteria + '_test']) > checker_value).astype(int) 
                    + (np.abs(df[column_to_sort] - df[model_selection_criteria + '_oot1']) > checker_value).astype(int) 
                    + (np.abs(df[column_to_sort] - df[model_selection_criteria + '_oot2']) > checker_value).astype(int)
    
    
    df = df.sort_values(['counter', column_to_sort], ascending=[True, False]).reset_index(drop=True)
    df['selected_model'] = ''
    df.loc[0,'selected_model'] = 'Champion'
    df.loc[1,'selected_model'] = 'Challenger'
    df.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/metrics.xlsx')
    return df
    
            
            
            
            
            
            






# %%










