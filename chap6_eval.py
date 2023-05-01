
#%% 
from pyspark.sql import SparkSession



#%%

filename = "bank-full.csv"

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(filename, 
                    header=True, 
                    inferSchema=True, 
                      sep=';'
                    )

df.show()

#%%  ####  Stratified Sampling Method ####
## option 1
train, test = df.randomSplit(weights=[0.7, 0.3], seed=12345)


#%% #### option 2 ####

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit



lr = LogisticRegression(maxIter=10, 
                        featuresCol='features',
                        labelCol='label'
                        )

# model parameters to try
paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build()
             )

# 70% for training nd 30% for validation
train_valid_clf = TrainValidationSplit(estimator=lr,
                                       estimatorParamMaps=paramGrid,
                                       evaluator=BinaryClassificationEvaluator(),
                                       trainRatio=0.7
                                       )

# assembled_df is the output of the vector assembler

model = train_valid_clf.fit(assembled_df)



#%% ### option 3 #####

# stratified sampling workaround using randomSplit

# split data for 0s and 1s

zero_df = df.filter(df['label']==0)
one_df = df.filter(df['label']==1)

# split data into train and test
train_zero, test_zero = zero_df.randomSplit([0.7, 0.3], seed=12345)
train_one, test_one = one_df.randomSplit([0.7,0.3], seed=12345)

# union datasets
train = train_zero.union(train_one)
test = test_zero.union(test_one)


#%% ### hold-out dataset

train, test, holdout = df.randomSplit([0.7,0.2,0.1], seed=12345)


#%% ##### k-fold cross-validation #####

from pyspark.ml.tuning import CrossValidator

# number of folds = 3
crossval_clf = CrossValidator(estimator=lr, 
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=3
                              )

# assembled_df is the output of the vector assembler

model = crossval_clf.fit(assembled_df)




#%% #### Leave-One-Group Out Cross Validation  #####

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import countDistinct
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
import numpy as np

#%%
filename = "bank-full.csv"

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df = df.withColumn('label', F.when(F.col('y') == 'yes', 1)
                    .otherwise(0)
                   )
df = df.drop('y')
df = df.select(['education', 'age', 'balance', 'day',
                'duration', 'campaign', 'pdays', 
                'previous', 'label'
                ]
               )
features_list = ['age', 'balance', 'day', 'duration', 'campaign',
                 'pdays', 'previous'
                 ]

# %%
def AssembleVectors(df, features_list, target_variable_name,
                    group_variable_name):
    assembler = VectorAssembler(inputCols=features_list,
                                outputCol='features'
                                )
    stages = [assembler]
    
    selectedCols = [group_variable_name, 
                    target_variable_name, 
                    'features'] #+ features_list
    
    # use pipeline to process sequentially
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    
    # apply assembler model on data
    df = assembleModel.transform(df).select(selectedCols)
    return df


# apply func
joined_df = AssembleVectors(df, features_list, 'label',
                            'education'
                            )


# find groups to apply cross validation

groups = list(joined_df.select('education').toPandas()['education']
              .unique())

# leave one-group out cross validation

def leave_one_group_out_validation(df, var_name, groups):
    train_metric_score = []
    test_metric_score = []
    
    for i in groups:
        train = df.filter(df[var_name]!=i)
        test = df.filter(df[var_name]==i)
        
        # model initialization
        lr = LogisticRegression(maxIter=10, featuresCol='features',
                                labelCol='label'
                                )
        evaluator = BinaryClassificationEvaluator(labelCol='label',
                                                  rawPredictionCol='rawPrediction',
                                                  metricName='areaUnderROC'
                                                  )
        
        # FIT MODEL
        lrModel = lr.fit(train)
        
        # make predictions
        predict_train = lrModel.transform(train)
        predict_test = lrModel.transform(test)
        train_metric_score.append(evaluator.evaluate(predict_train))
        test_metric_score.append(evaluator.evaluate(predict_test))
        print(str(i) + ' Group evaluation')
        print(' Train AUC - ', train_metric_score[-1])
        print(' Test AUC - ', test_metric_score[-1])
        
    print('Final evaluation for model')
    print('Train ROC', np.mean(train_metric_score))
    print('Test ROC', np.mean(test_metric_score))


#%%

leave_one_group_out_validation(joined_df, 'education', groups)



#%% ######## Evaluation Metric #####

## classificatio 
# Precision-Recall Curve
# what happens to PR Curce when the classifier is Bad

from sklearn.metrics import precision_recall_curve, average_precision_score
import random
import matplotlib.pyplot as plt
# random target and probabilities
rand_y = [random.choice([1, 0]) for i in range(0, 100)]
rand_prob = [random.uniform(0, 1) for i in range(0, 100)]

rand_precision, rand_recall, _ = precision_recall_curve(rand_y, probas_pred=rand_prob)
pr = average_precision_score(y_true=rand_y, y_score=rand_prob)

# plot random predicitons

plt.figure()
plt.title('PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(rand_recall, rand_precision, label='Average Precision = %0.2f' % pr)
plt.plot([0, 1], [0.5, 0.5], color='red', linestyle='--')
plt.legend(loc = 'lower right')



#%% ## Kolmogorov Smirnov (KS) statistics and Deciles ####

#load dataset, cleanup and fit model
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import countDistinct
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
import numpy as np
filename = "bank-full.csv"

#%%
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(filename, header=True, inferSchema=True, sep=';')
df = df.withColumn('label', F.when(F.col('y')=='yes', 1)
                   .otherwise(0)
                   )
df = df.drop('y')

# assemble individual columns to one column - features

def AssembleVectors(df, features_list, target_variable_name):
    assembler = VectorAssembler(inputCols=features_list,
                                outputCol='features'
                                )
    stages = [assembler]
    selectedCols = [target_variable_name, 'features']
    pipeline = Pipeline(stages=stages)
    assembleModel = pipeline.fit(df)
    
    df = assembleModel.transform(df).select(selectedCols)
    return df

#%%
df = df.select(['education', 'age', 'balance', 'day',
                'duration', 'campaign', 'pdays', 'previous',
                'label'
                ]
               )

features_list = ['age', 'balance', 'day', 'duration', 
                 'campaign', 'pdays', 'previous'
                 ]

assembled_df = AssembleVectors(df, features_list, 'label')

clf = LogisticRegression(featuresCol='features', labelCol='label')
train, test = assembled_df.randomSplit([0.7, 0.3], seed=12345)
clf_model = clf.fit(train)


#%% Deciling and KS calculation begins here
from pyspark.sql import Window

def CreateDeciles(df, clf, score, prediction, target, buckets):
    # get predictions from the model
    pred = clf.transform(df)
    
    # probability of 1's prediction and target
    pred = (pred.select(F.col(score), F.col(prediction), F.col(target))
            .rdd.map(lambda row: (float(row[score[1]]), 
                                  float(row['prediction']),
                                  float(row[target])
                                  )
                     )
            )
    predDF = pred.toDF(schema=[score, prediction, target])
    
    # remove ties in scores work around
    window = Window.orderBy(F.desc(score))
    predDF = predDF.withColumn("row_number", F.row_number().over(window))
    predDF.cache()
    
    predDF = predDF.withColumn("row_number", predDF["row_number"].cast("double"))
    
    # partition into 10 buckets
    window2 = Window.orderBy("row_number")
    final_predDF = predDF.withColumn("deciles", F.ntile(buckets).over(window2))
    final_predDF = final_predDF.withColumn("deciles", final_predDF["deciles"].cast("int"))
    # create non target column
    final_predDF = final_predDF.withColumn("non_target", 1-final_predDF[target])
    final_predDF.cache()
    
    # ks calculation starts here
    temp_deciles = final_predDF.groupby("deciles").agg(F.sum(target).alias(target)).toPandas()
    non_target_cnt = final_predDF.groupby("deciles").agg(F.sum("non_target").alias("non_target")).toPandas()
    temp_deciles = temp_deciles.merge(non_target_cnt, on="deciles", how="inner")
    temp_deciles = temp_deciles.sort_values(by="deciles", ascending=True)
    temp_deciles["total"] = temp_deciles[target] + temp_deciles["non_target"]
    temp_deciles["target_%"] = (temp_deciles[target] / temp_deciles["total"]) * 100
    temp_deciles["cum_target"] = temp_deciles[target].cumsum()
    temp_deciles["cum_non_target"] = temp_deciles["non_target"].cumsum()
    temp_deciles["target_dist"] = (temp_deciles["cum_target"] / temp_deciles[target].sum()) * 100
    temp_deciles["non_target_dist"] = (temp_deciles["cum_non_target"] / temp_deciles["non_target"].sum()) * 100
    temp_deciles["spread"] = temp_deciles["target_dist"] - temp_deciles["non_target_dist"]
    decile_table = temp_deciles.round(2)
    
    decile_table = decile_table[["deciles", "total", "label", "non_target",
                                 "target_%", "cum_target", "cum_non_target",
                                 "target_dist", "non_target_dist", "spread"
                                 ]]
    print("KS Value - ", round(temp_deciles["spread"].max(), 2))
    return final_predDF, decile_table


 #%% create deciles on train and test set
 
pred_train, train_deciles = CreateDeciles(train, clf_model, 'probability',
                                           "prediction", "label", 10
                                           )
pred_test, test_deciles = CreateDeciles(test, clf_model, "probability",
                                        "prediction", "label", 10
                                        )


# pandas styling funcs
from collections import OrderedDict
import pandas as pd
import sys


def plot_pandas_style(styler):
    from IPython.core.display import HTML
    html = '\n'.join([line.lstrip() for line in styler.render().split('\n')]) 
    return HTML(html)   
    
    
def highlight_max(s, color='yellow'):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ['background-color: {}'.format(color) if v else '' for v in is_max]


def decile_labels(agg1, target, color='skyblue'):
    agg1 = agg1.round(2)
    agg1 = agg1.style.apply(highlight_max, color = 'yellow', 
                            subset=['spread']
                            ).set_precision(2)
    agg1.bar(subset=[target], color='{}'.format(color), vmin=0)
    agg1.bar(subset=['total'], color='{}'.format(color), vmin=0)
    agg1.bar(subset=['target_%'], color='{}'.format(color), vmin=0)
    return agg1



#%% train deciles and KS
plot_decile_train = decile_labels(train_deciles, 'label', color='skyblue')
plot_pandas_style(plot_decile_train)


#%% test deciles and KS
plot_decile_test = decile_labels(test_deciles, 'label', color='skyblue')
plot_pandas_style(plot_decile_test)



#%% ### Actual vs Predicted, Gains Chart, Lift Chart

def plots(agg1, target, type):
    plt.figure(1, figsize=(20, 5))
    
    plt.subplot(131)
    plt.plot(agg1['DECILE'], agg1['ACTUAL'], label='Actual')
    plt.plot(agg1['DECILE'], agg1['PRED'], label='Pred')
    plt.xticks(range(10,110,10))
    plt.ylabel(str(target) + " " + str(type) +" %", fontsize=15)
    
    plt.subplot(132)
    X = agg1['DECILE'].tolist()
    X.append(0)
    Y = agg1['DIST_TAR'].tolist()
    Y.append(0)
    plt.plot(sorted(X), sorted(Y))
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xticks(range(0,110,10))
    plt.yticks(range(0,110,10))
    plt.grid(True)
    plt.title('Gains Chart', fontsize=20)
    plt.xlabel("Population %", fontsize=15)
    plt.ylabel(str(target) + str("DISTRIBUTION") + " %", fontsize=15)
    plt.annotate(round(agg1[agg1['DECILE'] == 30].DIST_TAR.item(), 2), xy=[30,30],
                 xytext=(25, agg1[agg1['DECILE'] == 30].DIST_TAR.item() + 5),
                 fontsize = 13
                )
    plt.annotate(round(agg1[agg1['DECILE'] == 50].DIST_TAR.item(), 2),
                 xy=[50,50],
                 xytext=(45, agg1[agg1['DECILE'] == 50].DIST_TAR.item() + 5),
                 fontsize = 13
                 )
    
    plt.subplot(133)
    plt.plot(agg1['DECILE'], agg1['LIFT'])
    plt.xticks(range(10,110,10))
    plt.grid(True)
    plt.title('Lift Chart', fontsize=20)
    plt.xlabel("Population %", fontsize=15)
    plt.ylabel("Lift", fontsize=15)
    
    plt.tight_layout()
    
    
#%% aggregation for actual vs predicted, gains and lift

def gains(data, decile_df, decile_by, target, score):
    
    agg1 = pd.DataFrame({}, index=[])
    agg1 = data.groupby(decile_by).agg(F.avg(target).alias('ACTUAL')).toPandas()
    score_agg = data.groupby(decile_by).agg(F.avg(score).alias('PRED')).toPandas()
    agg1 = agg1.sort_values(by=decile_by, ascending=True)
    agg1 = agg1[[decile_by, 'ACTUAL',  'PRED', 'target_dist']]
    agg1 = agg1.rename(columns={'target_dist': 'DIST_TAR', 
                                'deciles': 'DECILE'}
                       )
    decile_by = 'DECILE'
    agg1[decile_by] = agg1[decile_by]*10
    agg1['LEFT'] = agg1['DIST_TAR'] / agg1[decile_by]
    agg1.columns = [x.upper() for x in agg1.columns]
    plots(agg1, target, 'Distribution')
    
    
#%% train metrics

gains(pred_train, train_deciles, 'deciles', 'label', 'probability')

#%% test metrics

gains(pred_test, test_deciles, 'deciles', 'label', 'probability')











# %%
