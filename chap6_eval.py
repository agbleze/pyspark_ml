
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









# %%
