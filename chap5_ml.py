
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

# assemble feature vectors

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

linear_df =data.select(['age', 'balance', 'day', 'duration', 'campaign',
                        'pdays', 'previous']
                       )
target_variable_name = 'balance'

features_list = linear_df.columns










# %%
