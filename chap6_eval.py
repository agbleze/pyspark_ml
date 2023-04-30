
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













# %%
