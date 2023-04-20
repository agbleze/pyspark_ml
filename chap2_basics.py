
#%%
import pandas as pd
from pyspark.sql import SparkSession
from download_file_online import download_file


#%%

spark = SparkSession.builder.appName("Data Wrangling").getOrCreate()

#%%
data_link ="https://raw.githubusercontent.com/Apress/applied-data-science-using-pyspark/main/Ch02/Chapter2_Data/movie_data_part1.csv"

data_link2 = "https://github.com/Apress/applied-data-science-using-pyspark/blob/main/Ch02/Chapter2_Data/movie_data_part1.csv"

# %%

download_file(url=data_link2, filename_to_save="movie_data.csv")

#%%

file_location = "movie_data.csv"
file_type = "csv"
infer_schema = "False"
first_row_is_header = "F"
delimiter = "|"

#%%
df = (spark.read.format(file_type)
        #.option("inferSchema", infer_schema)
        .option("header", first_row_is_header)
       # .option("sep", delimiter)
        .load(file_location)
        )

#%%

df.show()



#%%
df2 = spark.read.csv(path="movie_data.csv")





# %%
# Read data
file_location = "movie_data_part1.csv"
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"
delimiter = "|"

df = spark.read.format(file_type)\
.option("inferSchema", infer_schema)\
.option("header", first_row_is_header)\
.option("sep", delimiter)\
.load(file_location)
# %%

df.printSchema()
# %%
df.count()

# %%

select_columns = ['id', 'budget', 'popularity', 'release_date', 'revenue', 'title']

df = df.select(*select_columns)

df.show()

#%%

df.select(df[2], df[1], df[6], df[9], df[10], df[14]).show()

#%%

from pyspark.sql.functions import (isnan, count,
                                   when,col, desc
                                   
                                   )

from pyspark.sql.types import IntegerType, FloatType, DateType

#%% count number of missing values

(df.filter((df['popularity']=='')|df['popularity'].isNull()|
           isnan(df['popularity'])).count()
 )


#%%
df.select([count(when((col(c)=='') | col(c).isNull() 
                      | isnan(c), c)).alias(c) 
           for c in df.columns]).show()


#%%

df.groupBy(df['title']).count().sort(desc('count')).show(10, False)

#%% subset and create temporary df without any missing values

df_temp=df.filter((df['title']!='')&(df['title'].isNotNull())&
          (~isnan(df['title']))
          )

#%% subset df to titles repeated more than 4

(df_temp.groupby(df_temp['title']).count()
 .filter("`count`>=4").sort(col("count").desc()).count()
 )

#%%

del df_temp

#%%
df.dtypes

#%% casting of datatypes

df = df.withColumn("budget", df['budget'].cast("float"))
df.dtypes


# %%

int_vars = ['id']
float_vars = ['budget', 'popularity', 'revenue']
date_vars = ['release_date']

#%%

for column in int_vars:
    df = df.withColumn(column, df[column].cast(IntegerType()))

for column in float_vars:
    df = df.withColumn(column, df[column].cast(FloatType()))

for column in date_vars:
    df = df.withColumn(column, df[column].cast(DateType()))

df.dtypes

#%%

df.describe().show()


# %% ###### cal median  #########
df_temp = df.filter((df['budget']!=0.0)&(df['budget'].isNotNull()) & 
                    (~isnan(df['budget']))
                    )

median = df_temp.approxQuantile(col='budget', probabilities=[0.5], 
                           relativeError=0.1
                           )
print('The median of budget is ' +str(median))


# %%
