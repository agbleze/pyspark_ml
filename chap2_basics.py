
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

from pyspark.sql.functions import *

from pyspark.sql.types import *

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

median = df_temp.approxQuantile(col=['budget','revenue'], probabilities=[0.5], 
                           relativeError=0.001
                           )
print('The median of budget is ' +str(median))


# %%
df.select('title').distinct().show(10, False)

#%%
df.agg(countDistinct(col("title")).alias("count")).show()

#%% extract year from the release date, cal distinct count by year

(df.withColumn('release_year', year('release_date'))
 .groupBy("release_year")
 .agg(countDistinct("title")).show(10, False)
 )

#%% extract month

df.withColumn('release_month', month('release_date'))\
    .select('release_month').show(10)

#%% extract day of month

(df.withColumn('release_day', dayofmonth('release_date'))
 .select('release_day').show(10)
    
)

#%% filter titles starting with Meet

df.filter(df['title'].like('Meet%')).show(10, False)

#%% filter titles that do not end with s
df.filter(~df['title'].like('%s')).show(10, False)

#%%  any title that contains ove

df.filter(df['title'].rlike('\w*ove')).show(10,False)

#%% filter title containing ove
df.filter(df.title.contains('ove')).show(10,False)

#%% identify columns with particular prefix or suffix
# identify columns starting with re
df.select(df.colRegex("`re\w*`")).printSchema()

#%% identify columns with suffix e
df.select(df.colRegex("`\w*e`")).printSchema()

#%% ####### calculate variance  ###########

mean_pop = df.agg({'popularity': 'mean'}).collect()[0]['avg(popularity)']
count_obs = df.count()

df = df.withColumn('mean_popularity', lit(mean_pop))

df = df.withColumn('variance', pow((df['popularity']-df['mean_popularity']), 2))

variance_sum = df.agg({'variance': 'sum'}).collect()[0]['sum(variance)']
variance_population = variance_sum / (count_obs-1)

#%%

def new_cols(budget, popularity):
    if budget < 10_000_000: budget_cat = 'Small'
    elif budget < 100_000_000: budget_cat = 'Medium'
    else: budget_cat = 'Big'
    if popularity<3: ratings='Low'
    elif popularity<5: ratings='Mid'
    else: ratings='High'
    return budget_cat, ratings
# %% Apply the user-defined function on the DataFrame

udfB = udf(new_cols, StructType([StructField("budget_cat", StringType(), True),
                                 StructField("ratings", StringType(), True)])
           )

temp_df=df.select('id', 'budget', 'popularity').withColumn("newcat", udfB("budget", "popularity"))

df_with_newcols = (temp_df.select('id', 'budget', 'popularity', 'newcat')
                    .withColumn('budget_cat', temp_df.newcat.getItem('budget_cat'))
                    .withColumn('ratings', temp_df.newcat.getItem('ratings'))
                    .drop('newcat')
                )
df_with_newcols.show(15, False)


#%%

df_with_newcols = (df.select('id', 'budget', 'popularity')
                   .withColumn('budget_cat', when(df['budget']<10_000_000, 'Small')
                               .when(df['budget']<100_000_000, 'Medium')
                               .otherwise('Big')
                               )
                   .withColumn('ratings', when(df['popularity']<3, 'Low')
                               .when(df['popularity']<5,'Mid')
                               .otherwise('High')
                               )
                   )

df_with_newcols.show(15, False)
# %% deleting and renaming columns
# delete column
columns_to_drop = ['budget_cat']
df_with_newcols = df_with_newcols.drop(*columns_to_drop)

df_with_newcols.printSchema()

#%% renaming columns
(df_with_newcols.withColumnRenamed('id', 'film_id')
    .withColumnRenamed(existing='ratings', new='film_ratings')
 ).printSchema()

#%% alternative way of renaming

new_names = [('budget', 'film_budget'), ('popularity', 'film_popularity')]

(df_with_newcols.select(list(map(lambda old, new:col(old).alias(new), *zip(*new_names))))
 ).printSchema()

# %%
