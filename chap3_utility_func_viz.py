
#%%

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SparkSession
#%%

spark = SparkSession.builder.appName('chap3').getOrCreate()

#spark.read.csv('movie_data_part1.csv').printSchema()


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

#%%


df.printSchema()


# %%  #### string functions ###

df_with_newcols = (df.select('id', 'budget', 'popularity')
                    .withColumn('budget_cat', when(df['budget']<10_000_000, 'Small')
                                .when(df['budget']<100_000_000, 'Medium')
                                .otherwise('Big')
                                )
                    .withColumn('ratings', when(df['popularity']<3, 'Low')
                                .when(df['popularity']<5, 'Mid').otherwise('High'))
                        
                    )

#%% conatenate 2 variables
df_with_newcols =(df_with_newcols.withColumn('BudgetRating_Category', 
                            concat(df_with_newcols.budget_cat,
                                   df_with_newcols.ratings
                                   )
                            )
)
# change variable to lower case
df_with_newcols = df_with_newcols.withColumn('BudgetRatiing_Category', trim(lower(df_with_newcols.BudgetRating_Category)))#.show(20, False)

df_with_newcols.show()

# %% Registering temporary table

df_with_newcols.registerTempTable('temp_data')

#%% applying function to show the results

spark.sql('''select ratings, count(ratings) as ratings_count 
          from temp_data 
          group by ratings''').show(10, False)


# %% ##########   Windows fucntions  #########
# cal deciles

from pyspark.sql.window import *

#%% ## filtering missing values
df_nonan_pop = (df_with_newcols.filter((df_with_newcols['popularity'].isNotNull())
                        & (~isnan(df_with_newcols['popularity']))
                        )
    
                )

## apply window function
df_decile_rank = (df_nonan_pop.select("id", "budget", "popularity",
 ntile(10)
 .over(Window.partitionBy()
       .orderBy(df_nonan_pop['popularity'].desc())
       )
 .alias("decile_rank")
)
)

# display values

(df_decile_rank.groupby("decile_rank")
 .agg(min('popularity')
      .alias('min_popularity'), 
      max('popularity').
      alias('max_popularity'), 
      count('popularity')
      )
).show()

#%%  #####   second most popular movie in 1970  #####

#%% select require subset

df_second_best = df.select('id', 'popularity', 'release_date')


# create year col
df_second_best_with_year = df_second_best.withColumn('release_year', year('release_date')).drop('release_date')

# define partition function

year_window = (Window.partitionBy(df_second_best_with_year['release_year'])
                .orderBy(df_with_newcols['popularity'].desc())
               )

# apply partition function

df_second_best_rank_yr = df_second_best_with_year.select('id', 'popularity', 
                                'release_year', 
                                rank().over(year_window).alias("rank")
                                )

# find second best rating for the year 1970

df_second_best_rank_yr.filter((df_second_best_rank_yr['release_year']==1970) & 
                              (df_second_best_rank_yr['rank']==2)).show()


# %%
"""
the difference between the revenue of the 
highest-grossing film of the year and other 
films within that year?
"""

# select required columns subset

df_revenue = df.select('id', 'revenue', 'release_date')

# create year column of release date

df_revenue = df_revenue.withColumn('release_year', year('release_date')).drop('release_date')

# define partition function along with range

windowRev = (Window.partitionBy(df_revenue['release_year'])
                .orderBy(df_revenue['revenue'].desc())
                .rangeBetween(-sys.maxsize, sys.maxsize)
                
            )

# apply partition function for revenue difference
revenue_diff = (max(df_revenue['revenue']).over(windowRev)
                - df_revenue['revenue']
    
                )

# final data

df_revenue.select('id', 'revenue', 'release_year', 
                  revenue_diff.alias("revenue_diff")).show(10, False)


# %% collect list
##### find all years where the title The Lost World is repeated.

# create year col from release date

df = df.withColumn('release_year', year('release_date'))

# Apply collect list function to gather all occurrences of

(df.filter("title=='The Lost World'")
 .groupby('title')
 .agg(collect_list("release_year")).show(1, False)
)

#%%  #### sampling ####

df_sample = df.sample(withReplacement=False, fraction=0.4, seed=2023)

df_sample.count()


##### stratified sampling ####

df_strata = df.sampleBy(col="release_year", fractions={1959: 0.2, 1960: 0.4, 1961: 0.4},
            seed=2023)

df_strata.count()


# %%  ###### saving data #######

df.write.format('csv').option('delimiter', '|').save('output_df')


#%%

df.coalesce(1).write.format('csv').option('delimiter', '|').save('output_coalesce_df')

#%% save partitioned data

(df.write.partitionBy('release_year').format('csv')
 .option('delimiter', '|')
 .save('output_df_partitioned')
)

#%% ### save data as hive table  #######
df.write.saveAsTable('film_ratings')


#%% #### pyspark_df to pandas df #####

df_pandas = df.toPandas()

#%% ### pandas to pyspark

df_py = spark.createDataFrame(df_pandas)

#%% #######  JOINS ########


#%%  ##### dropping duplicates  ########

df.dropDuplicates(['title', 'release_year']).count()


#%% ##### data visualization ########
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = df.withColumn('popularity', df['popularity'].cast(FloatType()))
#
histogram_data = (df.select('popularity')
                  .rdd.flatMap(lambda x: x)
                  ).histogram(25)

# load computed histogram into pandas df

hist_df = pd.DataFrame(list(zip(*histogram_data)), columns=['bin', 'frequency'])

# plotting the data

sns.set(rc={"figure.figsize": (12, 8)})
sns.barplot(x=hist_df['bin'], y=hist_df['frequency'])
plt.xticks(rotation=45)
plt.show()

#%% ####  filtering data to get a better viz ####

df_fil = df.filter('popularity < 22')

hist_data = (df_fil.select('popularity')
             .rdd.flatMap(lambda x: x)
             ).histogram(25)

hist_df = (pd.DataFrame(list(zip(*hist_data)), 
                        columns=['bin', 'frequency']
                        )
        )

sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.barplot(x=hist_df['bin'], y=hist_df['frequency'])
plt.xticks(rotation=25)
plt.title('Distribution of Popularity - Data is filtered')
plt.show()

#%% ## how many films were released between 1960 and 1970 by the year ##

# prepare df and convert to pandas

df_cat = (df.filter("(release_year > 1959) and (release_year < 1971)")
            .groupBy('release_year')
            .count()
            .toPandas()
          )

## sorting the values of display

df_sort = df_cat.sort_values(by=['release_year'], ascending=False)

# plotting the data

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.barplot(x=df_cat['release_year'], y=df_cat['count'])
plt.xticks(rotation=25)
plt.title('Number of films released each year from 1960 to 1970 in our dataset')
plt.show()

#%%
# %%
