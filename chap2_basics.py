
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
