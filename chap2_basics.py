
#%%
import pandas as pd
from pyspark.sql import SparkSession


#%%

spark = SparkSession.builder.appName("Data Wrangling").getOrCreate()

#%%
data_link ="https://raw.githubusercontent.com/Apress/applied-data-science-using-pyspark/main/Ch02/Chapter2_Data/movie_data_part1.csv"

#%%
pd.read_csv(data_link, on_bad_lines="skip")


#%%

import requests


url = "https://raw.githubusercontent.com/Apress/applied-data-science-using-pyspark/main/Ch02/Chapter2_Data/movie_data_part1.csv"

r = requests.get(url, allow_redirects=True)

open('movie_data.csv', 'wb').write(r.content)




# %%
