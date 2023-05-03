
#%%
from pyspark.sql.types import *
from pyspark.sql import SparkSession



#%%

file_location = "cluster_data.csv"
file_type = "csv"
infer_schema = False
first_row_is_header = "true"



