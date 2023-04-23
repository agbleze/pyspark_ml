



#%% ### CUSTOM TRANSFORMERS  ####

## CORRELATION ANALYSIS

from pyspark.mllib.stat import Statistics
import pandas as pd

from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol

#%%

class CustomCorrelation(Transformer, HasInputCol):
    """function to calulate correlation b/T 2 variables
    
    Parameters:
    ----------
    inputCol: feature column name to be used for correlation
    """
    def __init__(self, inputCol=None, correlation_type='pearson',
                 correlation_cutoff=0.7
                 ):
        super(CustomCorrelation, self).__init__()
        self.inputCol = inputCol
        
        assert correlation_type == 'pearson' or correlation_type == 'spearman', "Provide valid correlation type to be pearson or spearman"
        self.correlation_type = correlation_type
        
        assert 0.0 <= correlation_cutoff <= 1.0, "Provide valide value from 0 to 1"
        self.correlation_cutoff = correlation_cutoff
        
    def _transform(self, df):
        for k, v in df.schema[self.inputCol].metadata["ml_attr"]["attrs"].items():
            features_df = pd.DataFrame(v)
            column_names = list(features_df["name"])
            df_vector = df.rdd.map(lambda x: x[self.inputCol].toArray())
            matrix = Statistics.corr(df_vector, method=self.correlation_type)
            
            corr_df = pd.DataFrame(matrix, columns=column_names, index=column_names)
            final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort')).reset_index()
            final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_value'}, axis=1, inplace=True)
            final_corr_df = final_corr_df[final_corr_df['col1'] != final_corr_df['col2']]

            shortlisted_corr_df = final_corr_df[final_corr_df['correlation_value'] > self.correlation_cutoff]
            return corr_df, shortlisted_corr_df


