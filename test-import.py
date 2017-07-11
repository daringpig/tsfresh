#!/usr/bin/env python

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import logging
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

logging.basicConfig(level=logging.ERROR)

df = load_har_dataset('body_acc_x_train.txt')
#print timeseries
#extracted_features = extract_features(timeseries, column_id=None, column_sort=None)
df.head()
print df.shape
print df

#plt.title('accelerometer reading')
#plt.plot(df.ix[0,:])
#plt.show()

extraction_settings = ComprehensiveFCParameters()
# Fill in Infs and NaNs
extraction_settings.IMPUTE = impute 

# transpose since tsfresh reads times series data column-wise, not row-wise
df_t = df.copy().transpose()
df_t.shape

# rearrange sensor readings column-wise, not row-wise

master_df = pd.DataFrame(df_t[0])
master_df['id'] = 0

# grab first 500 readings to save time
for i in range(1, 10299):
    temp_df = pd.DataFrame(df_t[i])
    temp_df['id'] = i
    master_df = pd.DataFrame(np.vstack([master_df, temp_df]))
print(master_df.shape)
master_df.head()

X = extract_features(master_df, column_id=1, default_fc_parameters=extraction_settings);
print X.shape
print X.columns
print X
print type(X)