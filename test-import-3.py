#!/usr/bin/env python

import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import logging
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.ERROR)

filelists = [
    #"body_acc_x_",
    #"body_acc_y_",
    "body_acc_z_",
    #"body_gyro_x_",
    #"body_gyro_y_",
    #"body_gyro_z_",
]

result = None

for filename in filelists:
    file = filename + "train.txt"
    df = load_har_dataset(file)
    #print timeseries
    #extracted_features = extract_features(timeseries, column_id=None, column_sort=None)
    df.head()
    #print df.shape
    #print df

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
    #print X.shape
    #print X.columns
    #print X
    #print type(X)

    result = pd.concat([result, X], axis=1)
    print "------------------------------------------"
    print result.shape 

y = load_har_classes()[:10299]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(classification_report(y_test, cl.predict(X_test)))
