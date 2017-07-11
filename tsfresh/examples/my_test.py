#!/usr/bin/env python

#from tsfresh.feature_extraction.extraction import extract_features
from har_dataset import download_har_dataset, load_har_dataset

timeseries = load_har_dataset();
#print timeseries
#extracted_features = extract_features(timeseries)