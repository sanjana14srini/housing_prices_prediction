import numpy as np
import pandas as pd

def process_categorical(data, to_drop):
    """
    This function processes categorical features as follows:
    1. based on eda_raw_plots_features.png, we have decided to drop some features
    2. based on the same eda, we decide to combine the categories of some features
    """

    data_ = data.copy()

    # capturing all categorical features
    categorical_features = data.columns[~data.columns.isin(data._get_numeric_data().columns)].to_list()
    categorical_features = categorical_features + ['MSSubClass', 'OverallQual', 'OverallCond']

    # setting variables as categorical so that it doesn't break the model
    data_[categorical_features] = data_[categorical_features].astype('category')

    # dropping the chosen categorical features
    data_ = data_.drop(columns=to_drop)

    # combining categories
    categorical_recoding = pd.read_csv('categorical_recoding.csv')
    for c in categorical_recoding['column'].unique():
        cr = categorical_recoding.loc[categorical_recoding['column'] == c][['values', 'values_new']]
        cr.index = cr['values']
        cr.drop(columns='values', inplace=True)
        cr = cr.to_dict()
        data_[c] = data_[c].map(cr['values_new'])
        data_[c] = data_[c].astype('category')

    return data_


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

to_drop = ['Street', 'LandContour', 'Utilities', 'LandSlope', 'BldgType', 'RoofStyle', 'BsmtFinType2', 'Heating',
           'CentralAir', 'Functional', 'GarageQual', 'GarageCond', 'Fence', 'MiscFeature']


train_processed = process_categorical(train, to_drop)
