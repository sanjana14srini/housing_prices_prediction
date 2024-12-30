import numpy as np
import pandas as pd
# import sklearn.linear_model
# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
import scipy.stats as ss
import logging
from pdpbox import pdp

import xgboost as xgb
from sklearn.model_selection import KFold

from matplotlib.backends.backend_pdf import PdfPages
from eda import eda_plots_categorical_features, create_new_features

def rmse(y_pred, y_actual):
    l_y_pred = y_pred
    l_y_actual = y_actual
    # l_y_pred = np.log(y_pred)
    # l_y_actual = np.log(y_actual)


    error = mse(l_y_pred, l_y_actual)
    root_error = np.sqrt(error)

    return root_error

def model(X_train, X_cv, y_train, y_cv):
    X_train['SalePrice'] = y_train
    X_cv['SalePrice'] = y_cv

    X_train = create_new_features(X_train)
    X_train_encoded = X_train
    categorical_features = X_train_encoded.columns[~X_train_encoded.columns.isin(X_train_encoded._get_numeric_data().columns)].to_list()
    X_train_encoded[categorical_features] = \
        eda_plots_categorical_features(categorical_features, X_train, save_eda=False)[categorical_features]
    X_cv = create_new_features(X_cv)
    X_cv_encoded = X_cv
    X_cv_encoded[categorical_features] = \
        eda_plots_categorical_features(categorical_features, X_cv, save_eda=False)[categorical_features]

    #drop y
    y_train_encoded = X_train['SalePrice']
    y_cv_encoded = X_cv['SalePrice']
    X_train_encoded.drop(columns='SalePrice', inplace=True)
    X_cv_encoded.drop(columns='SalePrice', inplace=True)

    #drop other features
    X_train_encoded.drop(columns=['SaleType', 'Exterior2nd', 'KitchenQual', 'GarageCond',  # categorical
                          'GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea',
                          'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], inplace=True)  # numerical
    X_cv_encoded.drop(columns=['SaleType', 'Exterior2nd', 'KitchenQual', 'GarageCond',  # categorical
                          'GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea',
                          'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], inplace=True)  # numerical


    gbm = xgb.XGBRegressor(n_estimators=100, max_depth=5, eta=0.05, subsample=0.6, colsample_bytree=0.8)
    gbm = gbm.fit(X_train_encoded, y_train_encoded)
    y_train_encoded_pred_gbm = gbm.predict(X_train_encoded)
    y_cv_encoded_pred_gbm = gbm.predict(X_cv_encoded)
    print(f' gbm score on training set: {gbm.score(X_train_encoded, y_train_encoded)}')
    print(f' gbm rmse on training set: {rmse(y_train_encoded_pred_gbm, y_train_encoded)}')
    print(f' gbm score on CV set: {gbm.score(X_cv_encoded, y_cv_encoded)}')
    print(f' gbm rmse on CV set: {rmse(y_cv_encoded_pred_gbm, y_cv_encoded)}')
    gbm_feature_importances = pd.Series(gbm.feature_importances_, index=X_train_encoded.columns)
    gbm_top_ten_features = gbm_feature_importances.nlargest(10).keys()
    gbm_feature_importances.nlargest(30).plot(kind='barh', figsize=(35, 20))
    plt.savefig('top_30_features_encoded_cleaned.png')


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_X = train[train.columns[train.columns != 'SalePrice']]
y_ = train['SalePrice']

categorical_features = train_X.columns[~train_X.columns.isin(train_X._get_numeric_data().columns)].to_list()
numerical_features = train_X.columns[train_X.columns.isin(train_X._get_numeric_data().columns)].to_list()

X_train, X_cv, y_train, y_cv = train_test_split(
    train_X, y_, test_size=0.33, random_state=42)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_cv.reset_index(drop=True, inplace=True)
y_cv.reset_index(drop=True, inplace=True)

model(X_train, X_cv, y_train, y_cv)