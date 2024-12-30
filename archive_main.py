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

def rmse(y_pred, y_actual):
    l_y_pred = y_pred
    l_y_actual = y_actual
    # l_y_pred = np.log(y_pred)
    # l_y_actual = np.log(y_actual)

    # replacing na values after log transform to 0
    # l_y_pred = np.nan_to_num(l_y_pred)
    # l_y_actual.fillna(0, inplace=True)

    error = mse(l_y_pred, l_y_actual)
    root_error = np.sqrt(error)

    return root_error



def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return round(np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))),4)

def get_categorical_correlation(df):
    correlation_matrix = pd.DataFrame(columns=df.columns, index=df.columns)
    cols = df.columns
    df = df.fillna('nan')
    for c1 in cols:
        for c2 in cols:
            confusion_matrix = pd.crosstab(df[c1], df[c2])
            corr = cramers_v(confusion_matrix)
            correlation_matrix.loc[c1, c2] = corr

    return correlation_matrix

def centering(feature):
    feature = feature.fillna(0) #filling na with 0s will break the lograthmic calculation of rmse
    feature = (feature)/feature.var()
    return feature

def eda_plots_numerical_features(features, train, save='numerical'):
    num_features = len(features)
    fig, axes = plt.subplots(num_features, 3, figsize=(12, 4 * num_features))

    for i, feature in enumerate(features):
        # Boxplot
        sns.boxplot(data=train[features], x=feature, ax=axes[i, 0])
        axes[i, 0].set_title(f'Boxplot of {feature}')

        # Histogram
        sns.histplot(data=train[features], x=feature, kde=False, bins=30, ax=axes[i, 1])
        axes[i, 1].set_title(f'Histogram of {feature}')

        # KDE Plot
        sns.kdeplot(data=train[features], x=feature, ax=axes[i, 2])
        axes[i, 2].set_title(f'KDE of {feature}')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f'eda_plots_{save}_features.png')
    plt.close()


def eda_plots_categorical_features(features, train):
    encoder = TargetEncoder(target_type='continuous', smooth=5)
    encoded = encoder.fit_transform(train[features], train['SalePrice'])
    encoded = pd.DataFrame(encoded, columns=features)
    encoded['SalePrice'] = train['SalePrice']
    encoded['PoolQCUtilitiesFreq'] = train.groupby('Utilities')['PoolQC'].transform('count')
    
    eda_plots_numerical_features(features, encoded, save='categorical_after_encoding-')
    return encoded

def eda_scatter_plot(data):
    target_variable = 'SalePrice'
    features = data.columns[data.columns != 'SalePrice']

    num_features = len(features)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()


    for i, feature in enumerate(features):
        sns.scatterplot(x=data[feature], y=data[target_variable], ax=axes[i], s=10, color='blue')
        axes[i].set_title(f'Scatter plot for {feature} vs {target_variable}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target_variable)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.savefig('eda_encoded_scatter_plot.png')
    plt.close()

# reading in and processing data to right format
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# make new features from old ones
# 1. month-year sold as a datetime feature = month-year
# 2. total # of bathrooms (half + full) = BsmtFullBath + FullBath + BsmtHalfBath + HalfBath
#. 3. total square feet of the house = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
train['MoSold_Sin'] = np.sin(2*np.pi * train['MoSold'] / 12)

#TODO: how to encode this datetime feature?

train['TotalBath'] = train['BsmtFullBath'] + train['FullBath'] + 0.5 * train['BsmtHalfBath'] + 0.5 *train['HalfBath']
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
train['TotalPorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']
train.drop(columns='Id', inplace=True)
#group transform as MasVnrArea and MasVnrType are highly correlated
# train['MasVnrAreaByType'] = train.groupby('MasVnrType')['MasVnrArea'].transform('mean')
train['FrontageToArea'] = train['LotFrontage'] / train['LotArea']

# interaction between PoolQC and Utilities (encoding categorical feature interactions)
train['PoolQCUtilities'] = train['PoolQC'].astype(str) + '_' + train['Utilities'].astype(str)

#log transform for normalization
train['SalePrice'] = np.log(train['SalePrice'])
train[['PoolArea', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'WoodDeckSF', 'GarageArea', 'GrLivArea',
       '2ndFlrSF', 'BsmtFinSF2', 'MasVnrArea', 'LotArea']] = \
np.log1p(train[['PoolArea', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'WoodDeckSF', 'GarageArea', 'GrLivArea',
       '2ndFlrSF', 'BsmtFinSF2', 'MasVnrArea', 'LotArea']])

#data duplication check
num_duplicates = train[train.duplicated()]
if len(num_duplicates) > 0:
    train.drop_duplicates()
    logging.info("train data has duplicates. these will be dropped.")
else:
    logging.info("train data has no duplicates")


train_X = train[train.columns[train.columns != 'SalePrice']]
y_ = train['SalePrice']


#splitting categorical and numeric features
categorical_features = train_X.columns[~train_X.columns.isin(train_X._get_numeric_data().columns)].to_list()
numerical_features = train_X.columns[train_X.columns.isin(train_X._get_numeric_data().columns)].to_list()

# plotting correlation for categorical features
categorical_corr_ = get_categorical_correlation(train_X[categorical_features])
categorical_corr_.to_csv('correlation_categorical.csv')


# plotting correlation for numerical features

corr_ = train[numerical_features + ['SalePrice']].corr()
corr_.to_csv(f'correlation_numerical.csv')

eda_plots_numerical_features(numerical_features + ['SalePrice'], train)
encoded = eda_plots_categorical_features(categorical_features, train)

#creating a new data set replacing categorical variables with their encoded version
train_encoded = train
train_encoded[categorical_features + ['PoolQCUtilities']] = encoded[categorical_features + ['PoolQCUtilities']]

eda_scatter_plot(train_encoded)

# plotting correlation for encoded categorical features
categorical_corr_ = encoded.corr()

# correlation_filter = pd.DataFrame()
# categorical_dict = {}
#
#
# for i, row in categorical_corr_.iterrows():
#     groups = train.groupby(i).groups
#
#     subsets = []
#     for key in groups.keys():
#         subsets.append(train['SalePrice'].iloc[groups[key]])
#     pval = ss.f_oneway(*subsets).pvalue
#     # categorical_corr_.loc[i, 'pval'] = pval
#     if pval < 0.05: #at 5% level of significance
#         categorical_corr_.loc[i, 'corr_with_target'] = 1 #difference in subsets is significant hence correlated with target
#     else:
#         categorical_corr_.loc[i, 'corr_with_target'] = 0 #difference in subsets is insignificant hence no correlation with target
#
# categorical_corr_ = categorical_corr_[categorical_corr_.corr_with_target != 0]
# categorical_corr_ = categorical_corr_[[c for c in categorical_corr_.index]]
# for c in categorical_corr_.columns:
#     high = categorical_corr_[categorical_corr_[c] > 0.5]
#     categorical_dict[c] = list(high.index)
# cat_df = pd.DataFrame(list(categorical_dict.items()), columns=['Key', 'Value'])
#
# for i, row in corr_.iterrows():
#     f = train_X[i].fillna(0)
#     c, pval = ss.pearsonr(f, train['SalePrice'])
#     if pval < 0.05:
#         corr_.loc[i, 'corr_with_target'] = c
#     else:
#         corr_.loc[i, 'corr_with_target'] = 0
# numerical_dict = {}
# # filtering only features that have high correlation with the target
#
# corr_ = corr_[corr_.corr_with_target != 0]
# corr_ = corr_[[c for c in corr_.index]]
# for c in corr_.columns:
#     high = corr_[corr_[c] > 0.5]
#     numerical_dict[c] = list(high.index)
# num_df = pd.DataFrame(list(numerical_dict.items()), columns=['Key', 'Value'])
#
# correlation_filter = pd.concat([cat_df, num_df])
# correlation_filter.Value = correlation_filter.Value.str.join(',')
# correlation_filter.to_csv('correlation_filter.csv')
#
# #filtering those features that are correlated from the training set
# correlation_filter = correlation_filter.loc[correlation_filter.Value.str.contains(',')]
# for i, row in correlation_filter.iterrows():
#     cols_to_drop = row.Value.split(',')[1:]
#     cols_to_drop = [c for c in cols_to_drop if c in train_X.columns]
#     if len(cols_to_drop) == 1:
#         cols_to_drop = cols_to_drop[0]
#     #     correlation_filter = correlation_filter[correlation_filter.Key != cols_to_drop]
#     # else:
#     #     correlation_filter = correlation_filter[~correlation_filter.Key.isin(cols_to_drop)]
#     train_X.drop(columns=cols_to_drop, inplace=True)


X_train, X_cv, y_train, y_cv = train_test_split(
    train_X, y_, test_size=0.33, random_state=42)

with PdfPages('correlations.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(50, 45))
    sns.set(font_scale=2.5)
    ax = sns.heatmap(categorical_corr_, annot=True, xticklabels=True, yticklabels=True, annot_kws={"size": 20}, cmap="crest")
    ax.tick_params(axis='both', labelsize=12)
    plt.title('correlation_categorical')
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(45, 35))
    sns.set(font_scale=1.8)
    ax = sns.heatmap(corr_, annot=True, xticklabels=True, yticklabels=True, cmap="crest")
    plt.title('correlation_numerical')
    pdf.savefig()
    plt.close()

    fig, ax = plt.subplots(figsize=(80, 65))
    sns.set(font_scale=2.5)
    ax = sns.heatmap(train_encoded.corr(), annot=True, xticklabels=True, yticklabels=True, annot_kws={"size": 12}, cmap="crest")
    ax.tick_params(axis='both', labelsize=12)
    plt.title('correlation_all')
    pdf.savefig()
    plt.close()

columns = train_X.columns
# with PdfPages('feature_saleprice_relationship_eda.pdf') as pdf:
#     for col in train_X.columns[1:]:
#         x = train_X[col]
#         if x.dtype == 'int' or x.dtype == 'float':
#             x = x.fillna(0)
#         else:
#             x = x.fillna("")
#         plt.figure(figsize=(8, 6))
#         plt.bar(x, y_)
#         plt.xticks(rotation=90)
#         plt.title(f'{col} vs SalePrice')
#         pdf.savefig()
#         plt.close()
#
# with PdfPages('feature_dist_eda.pdf') as pdf:
#     for col in train_X.columns[1:]:
#         x = train_X[col]
#         if x.dtype == 'int' or x.dtype == 'float':
#             x = x.fillna(0)
#         else:
#             x = x.fillna("")
#         plt.hist(x)
#         plt.xticks(rotation=90)
#         plt.title(f'Distribution of {col}')
#         pdf.savefig()
#         plt.close()



X_train[[c for c in categorical_features if c in X_train.columns]] = \
    X_train[[c for c in categorical_features if c in X_train.columns]].astype("category")
X_cv[[c for c in categorical_features if c in X_train.columns]] = \
    X_cv[[c for c in categorical_features if c in X_train.columns]].astype("category")

# xgboost for feature engineering
gbm = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8, enable_categorical=True)
gbm = gbm.fit(X_train, y_train)
y_train_pred_gbm = gbm.predict(X_train)
y_cv_pred_gbm = gbm.predict(X_cv)
print(f' gbm score on training set: {gbm.score(X_train, y_train)}')
print(f' gbm rmse on training set: {rmse(y_train_pred_gbm, y_train)}')
print(f' gbm score on CV set: {gbm.score(X_cv, y_cv)}')
print(f' gbm rmse on CV set: {rmse(y_cv_pred_gbm, y_cv)}')
gbm_feature_importances = pd.Series(gbm.feature_importances_, index=X_train.columns)
gbm_top_ten_features = gbm_feature_importances.nlargest(10).keys()
gbm_feature_importances.nlargest(30).plot(kind='barh', figsize=(35, 20))
plt.savefig('top_30_features.png')


#xgboost on encoded data
train_X = train_encoded[train_encoded.columns[train_encoded.columns != 'SalePrice']]
y_ = train_encoded['SalePrice']

X_train, X_cv, y_train, y_cv = train_test_split(
    train_X, y_, test_size=0.33, random_state=42)


gbm = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
gbm = gbm.fit(X_train, y_train)
y_train_pred_gbm = gbm.predict(X_train)
y_cv_pred_gbm = gbm.predict(X_cv)
print(f' gbm score on training set: {gbm.score(X_train, y_train)}')
print(f' gbm rmse on training set: {rmse(y_train_pred_gbm, y_train)}')
print(f' gbm score on CV set: {gbm.score(X_cv, y_cv)}')
print(f' gbm rmse on CV set: {rmse(y_cv_pred_gbm, y_cv)}')
gbm_feature_importances = pd.Series(gbm.feature_importances_, index=X_train.columns)
gbm_top_ten_features = gbm_feature_importances.nlargest(10).keys()
gbm_feature_importances.nlargest(30).plot(kind='barh', figsize=(35, 20))
plt.savefig('top_30_features_encoded.png')

#xgboost on encoded data after dropping correlated features
train_X = train_encoded[train_encoded.columns[train_encoded.columns != 'SalePrice']]
y_ = train_encoded['SalePrice']

#drop features that have corr > 0.7
train_X.drop(columns=['SaleType', 'Exterior2nd', 'KitchenQual', 'GarageCond',                        #categorical
                      'GarageYrBlt', 'TotRmsAbvGrd', 'GarageArea',
                      'BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'], inplace=True)         #numerical



X_train, X_cv, y_train, y_cv = train_test_split(
    train_X, y_, test_size=0.33, random_state=42)


gbm = xgb.XGBRegressor(n_estimators=500, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
gbm = gbm.fit(X_train, y_train)
y_train_pred_gbm = gbm.predict(X_train)
y_cv_pred_gbm = gbm.predict(X_cv)
print(f' gbm score on training set: {gbm.score(X_train, y_train)}')
print(f' gbm rmse on training set: {rmse(y_train_pred_gbm, y_train)}')
print(f' gbm score on CV set: {gbm.score(X_cv, y_cv)}')
print(f' gbm rmse on CV set: {rmse(y_cv_pred_gbm, y_cv)}')
gbm_feature_importances = pd.Series(gbm.feature_importances_, index=X_train.columns)
gbm_top_ten_features = gbm_feature_importances.nlargest(10).keys()
gbm_feature_importances.nlargest(30).plot(kind='barh', figsize=(35, 20))
plt.savefig('top_30_features_encoded_cleaned.png')