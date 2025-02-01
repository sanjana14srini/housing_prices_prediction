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


def create_new_features(data):
    """
    data processing to incorporate new features based on eda
    """
    data['MoSold_Sin'] = np.sin(2 * np.pi * data['MoSold'] / 12)

    # TODO: how to encode this datetime feature?

    data['TotalBath'] =data['BsmtFullBath'] + data['FullBath'] + 0.5 * data['BsmtHalfBath'] + 0.5 * data[
        'HalfBath']
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']
    data.drop(columns='Id', inplace=True)
    
    # group transform as MasVnrArea and MasVnrType are highly correlated
    # train['MasVnrAreaByType'] = train.groupby('MasVnrType')['MasVnrArea'].transform('mean')
    data['FrontageToArea'] = data['LotFrontage'] / data['LotArea']

    # interaction between PoolQC and Utilities (encoding categorical feature interactions)
    # data['PoolQCUtilities'] = data['PoolQC'].astype(str) + '_' + data['Utilities'].astype(str)

    # log transform for normalization
    # data['SalePrice'] = np.log(data['SalePrice'])
    # data[['PoolArea', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'WoodDeckSF', 'GarageArea', 'GrLivArea',
    #        '2ndFlrSF', 'BsmtFinSF2', 'MasVnrArea', 'LotArea']] = \
    #     np.log1p(data[['PoolArea', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'WoodDeckSF', 'GarageArea',
    #                     'GrLivArea',
    #                     '2ndFlrSF', 'BsmtFinSF2', 'MasVnrArea', 'LotArea']])
    
    return data

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

def eda_raw_data_plots(train):
    num_features = len(train.columns)
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 4 * num_features))

    for i, feature in enumerate(train.columns):
        # Count Plot
        sns.countplot(data=train, x=feature, ax=axes[i, 0])

        #Box Plot
        sns.boxplot(data=train, x=feature, y='SalePrice', ax=axes[i, 1])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f'eda_raw_plots_features.png')
    plt.close()

def eda_plots_categorical_features(features, train, save_eda=True):
    encoder = TargetEncoder(target_type='continuous', smooth=5)
    encoded = encoder.fit_transform(train[features], train['SalePrice'])
    encoded = pd.DataFrame(encoded, columns=features)
    encoded['PoolQCUtilitiesFreq'] = train.groupby('Utilities')['PoolQC'].transform('count')
    
    if save_eda:
        encoded['SalePrice'] = train['SalePrice']
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

train = create_new_features(train)

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

categorical_features = categorical_features + ['MSSubClass', 'OverallQual', 'OverallCond']
numerical_features = [x for x in numerical_features if x not in ['MSSubClass', 'OverallQual', 'OverallCond']]
# plotting correlation for categorical features
# categorical_corr_ = get_categorical_correlation(train_X[categorical_features])
# categorical_corr_.to_csv('correlation_categorical.csv')

eda_raw_data_plots(train[categorical_features + ['SalePrice']])

# plotting correlation for numerical features

corr_ = train[numerical_features + ['SalePrice']].corr()
corr_.to_csv(f'correlation_numerical.csv')

eda_plots_numerical_features(numerical_features + ['SalePrice'], train)
encoded = eda_plots_categorical_features(categorical_features, train)

#creating a new data set replacing categorical variables with their encoded version
train_encoded = train
train_encoded[categorical_features + ['PoolQCUtilitiesFreq']] = encoded[categorical_features + ['PoolQCUtilitiesFreq']]

eda_scatter_plot(train_encoded)

# plotting correlation for encoded categorical features
categorical_corr_ = encoded.corr()

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

#FIXME: deep dive into each of the raw features and how they correlate with each other
#FIXME: why log transform and not other transformation? do you want to change the distribution of the data itself?
#FIXME: try other forms of data encoding? target encoding vs. other kinds
