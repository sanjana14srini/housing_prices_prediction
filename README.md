# housing_prices_prediction

This repository is based on the Kaggle Project which can be found at https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/discussion/542054

The idea behind this repository is to address the issue of dimensionality with the given data set.
Usually methods to resolve the curse of dimensionality are addressed in sparse datasets. However, 
the dataset for this project is dense with many features. 
The intent of this project is to reduce its dimensionality without losing valuable information that each
of the features can provide. 

Some of the things I tried are:
1. separating categorical features from numerical ones and understanding the relationship
between them using correlation heatmaps
2. created EDAs to deduce any linear relationships between features and the housing prices

Things that I want to try:
1. Feature Engineer: is  there an efficient way to combine one or more features such that
they all become meaningful together (instead of individually) without any loss of information?
2. Can this be achieved by solely studying and understanding various forms of EDAs or
do we want to apply any ML algorithms?
3. Test this using Boosting algorithms to check if these replacement features are
informative while being sufficiently correlated with original features

Since this is still and ongoing project, the code will break or tend to be faulty. 
I am working on this during my free time, so activity on this would be haphazard and random.


