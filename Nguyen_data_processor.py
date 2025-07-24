"""
Some code snippet to use later

"""

# ====================================================================================================
# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler      # Normalization methods
import scipy.stats as stats

# ====================================================================================================
# 0. Removing columns with many missing values
def remove_missing_columns(data, threshold = 0.25):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    data = data.loc[:, data.isnull().mean() <= threshold]
    return data


# ====================================================================================================
# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """

    # Subset feature column
    features = data.copy()

    # Loop through the features column names that contains numeric values, and fill in missing values in the respective data df column using appropriate method.
    for col in features.select_dtypes(include = np.number).columns:
        match strategy:
            case 'mean':
                data[col].fillna(data[col].mean(), inplace = True)
            case 'median':
                data[col].fillna(data[col].median(), inplace = True)
            case 'mode':
                data[col].fillna(data[col].mode(), inplace = True)
            case _:
                print(f"Unknown strategy: {strategy}")
    
    return data

# ====================================================================================================
# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    data.drop_duplicates(inplace=True)
    return data

# ====================================================================================================
# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """
    Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # Get features column names, and only include numerical columns
    features_col = data.copy().select_dtypes(include=np.number).columns

    # Defining scaling method
    match method:
        case 'minmax':
            scaler = MinMaxScaler() 
        case 'standard':
            scaler = StandardScaler()
        case _:
            print(f"Unknown method: {method}")
    
    # Apply method to data, only on the selected columns
    data[features_col] = scaler.fit_transform(data[features_col])

    return data

# ====================================================================================================
# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """

    # Subset feature columns with numerical values
    features = data.select_dtypes(include=np.number).iloc[:,1:]

    # On numerical columns, calculate the correlation (Pearson method) using .corr()
    # .abs() get absolute value. 
    corr_matrix = features.corr().abs()

    # np.ones() Create a new matrix with the same dimension as the original correlation matrix and fill the values with 1s.
    # np.triu() Get upper triangle of the matrix, indicated by 1s and 0s.
    # .astype(bool) Convert 0 -> F and 1 -> T.
    # corr_matrix.where() To apply selection to original correlation matrix.
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify any columns within the upper correlation matrix that has a high correlation value with another column/
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    # Drop those chosen columns
    data = data.drop(columns=to_drop)

    return data


# ====================================================================================================
# 5. Impute outliers
def impute_outlier(data, method, threshold = 3):
    """Impute outliers.
    :param data: pandas DataFrame
    :param method: str, imputation method ('mean', 'median', 'mode')
    :param threshold: float, z-score threshold for determining outliers
    :return: pandas DataFrame
    """
    # Create an "outliers" df with the same dimensions with the original dataframe
    # Fill it with True (if deemed as outliers) or False (if not deemed as outliers)
    outliers = pd.DataFrame(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) > threshold)
    
    # Loop though each columns in select_dtypes, if 
    for col_name in outliers.select_dtypes(include = np.number).columns:
        col_content = outliers[col_name]
        match method:
            case 'mean':
                data[col_content , col_name] = data[col_name].mean()
            case 'median':
                data[col_content, col_name] = data[col_name].median()
            case 'mode':
                data[col_content, col_name] = data[col_name].mode()[0]
            case _:
                print(f"Unknown method: {method}")
    
    return data
