#!/usr/bin/env python
# coding: utf-8

# In[44]:

import pandas as pd
import os
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# **Data Preprocessing in ML**
# 
#     1. Remove NA values
#     2. Remove outliers
#     3. Rescale column by column

# project_dir = "/home/jupyter-tfg2425paula/prediction_project_v2"
# os.chdir(project_dir)
# data_dir = os.path.join(project_dir, "raw_data")
# stocks_folder = os.path.join(data_dir, "single_names")
# stock = 'EQNR'
# filename = f'{stock}_Close.csv'
# 
# df = pd.read_csv(os.path.join(stocks_folder, filename), sep=";", decimal=",")

# project_dir = "/home/jupyter-tfg2425paula/prediction_project_v2"
# functions_dir = os.path.join(project_dir, "arima_garch/functions")
# sys.path.append(project_dir)
# python_files = [f for f in os.listdir(functions_dir) if f.endswith(".py")]
# 
# for file in python_files:
#     module_name = file.replace(".py", "")
#     print(f"Importing module: {module_name}")
#     globals()[module_name] = __import__(f"arima_garch.functions.{module_name}", fromlist=["*"])
# 

# **0. Deal with Return and date columns**

def appropiate_date_format(df, date_col_name, date_format="%d/%m/%y"):
    """ 
    """
    df[date_col_name] = pd.to_datetime(df[date_col_name], format=date_format)
    return df

def create_return_column(df, target_col_name, remove_close):
    """ 
    """
    df = df.copy()
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors="coerce")
    df["Return"] = df[target_col_name].pct_change(fill_method="pad") * 100 
    if remove_close:
        df = df.drop(columns = target_col_name)
    
    return df


# **1. Check for NA values**

# In[55]:


def remove_na(df, selected_cols):
    """
    Handles missing values in specified columns of a DataFrame using linear interpolation.
    Removes rows with missing values if they are at the beginning or end.
    """
    na_method = "linear"
    
    # Remove first and last because they usually cause problems 
    
    print("Missing values in each selected column before handling:")
    print(df[selected_cols].isna().sum())

    rows_with_na = df[df[selected_cols].isna().any(axis=1)]
    print("\nRows with missing values in the selected columns:")
    print(rows_with_na)

    # Interpolate missing values for the selected columns
    for col in selected_cols:
        df[col] = df[col].interpolate(na_method, limit_direction="both")
        
    df = df.iloc[1:-1].copy()

    print("\nMissing values in each selected column after handling:")
    print(df[selected_cols].isna().sum())

    return df


# **2. Remove outliers**

# In[56]:


def replace_outliers_iqr(df, output_col):
    """
    Replaces outliers in a DataFrame using the IQR method with interpolated or mean/median values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        selected_col (str): The column to process.
        threshold (float): The IQR threshold. Default is 1.5.
        method (str): The method to replace outliers ("interpolate", "mean", "median"). Default is "interpolate".
        
    Returns:
        pd.DataFrame: The DataFrame with outliers replaced.
    """
    method="linear"
    threshold = 1.5
    
    col = output_col
    Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range

    # Define outlier boundaries
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Identify outliers
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    extreme_outliers = df.loc[outliers, col]
    num_outliers = len(extreme_outliers)
    min_outlier = extreme_outliers.min() if not extreme_outliers.empty else None

    print(f"Number of outliers eliminated: {num_outliers}")
    print(f"Minimum extreme outlier value: {min_outlier}")
    
    df.loc[outliers, col] = np.nan
    df[col] = df[col].interpolate(method, limit_direction="both")

    return df


# **3. Normalize data column by column**
# 
#     - StandardScaler: subtracts mean and divides by standard deviation. 
#     - MinMaxScaler: arranges values on a specified scale (0, 1) as default. 

# In[73]:

def generate_target_column(df, output_col):
    df['Target'] = (df[output_col].shift(-1) > 0).astype(float)
    
    return df

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def scale_data(df, selected_scale_cols, scaling_method):
    """
    Scales specified columns in a DataFrame using the specified scaling method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        selected_cols (list): A list of column names to scale.
        scaling_method (str): The scaling method to use ("standard" or "minmax"). Default is "standard".
    
    Returns:
        pd.DataFrame: The DataFrame with specified columns scaled.
    """
    
    if scaling_method is not None:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")

        # Scale only the selected columns
        df_scaled = df.copy()
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally fill NaN with column mean or median
        df.fillna(df.mean(), inplace=True)

        df_scaled[selected_scale_cols] = scaler.fit_transform(df[selected_scale_cols])
    
    else:
        df_scaled = df
        
    return df_scaled

# **Condense all steps**

# In[76]:

def preprocess_data(df, selected_na_cols, output_col, selected_scale_cols, scaling_method):

    processed_df = remove_na(df, selected_na_cols)
    processed_df = replace_outliers_iqr(processed_df, output_col)
    processed_df = generate_target_column(processed_df, output_col)
    processed_df = scale_data(processed_df, selected_scale_cols, scaling_method)
    
    return processed_df
