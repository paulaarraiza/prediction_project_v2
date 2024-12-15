#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.interpolate import interp1d


# These functions do the following:
# 
# **Remove outliers**
# 
#     1. Calculates first and last quartiles, and applies IQR method. 
#     2. Replaces outliers with NA values
#     3. Inputs using an interpolation method.
#     
# **Check differencing**
# 
#     1. 

# In[1]:


def remove_outliers(df, column_name="Return", plot_outliers=True):
    """
    Removes outliers from the specified column of a DataFrame.
    Outliers are defined using the IQR method and replaced with interpolated values.
    """
    print(f"Removing outliers from the column: {column_name}...")
    
    # Calculate IQR
    q1 = np.percentile(df[column_name], 25)
    q3 = np.percentile(df[column_name], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"Outlier thresholds - Lower: {lower_bound}, Upper: {upper_bound}")
    
    # Identify outliers
    outlier_indices = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)].index
    print(f"{len(outlier_indices)} outliers detected.")
    
    # Replace outliers with NaN
    df.loc[outlier_indices, column_name] = np.nan
    
    # Plot the data with outliers highlighted
    if plot_outliers:
        plt.figure(figsize=(10, 6))
        plt.plot(df[column_name], color="blue", marker="o", label="Cleaned Data")
        plt.scatter(outlier_indices, df.loc[outlier_indices, column_name], color="red", label="Outliers")
        plt.title("Outlier Detection and Removal")
        plt.xlabel("Index")
        plt.ylabel(column_name)
        plt.legend()
        plt.grid()
        plt.show()
    
    # Interpolate missing values
    df[column_name] = df[column_name].interpolate(method="linear")
    print("Outliers removed and replaced with interpolated values.")
    
    return df


# In[3]:


def check_differencing(df, column_name="Return"):
    """
    Checks if normal and seasonal differencing (monthly and yearly) is required for the time series.
    """
    ts_data = df[column_name].dropna()
    print("Checking if differencing is required...")
    
    # Perform Augmented Dickey-Fuller (ADF) test for normal differencing
    print("Performing Augmented Dickey-Fuller (ADF) test for normal differencing...")
    adf_result = adfuller(ts_data, autolag="AIC")
    print(f"ADF Test p-value: {adf_result[1]}")
    
    # Check stationarity
    if adf_result[1] < 0.05:
        print("The series is stationary. No normal differencing required.")
        normal_diff_order = 0
    else:
        print("The series is not stationary. Normal differencing is required.")
        normal_diff_order = 1  # Default to 1 difference
        print(f"Recommended normal differencing order: {normal_diff_order}")
    
    # Check for monthly seasonality (30-day period)
    print("\nChecking for monthly seasonality (30-day period)...")
    seasonal_diff_order_monthly = 0
    ts_diff_monthly = ts_data.diff(periods=30).dropna()
    adf_seasonal_monthly = adfuller(ts_diff_monthly, autolag="AIC")
    print(f"Monthly ADF Test p-value: {adf_seasonal_monthly[1]}")
    if adf_seasonal_monthly[1] < 0.05:
        print("The series does not require seasonal differencing for monthly patterns.")
    else:
        print("The series requires seasonal differencing for monthly patterns (order: 1).")
        seasonal_diff_order_monthly = 1
    
    # Check for yearly seasonality (365-day period)
    print("\nChecking for yearly seasonality (365-day period)...")
    seasonal_diff_order_yearly = 0
    ts_diff_yearly = ts_data.diff(periods=365).dropna()
    adf_seasonal_yearly = adfuller(ts_diff_yearly, autolag="AIC")
    print(f"Yearly ADF Test p-value: {adf_seasonal_yearly[1]}")
    if adf_seasonal_yearly[1] < 0.05:
        print("The series does not require seasonal differencing for yearly patterns.")
    else:
        print("The series requires seasonal differencing for yearly patterns (order: 1).")
        seasonal_diff_order_yearly = 1

    # Return results
    return {
        "normal_diff_order": normal_diff_order,
        "seasonal_diff_order_monthly": seasonal_diff_order_monthly,
        "seasonal_diff_order_yearly": seasonal_diff_order_yearly
    }

