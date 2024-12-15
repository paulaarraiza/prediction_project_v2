#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime


# This function does the following:
# 
#     1. Select Date and Close columns from the df.  
#     2. Generate Return column. 
#     3. Convert to time series format. 
#     4. Create regular format (i.e., insert all dates because weekends or holidays have no records)
#     5. Generate output dataframe

# In[1]:


def generate_ts(df):
    # Step 1: Select the necessary columns and format the date
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    
    # Step 2: Generate df with returns instead of raw prices
    df["Return"] = df["Close"].pct_change(fill_method="pad") * 100  # Convert to percentage returns
    
    # Step 3: Delete the first row (no return for the first observation)
    df_ret = df.dropna(subset=["Return"]).reset_index(drop=True)
    
    # Step 5: Create time series objects
    ts_data = pd.Series(df_ret["Return"].values, index=df_ret["Date"])
    
    # Step 6: Create a regular date sequence for interpolation
    all_dates_ret = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq="D")
    
    # Step 7: Merge the time series with the regular date sequence and interpolate missing values
    ts = ts_data.reindex(all_dates_ret)
    ts = ts.interpolate(method="linear")  # Interpolate missing values
    
    # Step 8: Create a full dataframe for the time series (optional)
    df_full_ret = pd.DataFrame({"Date": ts.index, "Return": ts.values})
    
    return {"ts": ts, "df_full_ret": df_full_ret}


# In[2]:


def generate_log_ts(df):
    # Step 1: Select the necessary columns and format the date
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Step 2: Compute the natural log of the Close prices
    df["Log_Close"] = np.log(df["Close"])
    
    # Step 3: Compute the first difference of the log prices
    df["Return"] = df["Log_Close"].diff()
    
    # Step 3: Delete the first row (no return for the first observation)
    df_ret = df.dropna(subset=["Return"]).reset_index(drop=True)
    
    # Step 5: Create time series objects
    ts_data = pd.Series(df_ret["Return"].values, index=df_ret["Date"])
    
    # Step 6: Create a regular date sequence for interpolation
    all_dates_ret = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq="D")
    
    # Step 7: Merge the time series with the regular date sequence and interpolate missing values
    ts = ts_data.reindex(all_dates_ret)
    ts = ts.interpolate(method="linear")  # Interpolate missing values
    
    # Step 8: Create a full dataframe for the time series (optional)
    df_full_ret = pd.DataFrame({"Date": ts.index, "Return": ts.values})
    
    return {"ts": ts, "df_full_ret": df_full_ret}


# In[ ]:




