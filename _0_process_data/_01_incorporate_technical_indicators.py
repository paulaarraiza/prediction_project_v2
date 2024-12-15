import pandas as pd
import os
from ta import add_all_ta_features
from ta.utils import dropna
import yfinance as yf
from datetime import datetime


def generate_technical_ind_dataset(stock, start_date, end_date):
    ticker = stock
    
    # Download data using yfinance
    yf_data = yf.download(ticker, start=datetime.strptime(start_date, '%Y-%m-%d'), 
                          end=datetime.strptime(end_date, '%Y-%m-%d'))

    # Flatten multi-level column headers if necessary
    if isinstance(yf_data.columns, pd.MultiIndex):
        yf_data.columns = yf_data.columns.droplevel(1)

    # Reset the index to make 'Date' a regular column
    yf_data = yf_data.reset_index(drop = False)

    # Add technical indicators
    df_with_indicators = add_all_ta_features(
        yf_data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=False
    )

    # Interpolate missing values
    df_with_indicators = df_with_indicators.interpolate(method="linear")

    # Drop any remaining NaN values
    df_with_indicators = df_with_indicators.dropna()
    
    return df_with_indicators