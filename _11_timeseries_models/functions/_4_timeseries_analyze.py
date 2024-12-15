#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from arch import arch_model
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[ ]:


def evaluate_garch_model(ts_train, ts_test, p, q, r, s):
    """
    Evaluate a GARCH model using rolling predictions.
    
    Parameters:
        ts_data (array-like): The time series data.
        p (int): AR order for the mean model.
        q (int): MA order for the mean model.
        r (int): GARCH order.
        s (int): ARCH order.
        train_size (float): Proportion of the data to use for training (default: 0.95).
        
    Returns:
        dict: Dictionary containing rolling predictions, RMSE, and training index.
    """
    
    # 2. Define the GARCH model specification
    rolling_predictions = np.empty(len(ts_test))
    rolling_predictions[:] = np.nan  # Initialize predictions with NaNs
    
    # 3. Perform rolling predictions
    for i in tqdm(range(len(ts_test)), desc="Rolling Predictions"):
        # Incrementally increase the size of the training data with each new observation
        ts_train_window = np.append(ts_train, ts_test[:i])
        
        # Fit the GARCH model
        try:
            print("adios")
            model = arch_model(ts_train_window, vol="Garch", p=r, q=s, mean="ARX", lags=p, dist="normal")
            fit = model.fit(disp="off")
            
            # Forecast the next value
            forecast = fit.forecast(horizon=1)
            rolling_predictions[i] = forecast.mean.iloc[-1, 0]
            print("hola")
        except Exception as e:
            print(f"Error fitting GARCH model at step {i}: {e}")
            rolling_predictions[i] = np.nan
    
    # 4. Evaluate the model using RMSE
    actual_values = ts_test
    rmse = np.sqrt(np.nanmean((rolling_predictions - actual_values) ** 2))
    print(f"Rolling RMSE: {rmse:.4f}")
    
    # 5. Return the results
    return {"predictions": rolling_predictions, "rmse": rmse, "train_index": train_index}

