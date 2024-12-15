#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

def evaluate_direction_accuracy(ts_data, predictions, train_index):
    """
    Evaluate the direction prediction accuracy of a model.
    
    Parameters:
        ts_data (array-like): The original time series data (e.g., returns or prices).
        predictions (array-like): Predicted values (e.g., predicted returns).
        train_index (int): The index where the training data ends and testing begins.
    
    Returns:
        dict: Dictionary containing the dataframe of actual/predicted values and the accuracy score.
    """
    # Extract the test set (realized values)
    test_length = len(ts_data) - train_index
    realized_values = ts_data[train_index:]
    
    # Ensure both inputs are numeric arrays and of the same length
    if len(predictions) != len(realized_values):
        raise ValueError("Predictions and actual values must have the same length.")
    
    # Create a DataFrame with realized, predicted, and correctness of direction
    df = pd.DataFrame({
        "Realized": realized_values,
        "Predicted": predictions,
        "Correct": np.where(realized_values * predictions >= 0, 1, 0)  # 1 if both have the same sign
    })
    
    # Calculate the proportion of correct predictions
    accuracy = df["Correct"].mean()
    print(f"Proportion of correct direction predictions: {accuracy:.2f}")
    
    # Return the results
    return {"df": df, "accuracy": accuracy}

# Example Usage:
# Assuming `ts_ret` is your time series data (returns)
# Assuming `prediction_result["predictions"]` contains predicted returns
# and `prediction_result["train_index"]` is the training/testing split index.

# predictions = prediction_result["predictions"]
# train_index = prediction_result["train_index"]

# Evaluate direction accuracy
# results = evaluate_direction_accuracy(ts_ret, predictions, train_index)

# Access the results
# print(results["df"])
# print(f"Accuracy: {results['accuracy']:.2f}")

