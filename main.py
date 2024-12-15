#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import _0_process_data


# **Set directories**

# In[2]:


project_dir = "/home/jupyter-tfg2425paula/prediction_project_v2"
os.chdir(project_dir)

raw_data_dir = os.path.join(project_dir, "_00_data_raw")
transformed_data_dir = os.path.join(project_dir, "_01_data_transformed")
structured_data_dir = os.path.join(project_dir, "_02_data_structured")

from _0_process_data._00_preprocess_data import preprocess_data
from _0_process_data._00_preprocess_data import appropiate_date_format
from _0_process_data._00_preprocess_data import create_return_column

stock = 'AAPL'
data_type = "stock_single_name"
# standard, minmax or None
scaling_method = "standard"
processing_method = None


if data_type == "stock_single_name":

    securities = "single_names"
    stocks_folder = os.path.join(raw_data_dir, securities)
    
    filename = f'{stock}_Close.csv'

    df = pd.read_csv(os.path.join(stocks_folder, filename), sep=";", decimal=",")
    
if data_type == "stock_single_name":
    clean_stocks_folder = os.path.join(raw_data_dir, 'cleaned_'+ str(securities))

    date_col_name = "Date"
    target_col_name = "Close"
    return_col = "Return"

    # standard, minmax or None
    scaling_method = "standard"

    # Only if date_format is not appropiate
    df = appropiate_date_format(df, date_col_name, date_format="%d/%m/%y")
    df = create_return_column(df, target_col_name, remove_close=True)

    selected_na_cols = list(df.columns)
    selected_scale_cols = list(df.drop(columns=[date_col_name]).columns) # All but Date

    df_clean = preprocess_data(df, selected_na_cols, return_col, selected_scale_cols, scaling_method)
    df_clean.to_csv(os.path.join(clean_stocks_folder, filename), index=False)
    
    start_date = df_clean['Date'].iloc[0]
    end_date = df_clean['Date'].iloc[-1]
    
    df_clean.head()


if data_type == "stock_technical_ind":
    
    start_date = '2010-01-01'
    end_date = '2024-11-01'
    
    from _0_process_data._01_incorporate_technical_indicators import generate_technical_ind_dataset
    technical_ind_folder = os.path.join(transformed_data_dir, 'technical')
    filename = 'technical_' + str(stock) + '.csv'
    
    target_col_name = "Close"
    date_col_name = "Date"
    return_col = "Return"
    
    # start and end date in format 2024-11-13, '%Y-%m-%d'
    technical_ind_df = generate_technical_ind_dataset(stock, start_date, end_date)
    df = create_return_column(technical_ind_df, target_col_name, remove_close=False)
    
    selected_na_cols = list(df.columns)
    selected_scale_cols = list(df.drop(columns=[date_col_name]).columns) # All but Date

    df_clean = preprocess_data(df, selected_na_cols, return_col, selected_scale_cols, scaling_method)
    df_clean.head()
    
    df_clean.to_csv(os.path.join(technical_ind_folder, filename), index=False)


target_df = df_clean['Target']
features_df = df_clean.drop(columns = ["Date", "Target"])
df_clean = df_clean.drop(columns = ["Date"]) # date must be removed in any case


horizontal_filename = filename[:-4]
final_df = df_clean
if processing_method == "pca":
    from _0_process_data._01_pca import generate_rotated_pca_df
    pca_data_dir = os.path.join(transformed_data_dir, "pca")
    pca_df = generate_rotated_pca_df(features_df, target_df)
    final_df = pca_df

    pca_filename = 'pca_' + str(filename)
    horizontal_filename = pca_filename[:-4]
    pca_df.to_csv(os.path.join(pca_data_dir, pca_filename), index=False)


from _0_process_data._02_build_horizontal_df import split_dataframe, create_sequential_dataframe

window_size = 200
sequential_data, targets = split_dataframe(final_df, target_column='Target', window_size=window_size)
reshaped_df = create_sequential_dataframe(sequential_data, targets)

pkl_filename = f'{horizontal_filename}_{window_size}.pkl'
if data_type == "stock_single_name":
    structured_folder = os.path.join(structured_data_dir, "single_names")
elif data_type == "stock_technical_ind":
    structured_folder = os.path.join(structured_data_dir, "technical")
    
reshaped_df.to_pickle(os.path.join(structured_folder, pkl_filename))


model_type = "gru"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32

structured_df = pd.read_pickle(os.path.join(structured_folder, pkl_filename))


if model_type == "gru":
    from _12_nn_models.gru import apply_smote, convert_to_tensor
    
    X_resampled, y_resampled = apply_smote(structured_df, device) # balance data
    trainloader, testloader = convert_to_tensor(X_resampled, y_resampled, device, batch_size)


# **Choose hyperparameters**

# HYPERPARAMETERS

if model_type == "gru":
    from _12_nn_models.gru import GRU3DClassifier
    
    # parameters for GRU
    input_size = X_resampled.shape[2]
    hidden_size = 64  
    output_size = 2  
    num_layers = 2
    dropout = 0.2
    
    model = GRU3DClassifier(input_size, hidden_size, output_size, num_layers, dropout)
    model = model.to(device)
    
learning_rate = 0.1
num_epochs = 200

# choose optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# **Train model and test model**

import time
start_time = time.time()

if model_type == "gru":
    from _12_nn_models.gru import train_model, evaluate_model
    
    train_accuracy = train_model(model, optimizer, num_epochs, trainloader, criterion, device)
    test_accuracy = evaluate_model(model, testloader, criterion, device)

execution_time = end_time - start_time
end_time = time.time()


# ## **3. Store results**
# 


def store_results(
    output_filepath,  # Path to save or update the CSV file
    data_filepath,    # Path to the data file
    stock,            # Stock(s)
    data_type,        # Type of data (Technical/Economic/Options)
    start_date,       # Start date
    end_date,         # End date
    scaling_method,   # Scaling method (standard/minmax/none)
    processing_method,  # Processing method (PCA/none)
    window_size,      # Window size
    model_name,       # Model name
    batch_size,
    learning_rate,    # Learning rate
    num_epochs,       # Number of epochs
    execution_time,   # Execution time (seconds)
    num_layers,       # Number of layers
    criterion,
    optimizer,
    dropout_rate,
    train_accuracy,
    test_accuracy,
    **kwargs          # Additional model parameters (optional)
):
    """
    Appends results with data and model parameters to a CSV file.

    """
    # Organize all data-related columns
    data_params = {
        "Data Filepath": data_filepath,
        "Stock(s)": stock,
        "Data Type": data_type,
        "Start-End Date (Values)": f"{start_date} - {end_date}",
        "Scaling Method": scaling_method,
        "Processing Method": processing_method,
        "Window Size": window_size
    }

    # Organize all model-related columns
    model_params = {
        "Model": model_name,
        "Date Performed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Num Epochs": num_epochs,
        "Execution Time (s)": execution_time,
        "Number of Layers": num_layers,
        "Criterion" : criterion,
        "Optimizer" : optimizer,
        "Dropout Rate" : dropout_rate
    }
    
    result_params = {
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy
    }

    # Merge additional model parameters passed as kwargs
    model_params.update(kwargs)

    # Combine data and model parameters into one row
    result_row = {**data_params, **model_params, **result_params}

    # Load existing results or create a new DataFrame
    if os.path.exists(output_filepath):
        results_df = pd.read_csv(output_filepath)
    else:
        results_df = pd.DataFrame(columns=result_row.keys())

    # Append the new row using pd.concat
    new_row_df = pd.DataFrame([result_row])  # Create a DataFrame from the new row
    results_df = pd.concat([results_df, new_row_df], ignore_index=True)

    # Save back to the CSV
    results_df.to_csv(output_filepath, index=False)
    print(f"Results stored successfully in: {output_filepath}")

# In[24]:

results_dir = os.path.join(project_dir, "results")
results_filepath = os.path.join(results_dir, "results.csv")

store_results(
    results_filepath,
    horizontal_filename,
    stock,
    data_type,
    start_date,
    end_date,
    scaling_method,
    processing_method,
    window_size,
    model_type,
    batch_size,
    learning_rate,
    num_epochs,
    execution_time,
    num_layers,
    str(criterion),
    str(optimizer).split('\n')[0].strip(' ('),
    dropout,  # Example of additional model parameter
    train_accuracy,
    test_accuracy
)