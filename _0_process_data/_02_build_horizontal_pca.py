#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


project_dir = "/home/jupyter-tfg2425paula"
os.chdir(project_dir)
data_dir = os.path.join(project_dir, "raw_data")
options_dir = os.path.join(data_dir, "options_and_combinations")
pca_dir = os.path.join(data_dir, "pca")


# In[3]:


import sklearn.preprocessing

def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    
    for column in numeric_columns:  # Use 'df' instead of 'normalized_df' to iterate over its columns
        df[column] = min_max_scaler.fit_transform(df[column].values.reshape(-1, 1))  # Correct access to columns
    return df


# In[8]:


stocks = 'AAPL_MSFT_AMZN_NVDA_SPX'
filename = f'rotated_{stocks}_options.csv'
normalized_df = pd.read_csv(os.path.join(pca_dir, filename))
normalized_df.head()


# In[9]:


def split_dataframe(df, target_column, window_size):
    """
    Splits the DataFrame into sequential portions of size `window_size`.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be split.
    target_column (str): Name of the target column that indicates future changes.
    window_size (int): The size of each sequential portion.

    Returns:
    list: A list of DataFrames, each of size `window_size`.
    list: Corresponding targets for each sequential portion.
    """
    sequential_data = []
    targets = []

    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size].copy()  # Selects a window of size `window_size`
        target = df.iloc[i + window_size - 1][target_column]  # Target is the last value in the window
        sequential_data.append(window)
        targets.append(target)

    return sequential_data, targets


# In[10]:


import pandas as pd

window_size = 200
sequential_data, targets = split_dataframe(normalized_df, target_column='Target', window_size=window_size)

print(f"Number of sequential portions: {len(sequential_data)}")
print("Example of one sequential portion:")
print(sequential_data[0])  # Show the first sequential portion
print("Corresponding target:", targets[0])


# In[11]:


import pandas as pd

def create_sequential_dataframe(sequential_data, targets):
    """
    Creates a reshaped DataFrame where each row contains sequential data for each feature
    and a corresponding target value.

    Parameters:
    sequential_data (list): List of DataFrames representing sequential data portions.
    targets (list): List of target values corresponding to each sequence.

    Returns:
    pd.DataFrame: Reshaped DataFrame with each row containing sequential data for each feature
                  and the corresponding target value.
    """
    reshaped_rows = []

    for i, window_df in enumerate(sequential_data):
        row_data = {}
        # Iterate over columns (features) in the window
        for col in window_df.columns:
            # Create a new column for each feature across the window size
            row_data[col] = pd.Series(window_df[col].values)

        # Add the corresponding target for the sequence
        row_data['Target'] = targets[i]

        reshaped_rows.append(row_data)

    # Convert to DataFrame
    reshaped_df = pd.DataFrame(reshaped_rows)
    return reshaped_df


# In[12]:


sequential_data, targets = split_dataframe(normalized_df, target_column='Target', window_size=window_size)
reshaped_df = create_sequential_dataframe(sequential_data, targets)
reshaped_df


# In[15]:


output_filename = f'rotated_{stocks}_{window_size}.pkl'
output_data_folder = os.path.join(project_dir, 'processed_data/options_and_combinations')
reshaped_df.to_pickle(os.path.join(output_data_folder, output_filename))


# In[ ]:




