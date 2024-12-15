import pandas as pd


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