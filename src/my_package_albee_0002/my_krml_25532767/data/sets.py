import pandas as pd
import os

def pop_target(df, target_col):
    """
    Pop the target column from the dataframe and return it as a separate series.
    
    """
    target = df.pop(target_col)
    return df, target


def save_sets(X_train, y_train, X_val, y_val, X_test, y_test, path):
    """
    Save the features and target sets to CSV files.
    
    """
    if X_train is not None:
        X_train.to_csv(f"{path}/X_train.csv", index=False)
    if y_train is not None:
        y_train.to_csv(f"{path}/y_train.csv", index=False)
    if X_val is not None:
        X_val.to_csv(f"{path}/X_val.csv", index=False)
    if y_val is not None:
        y_val.to_csv(f"{path}/y_val.csv", index=False)
    if X_test is not None:
        X_test.to_csv(f"{path}/X_test.csv", index=False)
    if y_test is not None:
        y_test.to_csv(f"{path}/y_test.csv", index=False)


def load_sets(path):
    """
    Load the features and target sets from CSV files.
    
    """
    X_train = pd.read_csv(f"{path}/X_train.csv") if os.path.exists(f"{path}/X_train.csv") else None
    y_train = pd.read_csv(f"{path}/y_train.csv") if os.path.exists(f"{path}/y_train.csv") else None
    X_val = pd.read_csv(f"{path}/X_val.csv") if os.path.exists(f"{path}/X_val.csv") else None
    y_val = pd.read_csv(f"{path}/y_val.csv") if os.path.exists(f"{path}/y_val.csv") else None
    X_test = pd.read_csv(f"{path}/X_test.csv") if os.path.exists(f"{path}/X_test.csv") else None
    y_test = pd.read_csv(f"{path}/y_test.csv") if os.path.exists(f"{path}/y_test.csv") else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# subset: 子集合 or 擷取出一個子集合

def subset_x_y(target, features, start_index, end_index):
    """
    Subset the features and target dataframes based on the provided date range.
    
    """
    return features.iloc[start_index:end_index], target.iloc[start_index:end_index]


def split_sets_by_time(df, target_col, test_ratio=0.2):
    """
    Split the dataframe into training and testing sets based on the provided test ratio.
    
    """
    # Sort the dataframe by date if it has a date column
    if 'date' in df.columns:
        df = df.sort_values(by='date')
    
    # Calculate the index to split the data
    split_index = int(len(df) * (1 - test_ratio))
    
    # Split the dataframe into training and testing sets
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Further split the training set into training and validation sets
    val_index = int(len(train_df) * 0.2)
    val_df = train_df.iloc[:val_index]
    train_df = train_df.iloc[val_index:]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test