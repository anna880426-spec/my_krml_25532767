def pop_target(df, target_col):
    """
    Pop the target column from the dataframe and return it as a separate series.
    
    """
    target = df.pop(target_col)
    return df, target