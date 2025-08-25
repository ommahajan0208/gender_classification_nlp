import os
import pandas as pd

# def load_data(filename: str, data_dir: str = "data/raw") -> pd.DataFrame:
#     filepath = os.path.join(data_dir, filename)
#     return pd.read_csv(filepath)

def load_data(filename: str, data_dir: str = "data/raw") -> pd.DataFrame:
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath, encoding="latin1")