from pandas import read_csv, DataFrame
from icc.config import Config, Columns

def read_data(file_path=Config.PATH_TO_DATA) -> DataFrame:
    df = read_csv(file_path)
    df[Columns.SMOKER] = df[Columns.SMOKER].apply(lambda x: 1 if x=='yes' else 0)
    return df
