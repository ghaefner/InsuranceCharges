from pandas import read_csv, DataFrame
from config import Config

def read_data(file_path=Config.PATH_TO_DATA) -> DataFrame:
    return read_csv(file_path)
