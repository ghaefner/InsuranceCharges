from src.model import TaskModel
from src.plot import TaskEDA
from src.api import read_data
from config import HyperPars

# Initialize Dataframe
global _dataframe
_dataframe = read_data()

# Initialize Task Classes
task_eda = TaskEDA(df=_dataframe)
task_model = TaskModel(df=_dataframe)

# task_eda.run()
task_model.run(hyper_parameters=HyperPars)