from icc.src.model import Task_Model
from icc.src.plot import Task_EDA
from icc.src.api import read_data
from icc.config import HyperPars

# Initialize Dataframe
global _dataframe
_dataframe = read_data()

# Initialize Task Classes
task_eda = Task_EDA(df=_dataframe)
task_model = Task_Model(df=_dataframe)

task_eda.run()
task_model.run(hyperparameters=HyperPars)