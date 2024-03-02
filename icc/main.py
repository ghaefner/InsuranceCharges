from src.model import TaskModel
from src.plot import TaskEDA
from config import HyperPars, Config

# Initialize Task Classes
task_eda = TaskEDA(conf=Config)
task_model = TaskModel(conf=Config)

# Run plot routine
task_eda.run()

# Run model fits
task_model.run(hyper_parameters=HyperPars)

