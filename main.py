from icc.src.model import TaskModel, evaluate_model
from icc.src.plot import TaskEDA
from icc.src.person import Person
from icc.config import HyperPars, Config, Columns
import pandas as pd

# Initialize Task Classes
# task_eda = TaskEDA(conf=Config)
task_model = TaskModel(conf=Config)

# Run plot routine
# task_eda.run()

# Run model fits
# task_model.run(hyper_parameters=HyperPars)
mod = task_model.get_train_model()

print(mod)

p1 = Person(25, "male", 24.5, 0, False, "southeast")
# p1.predict_charges(mod)

print(p1.predict_charges(mod))