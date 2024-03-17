from src.model import TaskModel
from src.plot import TaskEDA
from src.person import Person
from config import HyperPars, Config

# Initialize Task Classes
# task_eda = TaskEDA(conf=Config)
task_model = TaskModel(conf=Config)

# Run plot routine
# task_eda.run()

# Run model fits
# task_model.run(hyper_parameters=HyperPars)
mod = task_model.get_train_model()

p1 = Person(25, "male", 24.5, 0, False, "southwest")
p1.predict_charges(mod)