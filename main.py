from icc.src.model import Model
from icc.src.plot import Plot
from icc.src.person import Person
from icc.config import HyperPars, Config, Columns
import pandas as pd

# Initialize Task Classes
plot = Plot(conf=Config)
model = Model(conf=Config)

# Run plot routine
plot.run()

# Run model fits
model.run(hyper_parameters=HyperPars)

# Create a User and predict charges
p1 = Person(25, "male", 24.5, 0, False, "southeast")
p1.predict_charges(model.get_train_model())