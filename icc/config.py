
class Config:
    PATH_TO_DATA = "icc/data/insurance.csv"
    PATH_TO_PLOT_FOLDER = "icc/plots/"

class Columns:
    ID = "index"
    AGE = "age"
    SEX = "sex"
    BMI = "bmi"
    KIDS = "children"
    SMOKER = "smoker"
    LOC = "region"
    FACT = "charges"

class HyperPars:
    # General Parameters
    N_ESTIMATOR = 10
    RANDOM_STATE = 42 
    TEST_SIZE = 20

    # Logistic Regressor
    MAX_DEPTH = 20
    MIN_SAMPLES_LEAF = 5
    MIN_SAMPLES_SPLIT= 5
    
    # KNN
    N_NEIGHBORS = 4

    # Gradient Booster
    LEARNING_RATE = 0.01