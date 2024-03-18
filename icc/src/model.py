import logging
from icc.config import Columns, HyperPars, Config
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import time
from icc.src.api import read_data

# Configure logging
logging.basicConfig(level=logging.INFO)

def prepare_model(df, test_size=HyperPars.TEST_SIZE, random_state=HyperPars.RANDOM_STATE):
    """
    Prepares DataFrame for model training and testing.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        test_size (float): Size of the test dataset. Default is HyperPars.TEST_SIZE.
        random_state (int): Random state for reproducibility. Default is HyperPars.RANDOM_STATE.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logging.info("Preparing DataFrame for Model.")
    df[Columns.SEX] = df[Columns.SEX].apply(lambda sex: 1 if sex=='female' else 0)
    df = get_dummies(df)

    X = df.drop(Columns.FACT, axis=1)
    y = df[Columns.FACT]

    logging.info("Splitting Model into Train and Test.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.info(f'Training Set Shape: {X_train.shape}.')
    logging.info(f'Testing Set Shape: {X_test.shape}.')

    sc = StandardScaler()

    logging.info("Transforming Numerical Columns using Standard Scaler.")
    X_train[[Columns.AGE, Columns.BMI]] = sc.fit_transform(X_train[[Columns.AGE, Columns.BMI]])
    X_test[[Columns.AGE, Columns.BMI]] = sc.transform(X_test[[Columns.AGE, Columns.BMI]])

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    logging.info("Done.")

    return X_train, X_test, y_train, y_test

def perform_cross_validation(df, model):
    """
    Performs cross-validation for the given model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        model (object): Model object to be cross-validated.
    """
    X_train, _, y_train, _ = prepare_model(df)

    logging.info("Performing Cross Validation.")
    kf = KFold(n_splits=5)
    cv_scores_train = []
    cv_scores_test = []

    logging.info("Fitting Model.")
    for train_indices, test_indices in kf.split(X_train):
        train = X_train.iloc[train_indices,:]
        train_targets = y_train.iloc[train_indices]
        test = X_train.iloc[test_indices,:]
        test_targets = y_train.iloc[test_indices]
        model.fit(train, train_targets)
        cv_scores_train.append(model.score(train, train_targets))
        cv_scores_test.append(model.score(test, test_targets))
    
    logging.info(f"Mean R2 score for train: {sum(cv_scores_train)/5:.2f}." )
    logging.info(f"Mean R2 score for test: {sum(cv_scores_test)/5:.2f}.")
    logging.info("Done.")
    

def evaluate_model(df, model):
    """
    Evaluates the trained model using test data.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        model (object): Trained model object.
    """
    logging.info("Running Validated Model.")
    X_train, X_test, y_train, y_test = prepare_model(df)
    model.fit(X_train, y_train)

    logging.info("Calculate Prediction on Test Data.")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"R2 Score on the Test Data is {r2:.2f}.")

    abs_error = mean_absolute_error(y_test, y_pred)
    logging.info(f"The Absolute Mean Error on the Test Data is {abs_error:.2f}.")
    logging.info("Done.")

    return r2

def run_random_forest_regressor(df, hyper_parameters=HyperPars):
    """
    Runs Random Forest Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        HyperPars (class): Class containing hyperparameters. Default is HyperPars.
    """
    model_rf = RandomForestRegressor(n_estimators=hyper_parameters.N_ESTIMATOR,
                                     max_depth=hyper_parameters.MAX_DEPTH,
                                     min_samples_leaf=hyper_parameters.MIN_SAMPLES_LEAF,
                                     min_samples_split=hyper_parameters.MIN_SAMPLES_SPLIT,
                                     random_state=hyper_parameters.RANDOM_STATE)
    perform_cross_validation(df, model=model_rf)
    evaluate_model(df, model_rf)

def run_knn(df, NN=HyperPars.N_NEIGHBORS):
    """
    Runs K Nearest Neighbors Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
    """
    model_knn = KNeighborsRegressor(n_neighbors=NN)
    perform_cross_validation(df, model_knn)
    evaluate_model(df, model_knn)

def run_SVM(df):
    """
    Runs Supported Vector Machine Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
    """
    model_svm = make_pipeline(StandardScaler(), SVR())
    evaluate_model(df=df, model=model_svm)


def run_gradient_boosting(df, hyper_parameters=HyperPars):
    """
    Runs Gradient Boosting Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        HyperPars (class): Class containing hyperparameters. Default is HyperPars.
    """
    model_gb = GradientBoostingRegressor(n_estimators=hyper_parameters.N_ESTIMATOR,
                                          max_depth=hyper_parameters.MAX_DEPTH,
                                          learning_rate=hyper_parameters.LEARNING_RATE,
                                          random_state=hyper_parameters.RANDOM_STATE)
    perform_cross_validation(df, model=model_gb)
    evaluate_model(df, model_gb)


class TaskModel:
    def __init__(self, conf=Config):
        self.file_path = conf.PATH_TO_DATA
        self.df = read_data(self.file_path)

    def run(self, hyper_parameters):
        start_time = time.time()
        logging.info("Starting Model Training Task.")

        logging.info("Running Random Forest Regressor.")
        run_random_forest_regressor(self.df, hyper_parameters=hyper_parameters)

        logging.info("Running KNN Regressor.")
        run_knn(self.df, NN=hyper_parameters.N_NEIGHBORS)

        logging.info("Running Supported Vector Machine.")
        run_SVM(self.df)

        logging.info("Running Gradient Boosting Regressor.")
        run_gradient_boosting(self.df, hyper_parameters=hyper_parameters)

        end_time = time.time()
        logging.info(f"Task finished in {end_time-start_time:.4f} seconds.")

    def get_train_model(self, model_name="RandomForest", hyper_parameters=HyperPars):
        """
        Trains and returns the specified model.

        Args:
            model_name (str): Name of the model to be trained.
            hyper_parameters (object): Hyperparameters object. Default is None.

        Returns:
            object: Trained model object.
        """
        if model_name == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=hyper_parameters.N_ESTIMATOR,
                max_depth=hyper_parameters.MAX_DEPTH,
                min_samples_leaf=hyper_parameters.MIN_SAMPLES_LEAF,
                min_samples_split=hyper_parameters.MIN_SAMPLES_SPLIT,
                random_state=hyper_parameters.RANDOM_STATE
            )

        elif model_name == "GradientBooster":
            model = GradientBoostingRegressor(n_estimators=hyper_parameters.N_ESTIMATOR,
                                          max_depth=hyper_parameters.MAX_DEPTH,
                                          learning_rate=hyper_parameters.LEARNING_RATE,
                                          random_state=hyper_parameters.RANDOM_STATE)
        
        elif model_name == "KNN":
            model = KNeighborsRegressor(n_neighbors=hyper_parameters.N_NEIGHBORS)

        elif model_name == "SVM":
            model = make_pipeline(StandardScaler(), SVR())

        else:
            ValueError("No valid model name. Please enter RandomForest, GradientBooster, KNN or SVM.")
        
        X_train, _, y_train, _ = prepare_model(self.df)
        model.fit(X_train, y_train)

        return model