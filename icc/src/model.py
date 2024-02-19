from config import Columns,HyperPars
from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import time


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
    print("[I] Preparing DataFrame for Model.")
    df[Columns.SEX] = df[Columns.SEX].apply(lambda sex: 1 if sex=='female' else 0)
    df = get_dummies(df)

    X = df.drop(Columns.FACT, axis=1)
    y = df[Columns.FACT]

    print("[I] Splitting Model into Train and Test.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f'[I] Training Set Shape: {X_train.shape}.')
    print(f'[I] Testing Set Shape: {X_test.shape}.')

    sc = StandardScaler()

    print("[I] Transforming Numerical Columns using Standard Scaler.")
    X_train[[Columns.AGE, Columns.BMI]] = sc.fit_transform(X_train[[Columns.AGE, Columns.BMI]])
    X_test[[Columns.AGE, Columns.BMI]] = sc.transform(X_test[[Columns.AGE, Columns.BMI]])

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("[I] Done.")

    return X_train, X_test, y_train, y_test

def perform_cross_validation(df, model):
    """
    Performs cross-validation for the given model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        model (object): Model object to be cross-validated.
    """
    X_train, _, y_train, _ = prepare_model(df)

    print("[I] Performing Cross Validation.")
    kf = KFold(n_splits=5)
    cv_scores_train = []
    cv_scores_test = []

    print("[I] Fitting Model.")
    for train_indices, test_indices in kf.split(X_train):
        train = X_train.iloc[train_indices,:]
        train_targets = y_train.iloc[train_indices]
        test = X_train.iloc[test_indices,:]
        test_targets = y_train.iloc[test_indices]
        model.fit(train, train_targets)
        cv_scores_train.append(model.score(train, train_targets))
        cv_scores_test.append(model.score(test, test_targets))
    
    print("[I] Mean R2 score for train: ", sum(cv_scores_train)/5)
    print("[I] Mean R2 score for test: ", sum(cv_scores_test)/5)
    print("[I] Done.")

def evaluate_model(df, model):
    """
    Evaluates the trained model using test data.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        model (object): Trained model object.
    """
    print("[I] Running Validated Model.")
    X_train, X_test, y_train, y_test = prepare_model(df)
    model.fit(X_train, y_train)

    print("[I] Calculate Prediction on Test Data.")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'[I] R2 Score on the Test Data is {r2:.2f}.')

    abs_error = mean_absolute_error(y_test, y_pred)
    print(f'[I] The Absolute Mean Error on the Test Data is {abs_error:,.2f}.')
    print("[I] Done.")

def run_random_forest_regressor(df, HyperPars=HyperPars):
    """
    Runs Random Forest Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
        HyperPars (class): Class containing hyperparameters. Default is HyperPars.
    """
    model_rf = RandomForestRegressor(n_estimators=HyperPars.N_ESTIMATOR,
                                     max_depth=HyperPars.MAX_DEPTH,
                                     min_samples_leaf=HyperPars.MIN_SAMPLES_LEAF,
                                     min_samples_split=HyperPars.MIN_SAMPLES_SPLIT,
                                     random_state=HyperPars.RANDOM_STATE)
    perform_cross_validation(df, model=model_rf)
    evaluate_model(df, model_rf)

def run_knn(df):
    """
    Runs K Nearest Neighbors Regressor model.

    Args:
        df (DataFrame): Input DataFrame containing features and target.
    """
    model_knn = KNeighborsRegressor(n_neighbors=4)
    perform_cross_validation(df, model_knn)
    evaluate_model(df, model_knn)

class Task_Model:
    def __init__(self, df):
        self.df = df

    def run(self):
        start_time = time.time()
        print("[I]: Info\n [W]: Warning\n [E]: Error")
        print("[I] Starting Model Training Task.")

        print("[I] Running Random Forest Regressor.")
        run_random_forest_regressor(self.df)

        end_time = time.time()
        print(f'[I] Task finished in {end_time-start_time: .4f} seconds.')
    


