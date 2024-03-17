from config import Columns
from pandas import DataFrame

class Person:
    def __init__(self, age, sex, bmi, children, smoker, region):
        self._age = age
        self._sex = sex
        self._bmi = bmi
        self._children = children
        self._smoker = smoker
        self._region = region

    # Setter functions
    def set_age(self, age):
        self._age = age

    def set_sex(self, sex):
        self._sex = sex

    def set_bmi(self, bmi):
        self._bmi = bmi

    def set_children(self, children):
        self._children = children

    def set_smoker(self, smoker):
        self._smoker = smoker

    def set_region(self, region):
        self._region = region

    def predict_charges(self, model):
        """
        Predicts insurance charges for the person using the trained model.

        Args:
            model (object): Trained model object.

        Returns:
            float: Predicted insurance charges.
        """
        data = {
            Columns.AGE: [self._age],
            Columns.SEX: [self._sex],
            Columns.BMI: [self._bmi],
            Columns.KIDS: [self._children],
            Columns.SMOKER: [self._smoker],
            Columns.LOC: [self._region]
        }
        
        data = DataFrame(data)
        charges_pred = model.predict(data)

        return charges_pred[0]
