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

        self.data = self.set_df()

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

    def set_df(self):
        if self._sex == "male":
            self._sex = 0
        else:
            self._sex = 1
        
        if self._smoker == "yes":
            self._smoker = 1
        else:
            self._smoker = 0

        locations = Columns.REGION.ALL

        if self._region not in locations:
            ValueError("No valid location used. Please use 'northwest', 'northeast', 'southwest' or 'southeast'.")
        
        match_location = [1 if location == self._region else 0 for location in locations]
        region_dict = {"region_" + location: res for location, res in zip(locations, match_location)}
        
        data = {
            Columns.ID: [1],
            Columns.AGE: [self._age],
            Columns.SEX: [self._sex],
            Columns.BMI: [self._bmi],
            Columns.KIDS: [self._children],
            Columns.SMOKER: [self._smoker],
        }
        data.update(region_dict)

        return DataFrame(data, index=[1])

    def predict_charges(self, model):
        """
        Predicts insurance charges for the person using the trained model.

        Args:
            model (object): Trained model object.

        Returns:
            float: Predicted insurance charges.
        """
        df_reordered = self.data[model.feature_names_in_] 
        charges_pred = model.predict(df_reordered)

        return charges_pred[0]
