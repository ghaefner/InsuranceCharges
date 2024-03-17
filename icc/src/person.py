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

    def get_charges(self):
        pass