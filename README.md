# ICC - Insurance ​Charge Calculator

## Introduction
​​
In this project, we will be working with a dataset on insurance charges posted by The Devastator on Kaggle. The dataset contains information about insurance customers including their age, sex, BMI, etc. We will be doing some Exploratory Data Analysis (EDA), followed by some preprocessing and then modelling using various machine learning algorithms to predict the insurance charges for a customer given their features.

## Columns in the Dataset

1. age: The age of the customer in years.
2. sex: The sex of the customer (male, female).
3. bmi: The Body Mass Index of the customer (kg/m^2)
4. children: The number of children the customer has.
5. smoker: Whether the customer is a smoker or not.
6. region: The region of the customer.
7. charges: The insurance charges paid by the customer.

## Head of the Dataframe

| index | age | sex    | bmi    | children | smoker | region    | charges     |
|-------|-----|--------|--------|----------|--------|-----------|-------------|
| 0     | 19  | female | 27.900 | 0        | yes    | southwest | 16884.92400 |
| 1     | 18  | male   | 33.770 | 1        | no     | southeast | 1725.55230  |
| 2     | 28  | male   | 33.000 | 3        | no     | southeast | 4449.46200  |
| 3     | 33  | male   | 22.705 | 0        | no     | northwest | 21984.47061 |
| 4     | 32  | male   | 28.880 | 0        | no     | northwest | 3866.85520  |

## Summary Statistics

|         | index       | age        | bmi        | children   | charges     |
|---------|-------------|------------|------------|------------|-------------|
| count   | 1338.000000 | 1338.000000 | 1338.000000 | 1338.000000 | 1338.000000 |
| mean    | 668.500000  | 39.207025  | 30.663397  | 1.094918   | 13270.422265 |
| std     | 386.391641  | 14.049960  | 6.098187   | 1.205493   | 12110.011237 |
| min     | 0.000000    | 18.000000  | 15.960000  | 0.000000   | 1121.873900  |
| 25%     | 334.250000  | 27.000000  | 26.296250  | 0.000000   | 4740.287150  |
| 50%     | 668.500000  | 39.000000  | 30.400000  | 1.000000   | 9382.033000  |
| 75%     | 1002.750000 | 51.000000  | 34.693750  | 2.000000   | 16639.912515 |
| max     | 1337.000000 | 64.000000  | 53.130000  | 5.000000   | 63770.428010 |

It does not seem that there are any outliers as such. For charges, the 75th percentile is 16639 whereas the max is 63770, which can be interpreted as being an outlier. However, to maintain some 'realness' in the data, I would not interpret these as such.

The columns make sense, as they should surely affect the insurance cost. Older people will probably pay more insurance cost, people with higher BMIs may be suffering from other illnesses such as diabetes and hence may have higher costs, smokers may pay more as well, etc.

# Structure

The project is structured as follows:

- api module: extracting the data
- plot module: plotting routines for EDA
- model module: preprocessing and running the model
- prediction module: prediction based on the model

# Requirements
pandas, scikit-learn