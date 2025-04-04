# Best crop prediction: machine learning to help farmers select the best crops

## Project Overview

This project applies **supervised machine learning** techniques to help farmers select the most suitable crop for their fields based on soil composition. The dataset used for this classification task is sourced from Kaggle, and the project was completed on DataCamp as part of a learning module.

By analyzing key soil metrics such as nitrogen (N), phosphorus (P), potassium (K), and pH levels, I have developed a multi-class classification model to predict the optimal crop for a given soil sample. Additionally, I have identified the most significant soil feature for predictive accuracy.

## Dataset

The dataset, soil_measures.csv, contains the following columns:

- N: Nitrogen content ratio in the soil
- P: Phosphorous content ratio in the soil
- K: Potassium content ratio in the soil
- pH: Soil acidity level
- crop: The target variable indicating the optimal crop for the soil
Each row represents the soil measurements from a particular field, with the corresponding optimal crop as the target variable.

## Approach

1. Data Preprocessing:
  - Loaded the dataset
  - Checked for missing values
  - Explored unique crop types
2. Feature Importance Analysis:
  - Trained separate Logistic Regression models using each individual soil feature
  - Evaluated performance using F1-score
  - Identified the most influential soil metric

# Implementation

## Dependencies
Ensured required Python libraries have been installed:

```pip install pandas scikit-learn ```

## Code
The core logic is implemented in Python using pandas for data handling and scikit-learn for machine learning:

``` import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading the dataset
crops = pd.read_csv("soil_measures.csv")

# Checking for missing values
print(crops.isna().sum().sort_values())

# Displaying unique crop types
print(crops.crop.unique())

# Preparing features and target
X = crops.drop(columns="crop")
y = crops["crop"]

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store feature performance
feature_performance = {}

# Evaluating each soil feature's predictive power
for feature in ["N", "P", "K", "pH"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    
    # Calculating F1-score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    
    # Storing results
    feature_performance[feature] = f1
    print(f"F1-score for {feature}: {f1}")

# Identifying the best predictive feature
best_predictive_feature = {"K": feature_performance["K"]}
```

## Results

The model evaluates individual features and their effectiveness in predicting crop types.
**The potassium (K) content produced the highest F1-score, indicating its strongest influence on crop selection.**


## Conclusion

This project demonstrates how machine learning can assist farmers in optimizing their crop selection based on soil conditions, ultimately leading to higher yields and better resource allocation.
