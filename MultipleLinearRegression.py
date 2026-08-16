# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("MultipleLinearRegression/50_Startups.csv")
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1].values

# Splitting the data into Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Encoding of the dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output= False), [3])],  # [3] = State column
    remainder='passthrough'
)

# Fit ONLY on X_train — the encoder learns categories from training data only
X_train = np.array(ct.fit_transform(X_train))
print("X_train after encoding:\n", X_train)

# Transform X_test using the SAME fitted encoder — do NOT re-fit
X_test = np.array(ct.transform(X_test))
print("X_test after encoding:\n", X_test)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

# Display the predicted and real profit
print("Predicted Profit  Actual Profit")
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred), 1),
         y_test.reshape(len(y_test), 1)),
        axis=1
    )
)
