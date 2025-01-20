import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Load the data
def load_data(path):
    data = pd.read_csv(path)
    return data

# Data Preprocessing
def preprocess_data(data):
    print(data.info())
    print(data.describe())
    
    # Droping missing values
    data = data.dropna()

    # One-Hot Encoding
    data_encoded = pd.get_dummies(data['ocean_proximity'], prefix='Location')
    data_encoded = data_encoded.astype(int)
    data = data.drop('ocean_proximity', axis=1)
    data = pd.concat([data, data_encoded], axis=1)
    print(data.head())
    print(data.info())



# Exploratory Data Analysis


# Model Building

# Main workflow
def main():
    # Load the data
    data = load_data(r"C:\Users\PRATHMESH\OneDrive\Documents\housing.csv")
    
    # Preprocess the data
    data = preprocess_data(data)
    
    '''# Perform exploratory data analysis
    perform_eda(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split_data(data)
    
    # Build and evaluate the model
    build_and_evaluate_model(X_train, X_test, y_train, y_test)'''

# Run the main function
if __name__ == "__main__":
    main()