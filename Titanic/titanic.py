# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, roc_curve)

# Load DataFrame
def load_df(file_path):
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

# Display basic information about the df
def display_info(df):
    """Display basic information and statistics about the DataFrame."""
    print(df.info())  # df types, non-null values
    print(df.describe())  # Summary statistics
    print(df.head())  # First 5 rows
    print(df.tail())  # Last 5 rows
    print(df.isnull().sum())  # Number of missing values in each column
    print(df.nunique())  # Number of unique values per column

    # Countplot for survival distribution
    sns.countplot(x=df["Survived"], palette=["red", "green"], hue=df["Survived"])
    plt.xticks([0, 1], ["Not Survived (0)", "Survived (1)"])
    plt.ylabel("Count")
    plt.title("Survival Distribution")
    plt.show()

# Data Preprocessing  
def preprocess_df(df):
    """Preprocess the DataFrame by cleaning and transforming data."""
    # Drop columns with too many missing values
    df = df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)

    # Drop Rows with Any Null Values
    df = df.dropna()

    # Convert categorical variables into numerical
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # Ensure all features are numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    df_int = df.select_dtypes(include=['int','float'])
    
    '''Plots for understanding the data'''
    '''
    # Correlation heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df_int.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Detect outliers in Fare
    sns.boxplot(y=df["Fare"])
    plt.title("Fare Outliers")
    plt.show()

    # Detect outliers in Age
    sns.boxplot(y=df["Age"])
    plt.title("Age Outliers")
    plt.show()'''
    return df 

def evaluate_model(y_test, y_pred, y_pred_proba=None):
    """Evaluate the model using various metrics."""
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")

    # Recall
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall}")

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1}")

    # ROC AUC
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {roc_auc}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color="blue")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    print("") # Blank line for separation

def find_best_threshold(y_test, y_probs):
    """Find the best threshold using Youden’s J statistic from the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    optimal_idx = (tpr - fpr).argmax()
    return thresholds[optimal_idx]

# Model Training Functions
def logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train and evaluate a Logistic Regression model."""
    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train_scaled, y_train)

    # Get predicted probabilities
    y_pred_proba = logreg.predict_proba(X_test_scaled)[:,1]
    
    # Find best threshold
    best_threshold = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("Logistic Regression")
    evaluate_model(y_test, y_pred, y_pred_proba)

def decision_tree(X_train, y_train, X_test, y_test, X):
    """Train and evaluate a Decision Tree model."""
    dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=40, random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Get predicted probabilities
    y_pred_proba = dt_classifier.predict_proba(X_test)[:,1]

    # Find best threshold
    best_threshold = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("Decision Tree")
    evaluate_model(y_test, y_pred, y_pred_proba)

    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(dt_classifier, feature_names=X.columns, class_names=["Not Survived", "Survived"], filled=True)
    plt.show()

def random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    rf_classifier = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=10, min_samples_split=5, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Get predicted probabilities
    y_pred_proba = rf_classifier.predict_proba(X_test)[:,1]

    # Find the best threshold using Youden’s J statistic (from ROC Curve)
    best_threshold = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("Random Forest")
    evaluate_model(y_test, y_pred, y_pred_proba)

def knn(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train and evaluate a K-Nearest Neighbors model."""
    knn_classifier = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_proba = knn_classifier.predict_proba(X_test_scaled)[:,1]

    # Find best threshold
    best_threshold = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("K-Nearest Neighbors")
    evaluate_model(y_test, y_pred, y_pred_proba)

def svm(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train and evaluate a Support Vector Machine model."""
    svm_classifier = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_proba = svm_classifier.decision_function(X_test_scaled)

    # Find best threshold
    best_threshold = find_best_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    print("Support Vector Machine")
    evaluate_model(y_test, y_pred, y_pred_proba)

# Main Function to Execute the Workflow
def main():
    file_path = r'D:\VsCode\Workspace\Machine Learning\Titanic\train.csv'
    
    # Load Data
    df = load_df(file_path)
    display_info(df)

    # Preprocess Data
    df = preprocess_df(df)
    
    # Define features and target variable
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and Evaluate Models
    logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    decision_tree(X_train, y_train, X_test, y_test, X)
    random_forest(X_train, y_train, X_test, y_test)
    knn(X_train_scaled, y_train, X_test_scaled, y_test)
    svm(X_train_scaled, y_train, X_test_scaled, y_test)

    print(X.head()) # Optional: Display first few rows of feature set

# Run the main function
if __name__ == "__main__":
    main()