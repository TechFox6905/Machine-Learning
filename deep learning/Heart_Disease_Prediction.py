import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt

url = r"D:\VsCode\Workspace\Machine Learning\deep learning\cardio_train.csv"
data = pd.read_csv(url, delimiter=';')

def preprocessing(data: pd.DataFrame) -> tuple:
    
    print(data.isnull().sum())
    data = data.dropna()  # Drop or use imputation
    data['age'] = data['age'] // 365 # To convert in years

    # Remove outliers
    data = data[(data['ap_hi'] > 80) & (data['ap_hi'] < 200)]
    data = data[(data['ap_lo'] > 40) & (data['ap_lo'] < 120)]
    data = data[data['ap_hi'] > data['ap_lo']]  # Ensures logical BP values

    print(data.info())  
    print(data.describe())
    print(data['cardio'].value_counts())

    # Define features and target
    X = data.drop(columns=['cardio'])
    y = data['cardio']

    # Define numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing pipelines
    num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

    preprocessor = make_column_transformer(
        (num_pipeline, num_cols),
        (cat_pipeline, cat_cols)
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    input_shape = [X_train.shape[1]]

    return X_train, X_test, y_train, y_test, input_shape

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics = ['binary_accuracy'])
    
    return model

# Plot the Graph
def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def main():
    X_train, X_test, y_train, y_test, input_shape = preprocessing(data)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, input_shape)
    '''
    model = build_model(input_shape)
    model.summary()
    
    early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)
    history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, batch_size=1002, callbacks=[early_stopping],)
    plot_history(history)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
'''

main()
