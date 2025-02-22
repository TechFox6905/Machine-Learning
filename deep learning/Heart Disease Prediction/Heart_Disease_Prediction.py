import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def preprocessing(X, preprocessor=None):
    """Preprocess the dataset: handle missing values, scale features, encode categorical variables."""
    num_features = X.select_dtypes(include=['number']).columns
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    if preprocessor is None:
        preprocessor = ColumnTransformer([('num', num_transformer, num_features)])
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    return X_transformed, preprocessor

def build_model(input_shape):
    """Build a simple neural network model for classification."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred.round())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def evaluate_model_comprehensively(model, X_train, y_train, X_test, y_test):
    """Evaluate model with multiple metrics and visualizations."""
    y_pred = model.predict(X_test)
    
    print("Classification Report:\n", classification_report(y_test, y_pred.round()))
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)
    
    # SHAP explainability
    explainer = shap.KernelExplainer(model.predict, X_train[:20])
    shap_values = explainer.shap_values(X_train[:20])
    shap.summary_plot(shap_values, X_train[:20])

# Load dataset
data = pd.read_csv('D:\VsCode\Workspace\Machine Learning\deep learning\Heart Disease Prediction\cardio_train.csv', delimiter=";")
# Remove unrealistic blood pressure values
data = data[(data['ap_hi'] >= 80) & (data['ap_hi'] <= 200)]
data = data[(data['ap_lo'] >= 40) & (data['ap_lo'] <= 150)]
data["bmi"] = data["weight"] / (data["height"] / 100) ** 2

X = data.drop(columns=['cardio'])  # Change 'cardio' to the actual column name
y = data['cardio']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train, preprocessor = preprocessing(X_train)  # Fit the preprocessor on training data
X_test, _ = preprocessing(X_test, preprocessor)  # Transform test data using fitted preprocessor

# Build model
model = build_model(X_train.shape[1])

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Plot training history
plot_training_history(history)

# Evaluate model
evaluate_model_comprehensively(model, X_train, y_train, X_test, y_test)

# Save Model & Preprocessor
model.save("heart_disease_model.h5")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Model and preprocessor saved successfully!")
