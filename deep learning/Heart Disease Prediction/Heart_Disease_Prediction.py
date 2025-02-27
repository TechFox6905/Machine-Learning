import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

# ------------------------------- Data Preprocessing -------------------------------
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

# ------------------------------- Model Building -------------------------------
def build_model(input_shape):
    """Build an optimized neural network model for classification."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    
    return model

# ------------------------------- Visualization Functions -------------------------------
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

def plot_roc_curve(y_true, y_pred_probs):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_precision_recall(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

# ------------------------------- Model Evaluation -------------------------------
def evaluate_model_comprehensively(model, X_train, y_train, X_test, y_test):
    """Evaluate model with multiple metrics and visualizations."""
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_probs)
    plot_precision_recall(y_test, y_pred)
    
    # SHAP explainability
    explainer = shap.GradientExplainer(model, X_train[:100])  # More efficient than DeepExplainer
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100])

# ------------------------------- Load & Prepare Dataset -------------------------------
data = pd.read_csv(r'D:\VsCode\Workspace\Machine Learning\deep learning\Heart Disease Prediction\cardio_train.csv', delimiter=";")

# Remove unrealistic blood pressure values
data = data[(data['ap_hi'] >= 80) & (data['ap_hi'] <= 200)]
data = data[(data['ap_lo'] >= 40) & (data['ap_lo'] <= 150)]
data["bmi"] = data["weight"] / (data["height"] / 100) ** 2

print(data.dtypes)

X = data.drop(columns=['cardio'])
y = data['cardio']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y).astype(np.int64)

# Split dataset
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Handle Imbalanced Data
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Preprocess Data
X_train, preprocessor = preprocessing(X_train)
X_test, _ = preprocessing(X_test, preprocessor)

# Build & Train Model
model = build_model(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping, reduce_lr])

plot_training_history(history)
evaluate_model_comprehensively(model, X_train, y_train, X_test, y_test)

# Save Model & Preprocessor
model.save("heart_disease_model.h5")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Model and preprocessor saved successfully!")
