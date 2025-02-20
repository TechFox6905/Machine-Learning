import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import stats
import shap
from sklearn.inspection import permutation_importance

# Previous preprocessing, build_model, and plotting functions remain the same...

def perform_cross_validation(X, y, n_splits=5):
    """
    Performs k-fold cross-validation and analyzes model stability across folds.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {
        'accuracy': [], 'auc': [], 'precision': [], 'recall': []
    }
    
    print(f"\n=== {n_splits}-Fold Cross-Validation Results ===")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        # Split data for this fold
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Build and train model
        model = build_model([X.shape[1]])
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=50,
            batch_size=1024,
            validation_data=(X_val_fold, y_val_fold),
            verbose=0
        )
        
        # Evaluate on validation fold
        metrics = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        
        # Store metrics
        for metric_name, value in zip(model.metrics_names, metrics):
            if metric_name in fold_metrics:
                fold_metrics[metric_name].append(value)
                
        print(f"\nFold {fold} Results:")
        for metric_name, value in zip(model.metrics_names, metrics):
            print(f"{metric_name}: {value:.4f}")
    
    # Calculate and display statistical summary
    print("\nCross-Validation Summary:")
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric}: {mean_val:.4f} (±{std_val:.4f})")

def plot_calibration_curve(y_true, y_pred):
    """
    Creates a reliability diagram to assess model calibration.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_feature_importance(model, X_train, y_train, feature_names):
    """
    Analyzes feature importance using multiple methods including permutation importance
    and SHAP values.
    """
    # Permutation Importance
    perm_importance = permutation_importance(
        model, X_train, y_train,
        n_repeats=10,
        random_state=42
    )
    
    # Sort features by importance
    importance_scores = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    importance_scores = importance_scores.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_scores.head(15), 
                x='Importance', y='Feature',
                palette='viridis')
    plt.title('Feature Importance (Permutation Method)')
    plt.show()
    
    # Calculate and plot SHAP values
    explainer = shap.KernelExplainer(model.predict, X_train[:100])
    shap_values = explainer.shap_values(X_train[:100])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[0], X_train[:100], 
                     feature_names=feature_names,
                     plot_type='bar')

def plot_precision_recall_curve(y_true, y_pred):
    """
    Plots the Precision-Recall curve and calculates average precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_prediction_distribution(y_pred):
    """
    Analyzes the distribution of model predictions to identify potential biases.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, bins=50)
    plt.title('Distribution of Model Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.show()
    
    # Calculate prediction statistics
    print("\nPrediction Distribution Statistics:")
    print(f"Mean prediction: {np.mean(y_pred):.4f}")
    print(f"Median prediction: {np.median(y_pred):.4f}")
    print(f"Std of predictions: {np.std(y_pred):.4f}")
    print(f"% predictions > 0.5: {(y_pred > 0.5).mean() * 100:.2f}%")

def evaluate_model_comprehensively(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Performs comprehensive model evaluation including all metrics and visualizations.
    """
    print("\n=== Starting Comprehensive Model Evaluation ===")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Basic classification metrics
    print("\n1. Classification Metrics:")
    print(classification_report(y_test, y_pred.round()))
    
    # Plot confusion matrix
    print("\n2. Confusion Matrix Visualization:")
    plot_confusion_matrix(y_test, y_pred)
    
    # ROC and Precision-Recall curves
    print("\n3. ROC Curve Analysis:")
    plot_roc_curve(y_test, y_pred)
    print("\n4. Precision-Recall Curve Analysis:")
    plot_precision_recall_curve(y_test, y_pred)
    
    # Model calibration
    print("\n5. Model Calibration Analysis:")
    plot_calibration_curve(y_test, y_pred)
    
    # Prediction distribution analysis
    print("\n6. Prediction Distribution Analysis:")
    analyze_prediction_distribution(y_pred)
    
    # Feature importance analysis
    print("\n7. Feature Importance Analysis:")
    analyze_feature_importance(model, X_train, y_train, feature_names)
    
    # Additional test metrics
    print("\n8. Additional Model Metrics:")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, input_shape, preprocessor = preprocessing(data)
    
    # Get feature names after preprocessing
    feature_names = (preprocessor.named_transformers_['num']
                    .get_feature_names_out().tolist())
    
    # Perform cross-validation
    perform_cross_validation(X_train, y_train)
    
    # Build and train final model
    model = build_model(input_shape)
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            monitor='val_auc'
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5,
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_split=0.2,
        batch_size=1024,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Comprehensive evaluation
    evaluate_model_comprehensively(
        model, X_train, X_test, y_train, y_test, feature_names
    )

if __name__ == "__main__":
    main()