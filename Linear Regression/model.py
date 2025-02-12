import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

# 1️⃣ Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Features (100 samples, 1 feature)
y = 4 + 3 * X + 0.2 * np.random.randn(100, 1)  # y = 4 + 3X + reduced noise

# 2️⃣ Split into Training and Validation sets using K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize hyperparameters
learning_rate = 0.3
n_iterations = 2000
lambda_reg = 0  # Regularization parameter (L2)

# Store loss history
loss_history = []
val_losses = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # 3️⃣ Initialize parameters
    m = np.random.randn()  # Random slope
    b = np.random.randn()  # Random intercept
    
    # 4️⃣ Gradient Descent Algorithm with L2 Regularization
    for _ in range(n_iterations):
        y_pred = m * X_train + b  # Predictions
        error = y_train - y_pred  # Errors
        
        # Compute Gradients with L2 Regularization
        dm = (-2 / len(X_train)) * np.sum(X_train * error) + 2 * lambda_reg * m
        db = (-2 / len(X_train)) * np.sum(error)
        
        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db
        
        # Compute loss (MSE)
        mse = np.mean(error ** 2)
        loss_history.append(mse)
    
    # 5️⃣ Evaluate on Validation Set
    y_val_pred = m * X_val + b
    val_mse = np.mean((y_val - y_val_pred) ** 2)
    val_losses.append(val_mse)
    
# Compute average validation loss
avg_val_loss = np.mean(val_losses)

# 6️⃣ Plot training loss
plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Iterations')
plt.show()

# 7️⃣ Bias-Variance Tradeoff Analysis
bias = np.mean((y - (m * X + b)) ** 2)  # Bias = Avg Error
variance = np.var(m * X + b)  # Variance = Spread of predictions
print(f'Bias: {bias:.4f}, Variance: {variance:.4f}')

print(f'Final Model: y = {m:.2f}X + {b:.2f}')
print(f'Average Validation MSE: {avg_val_loss:.4f}')

# 8️⃣ Visualize Data and Model
plt.scatter(X, y, label='Data')
plt.plot(X, m * X + b, color='black', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Model with Regularization')
plt.show()
