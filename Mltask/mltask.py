import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization

# Load dataset
df = pd.read_csv("TASK-ML-INTERN.csv")

# Data exploration
print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# Target distribution
plt.figure(figsize=(10,5))
sns.histplot(df["vomitoxin_ppb"], bins=30, kde=True, color="blue")
plt.xlabel("Vomitoxin (ppb)")
plt.ylabel("Frequency")
plt.title("Distribution of Vomitoxin Levels")
plt.show()

# Feature-target split
X = df.drop(columns=["hsi_id", "vomitoxin_ppb"])
y = df["vomitoxin_ppb"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance by first 20 components:", sum(pca.explained_variance_ratio_))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("R² Score:", r2_score(y_test, y_pred_rf))

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost
print("XGBoost Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("R² Score:", r2_score(y_test, y_pred_xgb))

# CNN Model
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

cnn_model = Sequential([
    Conv1D(filters=128, kernel_size=9, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss='mse')
cnn_model.summary()

# Train CNN
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=16, validation_data=(X_test_cnn, y_test))

# CNN Predictions
y_pred_cnn = cnn_model.predict(X_test_cnn)

# Evaluate CNN
print("CNN Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_cnn))
print("RMSE:", mean_squared_error(y_test, y_pred_cnn, squared=False))
print("R² Score:", r2_score(y_test, y_pred_cnn))

# Scatter plot of actual vs. predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, label="Random Forest", alpha=0.5)
sns.scatterplot(x=y_test, y=y_pred_xgb, label="XGBoost", alpha=0.5)
sns.scatterplot(x=y_test, y=y_pred_cnn.flatten(), label="CNN", alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()
