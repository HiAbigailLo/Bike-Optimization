# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Load your hourly demand data
# ==========================================
df = pd.read_csv('hourly.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# ==========================================
# STEP 2: Prepare features (X) and target (y)
# ==========================================

# Features (inputs)
feature_columns = ['hour', 'day_of_week', 'day_of_month', 
                   'weekend', 'workingday', 'is_rush_hour']

# If you have station_id as text, convert to numeric
if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]
    feature_columns.append('station_numeric')

X = df[feature_columns]
y = df['demand']  # Target variable

print("\nFeatures being used:", feature_columns)
print("Target variable: demand")

# ==========================================
# STEP 3: Split into train and test sets
# ==========================================

# 70% train, 30% test (same as Weka percentage split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ==========================================
# STEP 4: Train Random Forest
# ==========================================

print("\n" + "="*50)
print("RANDOM FOREST - Baseline Model")
print("="*50)

# Create and train model (100 trees like Weka)
rf_model = RandomForestRegressor(
    n_estimators=100,  # 100 trees (same as Weka)
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Correlation coefficient (like Weka)
correlation_rf = np.corrcoef(y_test, y_pred_rf)[0, 1]

print(f"Correlation coefficient:      {correlation_rf:.3f}")
print(f"Mean Absolute Error (MAE):    {mae_rf:.4f} bikes")
print(f"Root Mean Squared Error:      {rmse_rf:.4f} bikes")
print(f"R-squared:                    {r2_rf:.4f}")

# ==========================================
# STEP 5: Feature importance
# ==========================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# Save feature importance
feature_importance.to_csv('feature_importance_base.csv', index=False)

# ==========================================
# STEP 6: Visualizations
# ==========================================

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Demand (bikes)')
plt.ylabel('Predicted Demand (bikes)')
plt.title('Random Forest: Actual vs Predicted Demand')
plt.legend()
plt.tight_layout()
plt.savefig('rf_actual_vs_predicted_base.png', dpi=300)
print("\nSaved: rf_actual_vs_predicted.png")

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_base.png', dpi=300)
print("Saved: feature_importance.png")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nFiles created:")
print("  1. feature_importance.csv")
print("  2. rf_actual_vs_predicted.png")
print("  3. feature_importance.png")