import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

FIGURES_DIR = 'figures'
RESULTS_DIR = 'results'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# load data
df = pd.read_csv('hourly.csv')

# prepare features
feature_columns = ['hour', 'day_of_week', 'day_of_month', 
                   'weekend', 'workingday', 'is_rush_hour']

# convert to numeric
if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]
    feature_columns.append('station_numeric')

X = df[feature_columns]
y = df['demand']  # Target variable

print("\nFeatures being used:", feature_columns)
print("Target variable: demand")

# 70% train, 30% test (same as Weka percentage split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Create and train model
print("RANDOM FOREST - Baseline Model")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# get metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# correlation coefficient
correlation_rf = np.corrcoef(y_test, y_pred_rf)[0, 1]

print(f"Correlation coefficient:      {correlation_rf:.3f}")
print(f"Mean Absolute Error (MAE):    {mae_rf:.4f} bikes")
print(f"Root Mean Squared Error:      {rmse_rf:.4f} bikes")
print(f"R-squared:                    {r2_rf:.4f}")

# feature importance
print("\n FEATURE IMPORTANCE (Random Forest)")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))
feature_importance.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_base.csv'), index=False)

# create some plots

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Demand (bikes)')
plt.ylabel('Predicted Demand (bikes)')
plt.title('Random Forest (Actual vs Predicted Demand)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'rf_actual_vs_predicted_base.png'), dpi=300)
print(f"\nSaved: {FIGURES_DIR}/rf_actual_vs_predicted_base.png")

# Plot 2: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance_base.png'), dpi=300)
print(f"Saved: {FIGURES_DIR}/feature_importance_base.png")

print("\nFiles created:")
print(f"  1. {RESULTS_DIR}/feature_importance_base.csv")
print(f"  2. {FIGURES_DIR}/rf_actual_vs_predicted_base.png")
print(f"  3. {FIGURES_DIR}/feature_importance_base.png")