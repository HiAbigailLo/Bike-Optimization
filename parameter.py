# ==========================================
# PARAMETER TUNING (Simplified Model)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv('hourly.csv')

# Use SIMPLIFIED features (from feature selection)
selected_features = ['hour', 'day_of_week', 'day_of_month', 'station_numeric']

if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]

X = df[selected_features]
y = df['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("="*60)
print("PARAMETER TUNING (4-feature model)")
print("="*60)
print(f"Features: {selected_features}")

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

print(f"\nTesting {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} parameter combinations...")
print("This may take 5-10 minutes...\n")

grid_search.fit(X_train, y_train)

# Best parameters
print("\n" + "="*60)
print("BEST PARAMETERS FOUND")
print("="*60)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation MAE: {-grid_search.best_score_:.4f}")

# Test on held-out test set
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

corr_tuned = np.corrcoef(y_test, y_pred_tuned)[0, 1]
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE")
print("="*60)
print(f"Baseline (7 features, default params): 0.7009 correlation, 1.2578 MAE")
print(f"Simplified (4 features, default params): 0.7021 correlation, 1.2556 MAE")
print(f"Tuned (4 features, optimized params): {corr_tuned:.4f} correlation, {mae_tuned:.4f} MAE")

improvement_from_baseline = ((corr_tuned - 0.7009) / 0.7009) * 100
print(f"\nTotal improvement: {improvement_from_baseline:+.2f}%")

# Save results
tuning_results = pd.DataFrame([{
    'Model': 'Baseline (7 features)',
    'Correlation': 0.7009,
    'MAE': 1.2578,
    'RMSE': 1.9047
}, {
    'Model': 'Simplified (4 features)',
    'Correlation': 0.7021,
    'MAE': 1.2556,
    'RMSE': 1.9012
}, {
    'Model': 'Tuned (4 features)',
    'Correlation': corr_tuned,
    'MAE': mae_tuned,
    'RMSE': rmse_tuned
}])

tuning_results.to_csv('final_model_progression.csv', index=False)
print("\n✅ Saved: final_model_progression.csv")

# Feature importance of final model
feature_importance_final = pd.DataFrame({
    'Feature': selected_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("FINAL MODEL FEATURE IMPORTANCE")
print("="*60)
print(feature_importance_final.to_string(index=False))

feature_importance_final.to_csv('feature_importance_final_tuned.csv', index=False)
