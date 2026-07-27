import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv('hourly.csv')
selected_features = ['hour', 'day_of_week', 'day_of_month', 'station_numeric']

if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]

X = df[selected_features]
y = df['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Simplified model with default params (for side-by-side comparison)
rf_simplified = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_simplified.fit(X_train, y_train)
y_pred_simplified = rf_simplified.predict(X_test)

corr_simplified = np.corrcoef(y_test, y_pred_simplified)[0, 1]
mae_simplified = mean_absolute_error(y_test, y_pred_simplified)
rmse_simplified = np.sqrt(mean_squared_error(y_test, y_pred_simplified))
r2_simplified = r2_score(y_test, y_pred_simplified)

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
grid_search.fit(X_train, y_train)

# Best parameters
print("best params")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation MAE: {-grid_search.best_score_:.4f}")

with open('best_params.txt', 'w') as f:
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
    f.write(f"best_cv_mae: {-grid_search.best_score_:.4f}\n")

# Test on held-out test set
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

corr_tuned = np.corrcoef(y_test, y_pred_tuned)[0, 1]
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2_tuned = r2_score(y_test, y_pred_tuned)

print("final model")
print(f"Baseline (7 features, default params): 0.7009 correlation, 1.2578 MAE, R²=0.4805")
print(f"Simplified (4 features, default params): {corr_simplified:.4f} correlation, {mae_simplified:.4f} MAE, R²={r2_simplified:.4f}")
print(f"Tuned (4 features, optimized params): {corr_tuned:.4f} correlation, {mae_tuned:.4f} MAE, R²={r2_tuned:.4f}")

improvement_from_baseline = ((corr_tuned - 0.7009) / 0.7009) * 100
improvement_from_simplified = ((corr_tuned - corr_simplified) / corr_simplified) * 100
print(f"\nTotal improvement (tuned vs. baseline): {improvement_from_baseline:+.2f}%")
print(f"Improvement from tuning alone (tuned vs. simplified, default params): {improvement_from_simplified:+.2f}%")

# Save results
tuning_results = pd.DataFrame([{
    'Model': 'Baseline (7 features)',
    'Correlation': 0.7009,
    'MAE': 1.2578,
    'RMSE': 1.9047,
    'R2': 0.4805
}, {
    'Model': 'Simplified (4 features)',
    'Correlation': corr_simplified,
    'MAE': mae_simplified,
    'RMSE': rmse_simplified,
    'R2': r2_simplified
}, {
    'Model': 'Tuned (4 features)',
    'Correlation': corr_tuned,
    'MAE': mae_tuned,
    'RMSE': rmse_tuned,
    'R2': r2_tuned
}])

tuning_results.to_csv('final_model_progression.csv', index=False)
print("\nSaved: final_model_progression.csv")

# Feature importance of final model
feature_importance_final = pd.DataFrame({
    'Feature': selected_features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("feature importance")
print(feature_importance_final.to_string(index=False))
feature_importance_final.to_csv('feature_importance_final_tuned.csv', index=False)