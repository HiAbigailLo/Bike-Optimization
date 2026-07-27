# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load hourly demand
df = pd.read_csv('hourly.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# prepare features
feature_columns = ['hour', 'day_of_week', 'day_of_month', 
                   'weekend', 'workingday', 'is_rush_hour']
if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]
    feature_columns.append('station_numeric')

X = df[feature_columns]
y = df['demand']

print("\nFeatures being used:", feature_columns)
print("Target variable: demand")

# split dataset for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# baseline model
print("RANDOM FOREST")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

correlation_rf = np.corrcoef(y_test, y_pred_rf)[0, 1]

print(f"Correlation coefficient:      {correlation_rf:.3f}")
print(f"Mean Absolute Error (MAE):    {mae_rf:.4f} bikes")
print(f"Root Mean Squared Error:      {rmse_rf:.4f} bikes")
print(f"R-squared:                    {r2_rf:.4f}")

# try other models
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'k-NN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'k-NN (k=10)': KNeighborsRegressor(n_neighbors=10),
    'Linear Regression': LinearRegression()
}

results = {
    'Model': ['Random Forest (Baseline)'],
    'Correlation': [correlation_rf],
    'MAE': [mae_rf],
    'RMSE': [rmse_rf],
    'R²': [r2_rf]
}

print("OTHER MODELS")

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    
    # keep results
    results['Model'].append(name)
    results['Correlation'].append(corr)
    results['MAE'].append(mae)
    results['RMSE'].append(rmse)
    results['R²'].append(r2)

    # print results
    print(f"  Correlation: {corr:.3f}")
    print(f"  MAE: {mae:.4f} bikes")
    print(f"  RMSE: {rmse:.4f} bikes")

# comparison table

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Correlation', ascending=False)
print(results_df.to_string(index=False))

results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to: model_comparison_results.csv")

# feature importance
print("FEATURE IMPORTANCE (Random Forest)")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))
feature_importance.to_csv('feature_importance.csv', index=False)

# make some plots

# Plot 1: Actual vs Predicted (Random Forest)
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
plt.savefig('rf_actual_vs_predicted.png', dpi=300)
print("\nSaved: rf_actual_vs_predicted.png")

# Plot 2: Model Comparison
plt.figure(figsize=(10, 6))
results_df.plot(x='Model', y=['Correlation', 'MAE', 'RMSE'], 
                kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.legend(['Correlation', 'MAE (bikes)', 'RMSE (bikes)'])
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("Saved: model_comparison.png")

# Plot 3: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("Saved: feature_importance.png")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nFiles created:")
print("  1. model_comparison_results.csv")
print("  2. feature_importance.csv")
print("  3. rf_actual_vs_predicted.png")
print("  4. model_comparison.png")
print("  5. feature_importance.png")