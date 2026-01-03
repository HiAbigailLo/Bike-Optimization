# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time

# ==========================================
# STEP 1: Load data
# ==========================================
df = pd.read_csv('hourly.csv')

# Prepare features
feature_columns = ['hour', 'day_of_week', 'day_of_month', 
                   'weekend', 'workingday', 'is_rush_hour']

if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]
    feature_columns.append('station_numeric')

X = df[feature_columns]
y = df['demand']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("="*60)
print("COMPARING MULTIPLE ML MODELS")
print("="*60)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Features: {feature_columns}")

# ==========================================
# STEP 2: Define models to compare
# ==========================================

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    ),
    'Decision Tree': DecisionTreeRegressor(
        random_state=42
    ),
    'k-NN (k=5)': KNeighborsRegressor(
        n_neighbors=5
    ),
    'k-NN (k=10)': KNeighborsRegressor(
        n_neighbors=10
    ),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

# ==========================================
# STEP 3: Train and evaluate each model
# ==========================================

results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)
    
    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Time the prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Calculate metrics
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Correlation': correlation,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Train_Time_sec': train_time,
        'Predict_Time_sec': predict_time
    })
    
    # Print results
    print(f"Correlation:    {correlation:.4f}")
    print(f"MAE:            {mae:.4f} bikes")
    print(f"RMSE:           {rmse:.4f} bikes")
    print(f"R²:             {r2:.4f}")
    print(f"Train time:     {train_time:.2f} seconds")
    print(f"Predict time:   {predict_time:.2f} seconds")

# ==========================================
# STEP 4: Create comparison table
# ==========================================

results_df = pd.DataFrame(results)

# Sort by correlation (primary) and MAE (secondary)
results_df = results_df.sort_values(['Correlation', 'MAE'], 
                                     ascending=[False, True])

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY (sorted by Correlation)")
print("="*60)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_comparison_full.csv', index=False)
print("\n✅ Results saved to: model_comparison_full.csv")

# ==========================================
# STEP 5: Identify top 2 models
# ==========================================

print("\n" + "="*60)
print("TOP 2 MODELS FOR PARAMETER TUNING")
print("="*60)

top_2 = results_df.head(2)
print(top_2[['Model', 'Correlation', 'MAE', 'RMSE']].to_string(index=False))

print("\n📌 RECOMMENDATION:")
print(f"   Proceed with parameter tuning for:")
print(f"   1. {top_2.iloc[0]['Model']}")
print(f"   2. {top_2.iloc[1]['Model']}")

# ==========================================
# STEP 6: Visualizations
# ==========================================

# Plot 1: Model comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Correlation
axes[0, 0].bar(results_df['Model'], results_df['Correlation'], color='steelblue')
axes[0, 0].set_ylabel('Correlation')
axes[0, 0].set_title('Model Comparison: Correlation Coefficient')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].axhline(y=0.7, color='red', linestyle='--', label='Literature target (0.7)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE
axes[0, 1].bar(results_df['Model'], results_df['MAE'], color='coral')
axes[0, 1].set_ylabel('MAE (bikes)')
axes[0, 1].set_title('Model Comparison: Mean Absolute Error')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# RMSE
axes[1, 0].bar(results_df['Model'], results_df['RMSE'], color='lightgreen')
axes[1, 0].set_ylabel('RMSE (bikes)')
axes[1, 0].set_title('Model Comparison: Root Mean Squared Error')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# R²
axes[1, 1].bar(results_df['Model'], results_df['R²'], color='gold')
axes[1, 1].set_ylabel('R²')
axes[1, 1].set_title('Model Comparison: R-squared')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_full.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: model_comparison_full.png")

# Plot 2: Correlation vs MAE scatter
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Correlation'], results_df['MAE'], s=100)

for idx, row in results_df.iterrows():
    plt.annotate(row['Model'], 
                (row['Correlation'], row['MAE']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Mean Absolute Error (bikes)')
plt.title('Model Performance: Correlation vs MAE\n(Top-right corner = Best)')
plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Target correlation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('correlation_vs_mae.png', dpi=300)
print("✅ Saved: correlation_vs_mae.png")

# Plot 3: Training time comparison
plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['Train_Time_sec'], color='purple', alpha=0.6)
plt.xlabel('Training Time (seconds)')
plt.title('Model Training Time Comparison')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300)
print("✅ Saved: training_time_comparison.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nFiles created:")
print("  1. model_comparison_full.csv")
print("  2. model_comparison_full.png")
print("  3. correlation_vs_mae.png")
print("  4. training_time_comparison.png")
print("\n📋 Next step: Parameter tuning for top 2 models")