import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# prepare data

df = pd.read_csv('hourly.csv')

feature_columns_full = ['hour', 'day_of_week', 'day_of_month', 
                        'weekend', 'workingday', 'is_rush_hour']

if 'start_station_id' in df.columns:
    df['station_numeric'] = pd.factorize(df['start_station_id'])[0]
    feature_columns_full.append('station_numeric')

X = df[feature_columns_full]
y = df['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# baseline: use all features
print("All 7 Features")
rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)

corr_baseline = np.corrcoef(y_test, y_pred_baseline)[0, 1]
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

print(f"Correlation: {corr_baseline:.4f}")
print(f"MAE: {mae_baseline:.4f}")
print(f"RMSE: {rmse_baseline:.4f}")

# Feature importance
feature_importance_full = pd.DataFrame({
    'Feature': feature_columns_full,
    'Importance': rf_baseline.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_full.to_string(index=False))

# find irrelevant features (<5%)
low_importance = feature_importance_full[
    feature_importance_full['Importance'] < 0.05
]
print("\nLow importance features (< 5%):")
print(low_importance.to_string(index=False))

feature_corr = df[feature_columns_full].corr()
print("\nHighly correlated features (>0.7):")
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if abs(feature_corr.iloc[i, j]) > 0.7:
            print(f"  {feature_corr.columns[i]} <-> {feature_corr.columns[j]}: {feature_corr.iloc[i, j]:.3f}")

# test subsets
print("TESTING FEATURE SUBSETS")
feature_sets = {
    'Full (7 features)': feature_columns_full,
    
    # Remove low-importance features
    'Remove is_rush_hour (6)': [f for f in feature_columns_full if f != 'is_rush_hour'],
    'Remove weekend (6)': [f for f in feature_columns_full if f != 'weekend'],
    'Remove workingday (6)': [f for f in feature_columns_full if f != 'workingday'],
    
    # Remove multiple redundant features
    'Remove weekend + workingday (5)': [f for f in feature_columns_full 
                                        if f not in ['weekend', 'workingday']],
    'Remove all binary indicators (4)': [f for f in feature_columns_full 
                                         if f not in ['weekend', 'workingday', 'is_rush_hour']],
    
    # top three features
    'Top 3 only': feature_importance_full.head(3)['Feature'].tolist(),
    
    # Minimal set
    'Minimal (station + hour)': ['station_numeric', 'hour']
}

results = []

for name, features in feature_sets.items():
    print(f"\nTesting: {name}")
    print(f"Features: {features}")
    
    # Train model
    X_subset = df[features]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_subset, y, test_size=0.30, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train_s)
    y_pred = rf.predict(X_test_s)
    
    corr = np.corrcoef(y_test_s, y_pred)[0, 1]
    mae = mean_absolute_error(y_test_s, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_s, y_pred))
    
    corr_pct = ((corr - corr_baseline) / corr_baseline) * 100
    mae_pct = ((mae - mae_baseline) / mae_baseline) * 100
    
    results.append({
        'Feature_Set': name,
        'Num_Features': len(features),
        'Correlation': corr,
        'MAE': mae,
        'RMSE': rmse,
        'Corr_vs_Baseline_%': corr_pct,
        'MAE_vs_Baseline_%': mae_pct
    })
    
    print(f"  Correlation: {corr:.4f} ({corr_pct:+.1f}%)")
    print(f"  MAE: {mae:.4f} ({mae_pct:+.1f}%)")

# compare all results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Correlation', ascending=False)
print("FEATURE SELECTION RESULTS (sorted by correlation)")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('feature_selection_results.csv', index=False)

# Find simplest model with <2% performance loss
acceptable_loss = 0.98  # 98% of baseline performance

candidates = results_df[
    results_df['Correlation'] >= corr_baseline * acceptable_loss
].sort_values('Num_Features')

if len(candidates) > 0:
    recommended = candidates.iloc[0]
    
    print(f"\n Best feature set:")
    print(f"   {recommended['Feature_Set']}")
    print(f"   Features: {recommended['Num_Features']}")
    print(f"   Correlation: {recommended['Correlation']:.4f} ({recommended['Corr_vs_Baseline_%']:+.1f}%)")
    print(f"   MAE: {recommended['MAE']:.4f} ({recommended['MAE_vs_Baseline_%']:+.1f}%)")
    print(f"\n   Performance loss: {(1 - recommended['Correlation']/corr_baseline)*100:.1f}%")
    print(f"   Simplification: {7 - recommended['Num_Features']} fewer features")
    
    # Get the actual feature list
    recommended_features = feature_sets[recommended['Feature_Set']]
    print(f"\n   Final features: {recommended_features}")
    
else:
    print("\n All simplified models lose >2% performance")
    recommended_features = feature_columns_full


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Plot 1: Correlation vs Number of Features
axes[0].scatter(results_df['Num_Features'], results_df['Correlation'], s=100)
axes[0].axhline(y=corr_baseline, color='red', linestyle='--', 
                label='Baseline (all features)')
axes[0].axhline(y=corr_baseline*0.98, color='orange', linestyle='--', 
                label='98% of baseline')
axes[0].set_xlabel('Number of Features')
axes[0].set_ylabel('Correlation')
axes[0].set_title('Feature Selection: Correlation vs Complexity')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Annotate points
for idx, row in results_df.iterrows():
    axes[0].annotate(row['Num_Features'], 
                     (row['Num_Features'], row['Correlation']),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

# Plot 2: MAE vs Number of Features
axes[1].scatter(results_df['Num_Features'], results_df['MAE'], s=100, color='coral')
axes[1].axhline(y=mae_baseline, color='red', linestyle='--',
                label='Baseline (all features)')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('MAE (bikes)')
axes[1].set_title('Feature Selection: MAE vs Complexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=300)
print("\n Saved: feature_selection_analysis.png")

# keep best model
X_final = df[recommended_features]
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y, test_size=0.30, random_state=42
)

rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_final.fit(X_train_final, y_train_final)
y_pred_final = rf_final.predict(X_test_final)

corr_final = np.corrcoef(y_test_final, y_pred_final)[0, 1]
mae_final = mean_absolute_error(y_test_final, y_pred_final)

print(f"Final model performance:")
print(f"  Correlation: {corr_final:.4f}")
print(f"  MAE: {mae_final:.4f}")
print(f"  Features used: {len(recommended_features)}")

feature_importance_final = pd.DataFrame({
    'Feature': recommended_features,
    'Importance': rf_final.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFinal feature importance:")
print(feature_importance_final.to_string(index=False))

feature_importance_final.to_csv('feature_importance_simplified.csv', index=False)

print(f"   Recommended features saved for parameter tuning: {recommended_features}")

#save terminal
with open('selected_features.txt', 'w') as f:
    f.write(','.join(recommended_features))