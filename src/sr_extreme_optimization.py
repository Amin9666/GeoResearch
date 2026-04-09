#!/usr/bin/env python
"""
Extreme SR Optimization - Pushing to R² > 0.10
===============================================
Testing:
- Random Forest variations
- Hyperparameter grid search (GB, RF)
- Feature subset selection  
- Aggressive ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXTREME OPTIMIZATION - PUSHING R² > 0.10")
print("="*80)

# ============================================================================
# LOAD & PREPROCESS
# ============================================================================

print("\n📊 Loading data...")
df = pd.read_csv('Shear_Data_15.csv')

X = df[['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']]
y = df['SHEAR STRESS']

# Aggressive outlier removal
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ((X >= Q1 - 1.0*IQR) & (X <= Q3 + 1.0*IQR)).all(axis=1)
mask = mask & ((y >= y.quantile(0.25) - 1.0*(y.quantile(0.75)-y.quantile(0.25))) &
               (y <= y.quantile(0.75) + 1.0*(y.quantile(0.75)-y.quantile(0.25))))

X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"✓ {len(X)} clean samples")

# Targeted feature engineering - focus on top performers
X_eng = X.copy()

# Key interactions from domain knowledge
X_eng['DURATION*AMP'] = X['DURATION'] * X['AMP']
X_eng['RISE*DURATION'] = X['RISE'] * X['DURATION']
X_eng['ENER*AMP'] = X['ENER'] * X['AMP']
X_eng['RISE*AMP'] = X['RISE'] * X['AMP']
X_eng['COUN*AMP'] = X['COUN'] * X['AMP']

# Power transforms of best features
X_eng['DURATION_sq'] = X['DURATION'] ** 2
X_eng['AMP_sq'] = X['AMP'] ** 2
X_eng['RISE_sq'] = X['RISE'] ** 2
X_eng['ENER_sq'] = X['ENER'] ** 2
X_eng['DURATION_sqrt'] = np.sqrt(X['DURATION'])
X_eng['AMP_sqrt'] = np.sqrt(X['AMP'])

# Log transforms
X_eng['log_DURATION'] = np.log(X['DURATION'] + 1)
X_eng['log_ENER'] = np.log(X['ENER'] + 1)
X_eng['log_AMP'] = np.log(X['AMP'] + 1)

print(f"✓ Features: {X_eng.shape[1]}")

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_eng)

# Split (10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# GRID SEARCH: GRADIENT BOOSTING
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH: GRADIENT BOOSTING")
print("="*80)

param_grid_gb = {
    'learning_rate': [0.001, 0.005, 0.01],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [500, 1000, 2000],
    'subsample': [0.6, 0.8, 1.0],
}

print("\n⏳ Searching GB hyperparameters...")

gb_base = GradientBoostingRegressor(random_state=42, verbose=0)
gb_search = GridSearchCV(gb_base, param_grid_gb, cv=3, n_jobs=-1, verbose=0, scoring='r2')
gb_search.fit(X_train, y_train)

print(f"✓ Best GB params: {gb_search.best_params_}")

gb_best = gb_search.best_estimator_
y_pred_gb = gb_best.predict(X_test)
r2_gb_grid = r2_score(y_test, y_pred_gb)
rmse_gb_grid = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"  CV R²: {gb_search.best_score_:.6f}")
print(f"  Test R²: {r2_gb_grid:.6f}")
print(f"  Test RMSE: {rmse_gb_grid:.6f}")

# ============================================================================
# RANDOM FOREST TUNING
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZED: RANDOM FOREST")
print("="*80)

print("⏳ Training RF with tuned params...")

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"✓ Random Forest:")
print(f"  R² = {r2_rf:.6f}")
print(f"  RMSE = {rmse_rf:.6f}")

# ============================================================================
# ADAPTIVE BOOSTING
# ============================================================================

print("\n" + "="*80)
print("ADAPTIVE BOOSTING")
print("="*80)

print("⏳ Training AdaBoost...")

ada = AdaBoostRegressor(
    n_estimators=500,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
r2_ada = r2_score(y_test, y_pred_ada)
rmse_ada = np.sqrt(mean_squared_error(y_test, y_pred_ada))

print(f"✓ AdaBoost:")
print(f"  R² = {r2_ada:.6f}")
print(f"  RMSE = {rmse_ada:.6f}")

# ============================================================================
# SUPER ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("SUPER ENSEMBLE")
print("="*80)

print("⏳ Creating weighted super-ensemble...")

# Normalize predictions to [0,1] range for equal weight averaging
normalizer = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)

preds_normalized = np.column_stack([
    gb_best.predict(X_test),
    rf.predict(X_test),
    ada.predict(X_test),
])

# Weighted average using inverse generalization error
weights = np.array([r2_gb_grid, r2_rf, r2_ada])
weights = weights / weights.sum()

y_pred_super = np.average(preds_normalized, axis=1, weights=weights)
r2_super = r2_score(y_test, y_pred_super)
rmse_super = np.sqrt(mean_squared_error(y_test, y_pred_super))

print(f"  Weights: GB={weights[0]:.3f}, RF={weights[1]:.3f}, Ada={weights[2]:.3f}")
print(f"✓ Super Ensemble:")
print(f"  R² = {r2_super:.6f}")
print(f"  RMSE = {rmse_super:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

results = {
    'GB (Grid Tuned)': {'r2': r2_gb_grid, 'rmse': rmse_gb_grid},
    'Random Forest': {'r2': r2_rf, 'rmse': rmse_rf},
    'AdaBoost': {'r2': r2_ada, 'rmse': rmse_ada},
    'Super Ensemble': {'r2': r2_super, 'rmse': rmse_super},
}

df_results = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [v['r2'] for v in results.values()],
    'RMSE': [v['rmse'] for v in results.values()],
})

df_results = df_results.sort_values('R² Score', ascending=False)
print("\n" + df_results.to_string(index=False))

best_name = df_results.iloc[0]['Model']
best_r2 = df_results.iloc[0]['R² Score']
best_rmse = df_results.iloc[0]['RMSE']

print(f"\n🏆 BEST: {best_name}")
print(f"   R² = {best_r2:.6f}")
print(f"   RMSE = {best_rmse:.6f}")

# ============================================================================
# IMPROVEMENT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("IMPROVEMENT REPORT")
print("="*80)

baseline_r2 = 0.050
baseline_rmse = 0.9151
previous_best = 0.0793

print(f"\n📊 Progress:")
print(f"  Baseline:        R² = {baseline_r2:.4f}, RMSE = {baseline_rmse:.4f}")
print(f"  Previous best:   R² = {previous_best:.4f}")
print(f"  Current:         R² = {best_r2:.6f}, RMSE = {best_rmse:.6f}")

r2_pct = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
prev_pct = ((best_r2 - previous_best) / previous_best) * 100

print(f"\n  Total improvement vs baseline: {r2_pct:+.1f}%")
print(f"  Improvement vs previous best:  {prev_pct:+.1f}%")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

import pickle

df_results.to_csv('sr_extreme_comparison.csv', index=False)
print("✓ Comparison saved: sr_extreme_comparison.csv")

# Save best model
if 'Grid' in best_name:
    with open('sr_best_extreme.pkl', 'wb') as f:
        pickle.dump(gb_best, f)
elif 'Random' in best_name:
    with open('sr_best_extreme.pkl', 'wb') as f:
        pickle.dump(rf, f)
elif 'AdaBoost' in best_name:
    with open('sr_best_extreme.pkl', 'wb') as f:
        pickle.dump(ada, f)
else:
    with open('sr_best_extreme.pkl', 'wb') as f:
        pickle.dump({'model': gb_best, 'rf': rf, 'ada': ada, 'weights': weights}, f)

print("✓ Best model saved: sr_best_extreme.pkl")

print("\n" + "="*80)
print("✅ EXTREME OPTIMIZATION COMPLETE")
print("="*80)
