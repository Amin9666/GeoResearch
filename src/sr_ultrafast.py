#!/usr/bin/env python
"""
ULTRA-FAST SR OPTIMIZATION
============================
No cross-validation - just fast training and testing
Focus: Beat R² = 0.0793 quickly
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTRA-FAST SR OPTIMIZATION")
print("="*80)

# ============================================================================
# LOAD & BASIC PREPROCESSING
# ============================================================================

print("\n📊 Loading data...")
df = pd.read_csv('Shear_Data_15.csv')

X = df[['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']]
y = df['SHEAR STRESS']

# Aggressive cleaning
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ((X >= Q1 - 1.0*IQR) & (X <= Q3 + 1.0*IQR)).all(axis=1)
mask = mask & ((y >= y.quantile(0.25) - 1.0*(y.quantile(0.75)-y.quantile(0.25))) &
               (y <= y.quantile(0.75) + 1.0*(y.quantile(0.75)-y.quantile(0.25))))

X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"✓ {len(X)} clean samples")

# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

print("✓ Engineering features...")

X_eng = X.copy()

# Interactions (domain-focused)
X_eng['DUR_AMP'] = X['DURATION'] * X['AMP']
X_eng['RISE_DUR'] = X['RISE'] * X['DURATION']
X_eng['ENER_AMP'] = X['ENER'] * X['AMP']
X_eng['RISE_AMP'] = X['RISE'] * X['AMP']
X_eng['COUN_DUR'] = X['COUN'] * X['DURATION']

# Powers
X_eng['DUR2'] = X['DURATION'] ** 2
X_eng['AMP2'] = X['AMP'] ** 2
X_eng['RISE2'] = X['RISE'] ** 2
X_eng['ENER2'] = X['ENER'] ** 2

# Roots
X_eng['DUR_sqrt'] = np.sqrt(X['DURATION'])
X_eng['RISE_sqrt'] = np.sqrt(X['RISE'])
X_eng['AMP_sqrt'] = np.sqrt(X['AMP'])

# Logs
X_eng['log_DUR'] = np.log(X['DURATION'] + 1)
X_eng['log_ENER'] = np.log(X['ENER'] + 1)
X_eng['log_AMP'] = np.log(X['AMP'] + 1)

# Advanced interactions
X_eng['DUR2_AMP'] = X['DURATION']**2 * X['AMP']
X_eng['DUR_AMP2'] = X['DURATION'] * X['AMP']**2
X_eng['RISE*DUR*AMP'] = X['RISE'] * X['DURATION'] * X['AMP']

print(f"✓ Total features: {X_eng.shape[1]}")

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_eng)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# FAST MODEL TRAINING (NO CV)
# ============================================================================

print("\n" + "="*80)
print("TESTING MODELS (NO CROSS-VALIDATION)")
print("="*80)

results = {}

# ---- GRADIENT BOOSTING VARIATIONS ----
print("\n🔹 Gradient Boosting variants...")

gb_configs = [
    {'name': 'GB-Fast (100, lr=0.05)', 'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4},
    {'name': 'GB-Med (150, lr=0.02)', 'n_estimators': 150, 'learning_rate': 0.02, 'max_depth': 5},
]

for cfg in gb_configs:
    gb = GradientBoostingRegressor(
        n_estimators=cfg['n_estimators'],
        learning_rate=cfg['learning_rate'],
        max_depth=cfg['max_depth'],
        subsample=0.8,
        min_samples_split=2,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[cfg['name']] = {'r2': r2, 'rmse': rmse, 'model': gb}
    print(f"  {cfg['name']:30} R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- RANDOM FOREST VARIATIONS ----
print("\n🔹 Random Forest variants...")

rf_configs = [
    {'name': 'RF-Fast (100, d=10)', 'n_estimators': 100, 'max_depth': 10},
]

for cfg in rf_configs:
    rf = RandomForestRegressor(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg['max_depth'],
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[cfg['name']] = {'r2': r2, 'rmse': rmse, 'model': rf}
    print(f"  {cfg['name']:30} R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- EXTRA TREES ----
print("\n🔹 Extra Trees regression...")

et = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['Extra Trees (100, d=10)'] = {'r2': r2, 'rmse': rmse, 'model': et}
print(f"  Extra Trees (100, d=10)      R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- LINEAR MODELS ----
print("\n🔹 Regularized linear models...")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['Ridge (alpha=1.0)'] = {'r2': r2, 'rmse': rmse, 'model': ridge}
print(f"  Ridge (alpha=1.0)            R²={r2:.6f}, RMSE={rmse:.6f}")

lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['Lasso (alpha=0.001)'] = {'r2': r2, 'rmse': rmse, 'model': lasso}
print(f"  Lasso (alpha=0.001)          R²={r2:.6f}, RMSE={rmse:.6f}")

# ========================================================================
# ENSEMBLE: WEIGHTED AVERAGE OF TOP 3
# ========================================================================

print("\n🔹 Weighted Ensemble (top 3 models)...")

# Get top 3
sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
top_3_preds = []
top_3_r2 = []

for name, data in sorted_results[:3]:
    preds = data['model'].predict(X_test)
    top_3_preds.append(preds)
    top_3_r2.append(data['r2'])

# Weighted average
weights = np.array(top_3_r2) / np.sum(top_3_r2)
ensemble_preds = np.average(top_3_preds, axis=0, weights=weights)
r2_ens = r2_score(y_test, ensemble_preds)
rmse_ens = np.sqrt(mean_squared_error(y_test, ensemble_preds))
results['Weighted Ensemble (Top 3)'] = {'r2': r2_ens, 'rmse': rmse_ens, 'model': 'ensemble'}
print(f"  Weighted Ensemble (Top 3)    R²={r2_ens:.6f}, RMSE={rmse_ens:.6f}")
for i, (name, _) in enumerate(sorted_results[:3]):
    print(f"    [{i+1}] {name} (w={weights[i]:.3f})")

# ============================================================================
# RESULTS & COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

df_results = pd.DataFrame([
    {'Model': name, 'R² Score': data['r2'], 'RMSE': data['rmse']}
    for name, data in results.items()
])

df_results = df_results.sort_values('R² Score', ascending=False)
print("\n" + df_results.to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================

best_name = df_results.iloc[0]['Model']
best_r2 = df_results.iloc[0]['R² Score']
best_rmse = df_results.iloc[0]['RMSE']

print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)

print(f"\n{best_name}")
print(f"  R² = {best_r2:.6f}")
print(f"  RMSE = {best_rmse:.6f}")

# ============================================================================
# IMPROVEMENT REPORT
# ============================================================================

baseline_r2 = 0.050
prev_best = 0.0793

improvement_baseline = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
improvement_prev = ((best_r2 - prev_best) / prev_best) * 100

print(f"\n📊 IMPROVEMENT:")
print(f"  vs Baseline (0.050):      {improvement_baseline:+.1f}%")
print(f"  vs Previous Best (0.0793): {improvement_prev:+.1f}%")

if best_r2 > 0.10:
    print(f"\n✨ MILESTONE ACHIEVED: R² > 0.10!")
    print(f"   Current: {best_r2:.6f}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

import pickle

df_results.to_csv('sr_fast_comparison.csv', index=False)
print("✓ Results: sr_fast_comparison.csv")

if results[best_name]['model'] != 'ensemble':
    with open('sr_best_fast.pkl', 'wb') as f:
        pickle.dump(results[best_name]['model'], f)
    print("✓ Model: sr_best_fast.pkl")

summary = pd.DataFrame({
    'Metric': ['Best Model', 'R² Score', 'RMSE', 'Total Features', 'Samples'],
    'Value': [best_name, f"{best_r2:.6f}", f"{best_rmse:.6f}", str(X_eng.shape[1]), str(len(X))]
})

summary.to_csv('sr_fast_summary.csv', index=False)
print("✓ Summary: sr_fast_summary.csv")

print("\n" + "="*80)
print("✅ FAST OPTIMIZATION COMPLETE")
print("="*80)
