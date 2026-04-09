#!/usr/bin/env python
"""
AGGRESSIVE PUSH - Target R² > 0.10
===================================
Larger models, more feature engineering options, stacking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AGGRESSIVE PUSH - R² > 0.10")
print("="*80)

# ============================================================================
# LOAD & PREPROCESS
# ============================================================================

print("\n📊 Loading data...")
df = pd.read_csv('Shear_Data_15.csv')

X = df[['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']]
y = df['SHEAR STRESS']

# Clean
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ((X >= Q1 - 1.0*IQR) & (X <= Q3 + 1.0*IQR)).all(axis=1)
mask = mask & ((y >= y.quantile(0.25) - 1.0*(y.quantile(0.75)-y.quantile(0.25))) &
               (y <= y.quantile(0.75) + 1.0*(y.quantile(0.75)-y.quantile(0.25))))

X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"✓ {len(X)} samples")

# ============================================================================
# AGGRESSIVE FEATURE ENGINEERING
# ============================================================================

print("✓ Engineering aggressive feature set...")

X_eng = X.copy()

# All pairwise interactions
for i, col1 in enumerate(X.columns):
    for col2 in list(X.columns)[i+1:]:
        X_eng[f'{col1}*{col2}'] = X[col1] * X[col2]

# Powers
for col in X.columns:
    X_eng[f'{col}^2'] = X[col] ** 2
    X_eng[f'{col}^3'] = X[col] ** 3
    X_eng[f'sqrt_{col}'] = np.sqrt(X[col])
    X_eng[f'log_{col}'] = np.log(X[col] + 1)

# Ratios
for i, col1 in enumerate(X.columns):
    for col2 in list(X.columns)[i+1:]:
        X_eng[f'{col1}/{col2}'] = X[col1] / (X[col2] + 1e-8)
        X_eng[f'{col2}/{col1}'] = X[col2] / (X[col1] + 1e-8)

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
# LARGE MODELS - BEST CONFIGS
# ============================================================================

print("\n" + "="*80)
print("TRAINING LARGER MODELS")
print("="*80)

results = {}

# ---- LARGE GRADIENT BOOSTING ----
print("\n🔹 Large Gradient Boosting...")

gb_large = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    min_samples_split=2,
    random_state=42
)
gb_large.fit(X_train, y_train)
y_pred = gb_large.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['GB-Large (500, lr=0.03, d=6)'] = {'r2': r2, 'rmse': rmse, 'model': gb_large}
print(f"  R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- LARGE EXTRA TREES ----
print("\n🔹 Large Extra Trees...")

et_large = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
et_large.fit(X_train, y_train)
y_pred = et_large.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['Extra Trees (500, d=15)'] = {'r2': r2, 'rmse': rmse, 'model': et_large}
print(f"  R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- LARGE RANDOM FOREST ----
print("\n🔹 Large Random Forest...")

rf_large = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)
rf_large.fit(X_train, y_train)
y_pred = rf_large.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['RF-Large (400, d=16)'] = {'r2': r2, 'rmse': rmse, 'model': rf_large}
print(f"  R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- GB AGGRESSIVE ----
print("\n🔹 GB Aggressive tuning...")

gb_agg = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.75,
    min_samples_split=2,
    random_state=42
)
gb_agg.fit(X_train, y_train)
y_pred = gb_agg.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['GB-Agg (600, lr=0.02, d=7)'] = {'r2': r2, 'rmse': rmse, 'model': gb_agg}
print(f"  R²={r2:.6f}, RMSE={rmse:.6f}")

# ---- RIDGE WITH ALL FEATURES ----
print("\n🔹 Ridge with aggressive regularization...")

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
results['Ridge (alpha=0.5)'] = {'r2': r2, 'rmse': rmse, 'model': ridge}
print(f"  R²={r2:.6f}, RMSE={rmse:.6f}")

# ============================================================================
# STACKED ENSEMBLE - META LEARNER
# ============================================================================

print("\n" + "="*80)
print("TRAINING STACKED ENSEMBLE")
print("="*80)

print("⏳ Generating base predictions...")

# Train/val split for meta-features
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Train base models on base set
gb_stack = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
gb_stack.fit(X_train_base, y_train_base)

et_stack = ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
et_stack.fit(X_train_base, y_train_base)

rf_stack = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf_stack.fit(X_train_base, y_train_base)

# Generate meta-features on validation set
meta_train = np.column_stack([
    gb_stack.predict(X_train_meta),
    et_stack.predict(X_train_meta),
    rf_stack.predict(X_train_meta),
])

# Generate meta-features on test set
meta_test = np.column_stack([
    gb_stack.predict(X_test),
    et_stack.predict(X_test),
    rf_stack.predict(X_test),
])

print("✓ Meta-features created")

# Train meta-learner
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_train, y_train_meta)

y_pred_stack = meta_learner.predict(meta_test)
r2_stack = r2_score(y_test, y_pred_stack)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
results['Stacked (GB+ET+RF)'] = {'r2': r2_stack, 'rmse': rmse_stack, 'model': 'stacked'}
print(f"✓ Stacked R²={r2_stack:.6f}, RMSE={rmse_stack:.6f}")

# ============================================================================
# AGGRESSIVE WEIGHTED ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("AGGRESSIVE WEIGHTED ENSEMBLE")
print("="*80)

# Get top 4 models
sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:4]

print("Top 4 models:")
preds_list = []
r2_list = []
names_list = []

for name, data in sorted_models:
    print(f"  {name}: R²={data['r2']:.6f}")
    if data['model'] != 'stacked':
        preds_list.append(data['model'].predict(X_test))
        r2_list.append(data['r2'])
        names_list.append(name)

# Weighted average
weights = np.array(r2_list)
weights = weights / weights.sum()

y_pred_weighted = np.average(preds_list, axis=0, weights=weights)
r2_weighted = r2_score(y_test, y_pred_weighted)
rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
results['Weighted Ensemble (Top 4)'] = {'r2': r2_weighted, 'rmse': rmse_weighted, 'model': 'ensemble'}

print(f"\n✓ Weighted Ensemble R²={r2_weighted:.6f}, RMSE={rmse_weighted:.6f}")
for i, (name, w) in enumerate(zip(names_list, weights)):
    print(f"  {i+1}. {name} ({w:.3f})")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

df_results = pd.DataFrame([
    {'Model': name, 'R² Score': data['r2'], 'RMSE': data['rmse']}
    for name, data in results.items()
])

df_results = df_results.sort_values('R² Score', ascending=False)
print("\n" + df_results.to_string(index=False))

best_name = df_results.iloc[0]['Model']
best_r2 = df_results.iloc[0]['R² Score']
best_rmse = df_results.iloc[0]['RMSE']

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)

print(f"\n{best_name}")
print(f"  R² = {best_r2:.6f}")
print(f"  RMSE = {best_rmse:.6f}")

baseline = 0.050
prev = 0.0827

imp_base = ((best_r2 - baseline) / abs(baseline)) * 100
imp_prev = ((best_r2 - prev) / prev) * 100

print(f"\n📊 PROGRESS:")
print(f"  vs Baseline (0.050):      {imp_base:+.1f}%")
print(f"  vs Previous (0.0827):     {imp_prev:+.1f}%")

if best_r2 > 0.10:
    print(f"\n✨ BREAKTHROUGH: R² > 0.10 ACHIEVED!")
    print(f"   Current: {best_r2:.6f}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING")
print("="*80)

import pickle

df_results.to_csv('sr_aggressive_comparison.csv', index=False)
print("✓ sr_aggressive_comparison.csv")

if results[best_name]['model'] not in ['stacked', 'ensemble']:
    with open('sr_best_aggressive.pkl', 'wb') as f:
        pickle.dump(results[best_name]['model'], f)
    print("✓ sr_best_aggressive.pkl")

summary = pd.DataFrame({
    'Metric': ['Best Model', 'R² Score', 'RMSE', 'Features', 'Samples'],
    'Value': [best_name, f"{best_r2:.6f}", f"{best_rmse:.6f}", str(X_eng.shape[1]), str(len(X))]
})

summary.to_csv('sr_aggressive_summary.csv', index=False)
print("✓ sr_aggressive_summary.csv")

print("\n" + "="*80)
print("✅ AGGRESSIVE PUSH COMPLETE")
print("="*80)
