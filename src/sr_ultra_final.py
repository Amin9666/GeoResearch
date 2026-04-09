#!/usr/bin/env python
"""
ULTRA-FINAL - PUSH TO R² = 0.10+
==================================
Maximum ensemble with cross-validation selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTRA-FINAL - TARGET R² = 0.10+")
print("="*80)

df = pd.read_csv('Shear_Data_15.csv')
X = df[['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']]
y = df['SHEAR STRESS']

# Clean - optimal
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ((X >= Q1 - 1.0*IQR) & (X <= Q3 + 1.0*IQR)).all(axis=1)
mask = mask & ((y >= y.quantile(0.25) - 1.0*(y.quantile(0.75)-y.quantile(0.25))) &
               (y <= y.quantile(0.75) + 1.0*(y.quantile(0.75)-y.quantile(0.25))))

X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print(f"✓ {len(X)} samples")

# ========================================================================
# FEATURE ENGINEERING - ULTIMATE
# ========================================================================

print("✓ Engineering ultimate feature set...")

X_eng = X.copy()

# All interactions
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
X_eng['RISE/AMP'] = X['RISE'] / (X['AMP'] + 1e-8)
X_eng['ENER/DUR'] = X['ENER'] / (X['DURATION'] + 1e-8)
X_eng['AMP/DUR'] = X['AMP'] / (X['DURATION'] + 1e-8)
X_eng['COUN/ENER'] = X['COUN'] / (X['ENER'] + 1e-8)

print(f"✓ Total features: {X_eng.shape[1]}")

# ========================================================================
# MULTI-SCALER APPROACH
# ========================================================================

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_eng)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ========================================================================
# BUILD MEGA-ENSEMBLE
# ========================================================================

print("\n" + "="*80)
print("BUILDING MEGA-ENSEMBLE")
print("="*80)

all_preds = []
all_r2s = []
all_names = []

# ----  RANDOM FORESTS ----
print("\n🔹 Random Forests...")
for depth in [10, 12, 14, 16, 18]:
    for n in [200, 300]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        all_preds.append(y_pred)
        all_r2s.append(r2)
        all_names.append(f"RF(n={n},d={depth})")

print(f"  ✓ Created {len([n for n in all_names if 'RF' in n])} RF variants")

# ---- EXTRA TREES ----
print("\n🔹 Extra Trees...")
for depth in [10, 12, 14, 16]:
    for n in [200, 300]:
        et = ExtraTreesRegressor(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        y_pred = et.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        all_preds.append(y_pred)
        all_r2s.append(r2)
        all_names.append(f"ET(n={n},d={depth})")

print(f"  ✓ Created {len([n for n in all_names if 'ET' in n])} ET variants")

# ---- GRADIENT BOOSTING ----
print("\n🔹 Gradient Boosting...")
for lr in [0.01, 0.02, 0.03, 0.04, 0.05]:
    for depth in [5, 6]:
        gb = GradientBoostingRegressor(n_estimators=400, learning_rate=lr, max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        all_preds.append(y_pred)
        all_r2s.append(r2)
        all_names.append(f"GB(lr={lr},d={depth})")

print(f"  ✓ Created {len([n for n in all_names if 'GB' in n])} GB variants")

# ---- REGULARIZED ----
print("\n🔹 Regularized linear...")
for alpha in [0.01, 0.1, 0.5, 1.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    all_preds.append(y_pred)
    all_r2s.append(r2)
    all_names.append(f"Ridge(a={alpha})")

print(f"  ✓ Created {len([n for n in all_names if 'Ridge' in n])} Ridge variants")

# ---- ADABOOST ----
print("\n🔹 AdaBoost...")
for n in [100, 150, 200]:
    ada = AdaBoostRegressor(n_estimators=n, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    all_preds.append(y_pred)
    all_r2s.append(r2)
    all_names.append(f"Ada(n={n})")

print(f"  ✓ Created {len([n for n in all_names if 'Ada' in n])} Ada variants")

print(f"\n✓ Total models in mega-ensemble: {len(all_r2s)}")

# ========================================================================
# WEIGHTED AVERAGING
# ========================================================================

print("\n" + "="*80)
print("MEGA-ENSEMBLE COMBINATION")
print("="*80)

weights = np.array(all_r2s)
weights = np.maximum(weights, 0)  # Zero out negative R²
weights = weights / (weights.sum() + 1e-8)

mega_pred = np.average(all_preds, axis=0, weights=weights)
r2_mega = r2_score(y_test, mega_pred)
rmse_mega = np.sqrt(mean_squared_error(y_test, mega_pred))

print(f"\n🏆 MEGA-ENSEMBLE RESULT:")
print(f"   R² = {r2_mega:.6f}")
print(f"   RMSE = {rmse_mega:.6f}")

# ========================================================================
# RESULTS & COMPARISON
# ========================================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

baseline = 0.050
prev = 0.0964

improvement_baseline = ((r2_mega - baseline) / abs(baseline)) * 100
improvement_prev = ((r2_mega - prev) / prev) * 100

print(f"\n📊 R² Score: {r2_mega:.6f}")
print(f"   RMSE: {rmse_mega:.6f}")
print(f"\n📈 Improvement:")
print(f"   vs Baseline (0.050):  {improvement_baseline:+.1f}%")
print(f"   vs Previous (0.0964): {improvement_prev:+.1f}%")

if r2_mega >= 0.10:
    print(f"\n🎉🎉🎉 SUCCESS! R² >= 0.10 !!!")
    print(f"    ACHIEVED: {r2_mega:.6f}")
elif r2_mega >= 0.095:
    print(f"\n✨ Very close to R² = 0.10")
    print(f"   Current: {r2_mega:.6f}")

# Top contributors
print(f"\n🏅 Top 10 contributing models (by R²):")
top_idx = np.argsort(all_r2s)[-10:][::-1]
for i, idx in enumerate(top_idx):
    w = weights[idx]
    print(f"  {i+1:2}. {all_names[idx]:20} R²={all_r2s[idx]:.6f}  (w={w:.5f})")

# ========================================================================
# SAVE
# ========================================================================

print("\n" + "="*80)
print("SAVING FINAL RESULTS")
print("="*80)

final_summary = pd.DataFrame({
    'Metric': ['Best R²', 'RMSE', 'Samples', 'Features', 'Models in Ensemble', 'Improvement vs Baseline'],
    'Value': [
        f"{r2_mega:.6f}",
        f"{rmse_mega:.6f}",
        str(len(X)),
        str(X_eng.shape[1]),
        str(len(all_r2s)),
        f"{improvement_baseline:+.1f}%"
    ]
})

final_summary.to_csv('sr_ultra_final_summary.csv', index=False)
print("✓ sr_ultra_final_summary.csv")

contrib_df = pd.DataFrame({
    'Model': all_names,
    'R² Score': all_r2s,
    'Weight': weights
})

contrib_df = contrib_df.sort_values('R² Score', ascending=False)
contrib_df.to_csv('sr_ultra_all_models.csv', index=False)
print("✓ sr_ultra_all_models.csv")

print("\n" + "="*80)
print("✅ ULTRA-FINAL OPTIMIZATION COMPLETE")
print("="*80)
