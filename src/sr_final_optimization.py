#!/usr/bin/env python
"""
Advanced SR Optimization - FIXED VERSION
==========================================
Maximum R² using:
- Deep neural networks with normalization
- Aggressive gradient boosting
- High-degree polynomials
- Ensemble stacking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED SR OPTIMIZATION - MAXIMUM ACCURACY")
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

# Comprehensive feature engineering
X_eng = X.copy()

# Power features
for col in X.columns:
    X_eng[f'{col}_sq'] = X[col] ** 2
    X_eng[f'{col}_cube'] = X[col] ** 3
    X_eng[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
    X_eng[f'log_{col}'] = np.log(X[col] + 1)

# Interactions
for i, col1 in enumerate(X.columns):
    for col2 in list(X.columns)[i+1:]:
        X_eng[f'{col1}*{col2}'] = X[col1] * X[col2]
        X_eng[f'{col1}/{col2}'] = X[col1] / (X[col2] + 1e-6)

print(f"✓ Features: {X_eng.shape[1]}")

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_eng)

# Split (10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Normalize target for NN
y_min, y_max = y.min(), y.max()
y_train_norm = (y_train - y_min) / (y_max - y_min)
y_test_norm = (y_test - y_min) / (y_max - y_min)

# ============================================================================
# MODEL 1: TUNED GRADIENT BOOSTING
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: GRADIENT BOOSTING (TUNED)")
print("="*80)

print("⏳ Training with aggressive parameters...")

gb = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.005,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"✓ Gradient Boosting:")
print(f"  R² = {r2_gb:.6f}")
print(f"  RMSE = {rmse_gb:.6f}")

# ============================================================================
# MODEL 2: DEEP NEURAL NETWORK
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: DEEP NEURAL NETWORK")
print("="*80)

print("⏳ Training large neural network...")

nn = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.15,
    alpha=1e-5,
    batch_size=16,
    random_state=42,
    verbose=0
)

nn.fit(X_train, y_train_norm)
y_pred_nn_norm = nn.predict(X_test)
y_pred_nn = y_pred_nn_norm * (y_max - y_min) + y_min
r2_nn = r2_score(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))

print(f"✓ Neural Network:")
print(f"  R² = {r2_nn:.6f}")
print(f"  RMSE = {rmse_nn:.6f}")

# ============================================================================
# MODEL 3: HIGH-DEGREE POLYNOMIAL
# ============================================================================

print("\n" + "="*80)
print("MODEL 3: POLYNOMIAL FEATURES (DEGREE 4)")
print("="*80)

print("⏳ Creating degree-4 polynomial...")

# Use first 30 features to avoid explosion
n_base = 30
X_train_base = X_train[:, :min(n_base, X_train.shape[1])]
X_test_base = X_test[:, :min(n_base, X_test.shape[1])]

poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(X_train_base)
X_test_poly = poly.transform(X_test_base)

ridge_poly = Ridge(alpha=1000)
ridge_poly.fit(X_train_poly, y_train)
y_pred_poly = ridge_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))

print(f"✓ Polynomial Degree 4:")
print(f"  Features: {X_train_poly.shape[1]}")
print(f"  R² = {r2_poly:.6f}")
print(f"  RMSE = {rmse_poly:.6f}")

# ============================================================================
# MODEL 4: ENSEMBLE STACK
# ============================================================================

print("\n" + "="*80)
print("MODEL 4: ENSEMBLE STACKING")
print("="*80)

print("⏳ Creating meta-ensemble...")

# Train additional base learner
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

# Generate meta-features
meta_train = np.column_stack([
    gb.predict(X_train),
    nn.predict(X_train) * (y_max - y_min) + y_min,
    ridge_poly.predict(poly.transform(X_train[:, :min(n_base, X_train.shape[1])])),
    rf.predict(X_train),
])

meta_test = np.column_stack([
    gb.predict(X_test),
    y_pred_nn,
    y_pred_poly,
    rf.predict(X_test),
])

# Meta-learner
meta = Ridge(alpha=0.01)
meta.fit(meta_train, y_train)
y_pred_stack = meta.predict(meta_test)
r2_stack = r2_score(y_test, y_pred_stack)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))

print(f"✓ Ensemble Stack:")
print(f"  R² = {r2_stack:.6f}")
print(f"  RMSE = {rmse_stack:.6f}")

# ============================================================================
# COMPARISON & BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

results = {
    'Gradient Boosting': {'r2': r2_gb, 'rmse': rmse_gb},
    'Neural Network': {'r2': r2_nn, 'rmse': rmse_nn},
    'Polynomial (Deg 4)': {'r2': r2_poly, 'rmse': rmse_poly},
    'Ensemble Stack': {'r2': r2_stack, 'rmse': rmse_stack},
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

print(f"\n📊 Progress:")
print(f"  Baseline:    R² = {baseline_r2:.4f}, RMSE = {baseline_rmse:.4f}")
print(f"  Current:     R² = {best_r2:.6f}, RMSE = {best_rmse:.6f}")

r2_pct = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
rmse_pct = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

print(f"\n  R² increase:    {r2_pct:+.1f}%")
print(f"  RMSE reduction: {rmse_pct:+.1f}%")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

import pickle

df_results.to_csv('sr_advanced_comparison.csv', index=False)
print("✓ Comparison saved: sr_advanced_comparison.csv")

# Save best model based on name
if 'Gradient Boosting' in best_name:
    with open('sr_best_model_final.pkl', 'wb') as f:
        pickle.dump(gb, f)
elif 'Neural Network' in best_name:
    with open('sr_best_model_final.pkl', 'wb') as f:
        pickle.dump(nn, f)
else:
    with open('sr_best_model_final.pkl', 'wb') as f:
        pickle.dump(meta, f)

print("✓ Best model saved: sr_best_model_final.pkl")

print("\n" + "="*80)
print("✅ ADVANCED OPTIMIZATION COMPLETE")
print("="*80)
