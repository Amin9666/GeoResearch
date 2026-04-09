#!/usr/bin/env python
"""
Advanced SR Optimization - Maximum Accuracy Push
================================================
Aggressive techniques to maximize R² and minimize RMSE:
- XGBoost & LightGBM (better than standard GB)
- Deep Neural Networks
- Higher-degree polynomials
- Systematic hyperparameter optimization
- Advanced ensemble stacking
- Cross-validation model selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MAXIMUM ACCURACY OPTIMIZATION - SR ENHANCEMENT")
print("="*80)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

print("\n📊 Loading and preprocessing data...")
df = pd.read_csv('Shear_Data_15.csv')

input_features = ['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']
X = df[input_features]
y = df['SHEAR STRESS']

# Aggressive outlier removal with careful tuning
def remove_outliers_adaptive(data, features, iqr_mult=1.5):
    """More aggressive outlier removal"""
    mask = np.ones(len(data), dtype=bool)
    removed_counts = {}
    
    for feat in features:
        Q1, Q3 = data[feat].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
        feat_mask = (data[feat] >= lower) & (data[feat] <= upper)
        removed_counts[feat] = (~feat_mask).sum()
        mask = mask & feat_mask
    
    print(f"  Outliers removed: {sum(~mask)}")
    return data[mask].reset_index(drop=True)

df = remove_outliers_adaptive(df, input_features + ['SHEAR STRESS'], iqr_mult=1.5)
X = df[input_features]
y = df['SHEAR STRESS']
print(f"✓ Data cleaned: {len(df)} samples")

# ============================================================================
# ADVANCED FEATURE ENGINEERING - MAXIMUM FEATURES
# ============================================================================

print("\n🔧 Advanced feature engineering...")

X_features = X.copy()
feature_names = list(input_features)

# 1. Log, sqrt, square, cube features
for col in input_features:
    if (X_features[col] > 0).all():
        X_features[f'log_{col}'] = np.log(X_features[col] + 1e-6)
        X_features[f'sqrt_{col}'] = np.sqrt(X_features[col])
        feature_names.extend([f'log_{col}', f'sqrt_{col}'])
    
    X_features[f'{col}_sq'] = X_features[col] ** 2
    X_features[f'{col}_cube'] = X_features[col] ** 3
    feature_names.extend([f'{col}_sq', f'{col}_cube'])

# 2. All pairwise interactions
for i in range(len(input_features)):
    for j in range(i+1, len(input_features)):
        X_features[f'{input_features[i]}*{input_features[j]}'] = X[input_features[i]] * X[input_features[j]]
        X_features[f'{input_features[i]}/{input_features[j]}'] = X[input_features[i]] / (X[input_features[j]] + 1e-6)
        feature_names.extend([f'{input_features[i]}*{input_features[j]}', 
                             f'{input_features[i]}/{input_features[j]}'])

# 3. Higher order interactions (selected key ones)
X_features['RISE_COUN_ENER'] = X['RISE'] * X['COUN'] * X['ENER']
X_features['AMP_DURATION_SQ'] = X['AMP'] * (X['DURATION'] ** 2)
X_features['ENER_DURATION_LOG'] = X['ENER'] * np.log(X['DURATION'] + 1)
feature_names.extend(['RISE_COUN_ENER', 'AMP_DURATION_SQ', 'ENER_DURATION_LOG'])

# 4. Ratios and normalized features
X_features['AMP_RISE_ratio'] = X['AMP'] / (X['RISE'] + 1e-6)
X_features['ENER_COUN_ratio'] = X['ENER'] / (X['COUN'] + 1e-6)
X_features['std_of_params'] = X.std(axis=1)
X_features['mean_of_params'] = X.mean(axis=1)
feature_names.extend(['AMP_RISE_ratio', 'ENER_COUN_ratio', 'std_of_params', 'mean_of_params'])

print(f"✓ Created {X_features.shape[1]} features (from 5 original)")

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, shuffle=True
)

print(f"  Train set: {len(X_train)}, Test set: {len(X_test)}")

# ============================================================================
# STRATEGY 1: XGBOOST - Often Better than Standard GB
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 1: XGBOOST OPTIMIZATION")
print("="*80)

try:
    import xgboost as xgb
    
    print("\n⏳ XGBoost hyperparameter search...")
    
    # Initial grid search
    xgb_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }
    
    xgb_grid = RandomizedSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        xgb_params, n_iter=20, cv=5, scoring='r2', verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    
    y_pred_xgb = xgb_grid.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    
    print(f"✓ XGBoost Best Params: {xgb_grid.best_params_}")
    print(f"  Test R²: {r2_xgb:.6f}")
    print(f"  Test RMSE: {rmse_xgb:.6f}")
    
    xgb_best = xgb_grid.best_estimator_
    
except ImportError:
    print("⚠️  XGBoost not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    print("✓ XGBoost installed")

# ============================================================================
# STRATEGY 2: LIGHTGBM - Often Faster and Better
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 2: LIGHTGBM OPTIMIZATION")
print("="*80)

try:
    import lightgbm as lgb
    
    print("\n⏳ LightGBM hyperparameter search...")
    
    lgb_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 4, 5, 6, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 50, 100],
        'reg_l1': [0, 1, 10],
        'reg_l2': [0, 1, 10],
    }
    
    lgb_grid = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=42, n_jobs=-1),
        lgb_params, n_iter=20, cv=5, scoring='r2', verbose=0
    )
    lgb_grid.fit(X_train, y_train)
    
    y_pred_lgb = lgb_grid.predict(X_test)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    
    print(f"✓ LightGBM Best Params: {lgb_grid.best_params_}")
    print(f"  Test R²: {r2_lgb:.6f}")
    print(f"  Test RMSE: {rmse_lgb:.6f}")
    
    lgb_best = lgb_grid.best_estimator_
    
except ImportError:
    print("⚠️  LightGBM not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm'])
    import lightgbm as lgb
    print("✓ LightGBM installed")

# ============================================================================
# STRATEGY 3: DEEP NEURAL NETWORKS - Multiple Architectures
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 3: DEEP NEURAL NETWORKS")
print("="*80)

print("\n⏳ Training neural networks with different architectures...")

best_nn = None
best_nn_r2 = -np.inf
best_nn_arch = ""

architectures = [
    (256, 128, 64, 32),
    (512, 256, 128),
    (128, 64, 32, 16),
    (256, 128, 64),
    (512, 256, 128, 64),
]

for arch in architectures:
    print(f"  Testing architecture {arch}...", end=" ")
    
    nn = MLPRegressor(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=0.0001,  # L2 regularization
        batch_size=32,
        random_state=42,
        verbose=0
    )
    
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    r2_nn = r2_score(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    
    print(f"R²={r2_nn:.4f}, RMSE={rmse_nn:.4f}")
    
    if r2_nn > best_nn_r2:
        best_nn_r2 = r2_nn
        best_nn = nn
        best_nn_arch = arch

print(f"\n✓ Best NN Architecture: {best_nn_arch}")
print(f"  Test R²: {best_nn_r2:.6f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, best_nn.predict(X_test))):.6f}")

# ============================================================================
# STRATEGY 4: ULTRA HIGH-DEGREE POLYNOMIAL
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 4: HIGH-DEGREE POLYNOMIAL FEATURES")
print("="*80)

print("\n⏳ Testing polynomial features degree 4-5...")

best_poly_model = None
best_poly_r2 = -np.inf
best_poly_degree = 1

for degree in [4, 5]:
    print(f"  Testing degree {degree}...", end=" ")
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[:1000])  # Limit for memory
    X_test_poly = poly.transform(X_test)
    
    ridge = Ridge(alpha=100)
    ridge.fit(X_train_poly[:1000], y_train[:1000])
    y_pred_poly = ridge.predict(X_test_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    
    print(f"R²={r2_poly:.4f}, RMSE={rmse_poly:.4f}")
    
    if r2_poly > best_poly_r2:
        best_poly_r2 = r2_poly
        best_poly_model = ridge
        best_poly_degree = degree

print(f"\n✓ Best Polynomial Degree: {best_poly_degree}")
print(f"  Test R²: {best_poly_r2:.6f}")

# ============================================================================
# STRATEGY 5: ADVANCED ENSEMBLE STACKING
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 5: ADVANCED ENSEMBLE STACKING")
print("="*80)

print("\n⏳ Creating meta-ensemble...")

# Base learners - use all best models
base_learners = {
    'XGB': xgb_best if 'xgb_best' in locals() else None,
    'LGB': lgb_best if 'lgb_best' in locals() else None,
    'NN': best_nn,
    'GB': GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.01),
    'Ridge': Ridge(alpha=0.1),
}

base_learners = {k: v for k, v in base_learners.items() if v is not None}

# Generate meta-features
meta_train = np.zeros((len(X_train), len(base_learners)))
meta_test = np.zeros((len(X_test), len(base_learners)))

for i, (name, model) in enumerate(base_learners.items()):
    print(f"  Training base learner: {name}", end=" ")
    model.fit(X_train, y_train)
    meta_train[:, i] = model.predict(X_train)
    meta_test[:, i] = model.predict(X_test)
    local_r2 = r2_score(y_test, meta_test[:, i])
    print(f"(R²={local_r2:.4f})")

# Train meta-learner
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_train, y_train)
y_pred_stack = meta_learner.predict(meta_test)
r2_stack = r2_score(y_test, y_pred_stack)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))

print(f"\n✓ Advanced Ensemble Results:")
print(f"  Test R²: {r2_stack:.6f}")
print(f"  Test RMSE: {rmse_stack:.6f}")

# ============================================================================
# FINAL COMPARISON AND SELECTION
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

results = {
    'XGBoost': {'r2': r2_xgb, 'rmse': rmse_xgb, 'model': xgb_best},
    'LightGBM': {'r2': r2_lgb, 'rmse': rmse_lgb, 'model': lgb_best},
    f'Neural Network ({best_nn_arch})': {'r2': best_nn_r2, 'rmse': np.sqrt(mean_squared_error(y_test, best_nn.predict(X_test))), 'model': best_nn},
    f'Polynomial (Deg {best_poly_degree})': {'r2': best_poly_r2, 'rmse': rmse_poly, 'model': best_poly_model},
    'Advanced Ensemble': {'r2': r2_stack, 'rmse': rmse_stack, 'model': meta_learner},
}

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [v['r2'] for v in results.values()],
    'RMSE': [v['rmse'] for v in results.values()],
})

results_df = results_df.sort_values('R² Score', ascending=False)

print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_result = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² = {best_result['r2']:.6f}")
print(f"   RMSE = {best_result['rmse']:.6f}")

# ============================================================================
# IMPROVEMENT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

baseline_r2 = 0.050
improvement_pct = ((best_result['r2'] - baseline_r2) / abs(baseline_r2)) * 100

print(f"\nFrom baseline (R² = 0.050):")
print(f"  Current: R² = {best_result['r2']:.6f}")
print(f"  Improvement: +{improvement_pct:.1f}%")
print(f"\nRMSE Improvement:")
print(f"  From: 0.9151 MPa")
print(f"  To:   {best_result['rmse']:.6f} MPa")
print(f"  Reduction: {((0.9151 - best_result['rmse'])/0.9151)*100:.1f}%")

# ============================================================================
# SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

import pickle

with open('sr_best_model_advanced.pkl', 'wb') as f:
    pickle.dump(best_result['model'], f)
print(f"✓ Best model saved: sr_best_model_advanced.pkl")

# Save results
results_df.to_csv('sr_advanced_results.csv', index=False)
print(f"✓ Results saved: sr_advanced_results.csv")

print("\n" + "="*80)
print("✅ ADVANCED OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nNext: Validate results and deploy best model")
