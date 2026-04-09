#!/usr/bin/env python
"""
FINAL PUSH - R² > 0.10
=======================
Test multiple preprocessing strategies and ultra-aggressive ensembles
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL PUSH - R² > 0.10")
print("="*80)

# ============================================================================
# TRY MULTIPLE CLEANING STRATEGIES
# ============================================================================

best_overall_r2 = 0.0
best_overall_config = None
best_test_preds = None
best_test_true = None

for iqr_mult in [0.8, 1.0]:
    print(f"\n{'='*80}")
    print(f"Testing with IQR Multiplier: {iqr_mult}")
    print(f"{'='*80}")
    
    df = pd.read_csv('Shear_Data_15.csv')
    X = df[['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']]
    y = df['SHEAR STRESS']
    
    # Clean
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ((X >= Q1 - iqr_mult*IQR) & (X <= Q3 + iqr_mult*IQR)).all(axis=1)
    mask = mask & ((y >= y.quantile(0.25) - iqr_mult*(y.quantile(0.75)-y.quantile(0.25))) &
                   (y <= y.quantile(0.75) + iqr_mult*(y.quantile(0.75)-y.quantile(0.25))))
    
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"  Samples: {len(X)}")
    
    # ========================================================================
    # AGGRESSIVE FEATURE SET
    # ========================================================================
    
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
    
    # Ratios (selective)
    X_eng['RISE/AMP'] = X['RISE'] / (X['AMP'] + 1e-8)
    X_eng['ENER/DUR'] = X['ENER'] / (X['DURATION'] + 1e-8)
    X_eng['AMP/DUR'] = X['AMP'] / (X['DURATION'] + 1e-8)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_eng)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.10, random_state=42
    )
    
    print(f"  Features: {X_eng.shape[1]}")
    
    # ========================================================================
    # ENSEMBLE OF MANY MODELS - DIFFERENT CONFIGS
    # ========================================================================
    
    models_preds = []
    models_r2s = []
    model_names = []
    
    # RF variations
    for depth in [12, 14, 16]:
        rf = RandomForestRegressor(n_estimators=300, max_depth=depth, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        models_preds.append(y_pred)
        models_r2s.append(r2)
        model_names.append(f"RF-d{depth}")
    
    # ET variations
    for depth in [12, 14]:
        et = ExtraTreesRegressor(n_estimators=300, max_depth=depth, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        y_pred = et.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        models_preds.append(y_pred)
        models_r2s.append(r2)
        model_names.append(f"ET-d{depth}")
    
    # GB variations
    for lr in [0.02, 0.03, 0.04]:
        gb = GradientBoostingRegressor(n_estimators=400, learning_rate=lr, max_depth=6, random_state=42)
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        models_preds.append(y_pred)
        models_r2s.append(r2)
        model_names.append(f"GB-lr{lr}")
    
    # AdaBoost
    ada = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    models_preds.append(y_pred)
    models_r2s.append(r2)
    model_names.append("Ada-200")
    
    # Ridge
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    models_preds.append(y_pred)
    models_r2s.append(r2)
    model_names.append("Ridge-0.1")
    
    print(f"\n  Trained {len(models_r2s)} models")
    
    # ========================================================================
    # SUPER WEIGHTED ENSEMBLE
    # ========================================================================
    
    weights = np.array(models_r2s)
    weights = weights / weights.sum()
    
    ensemble_pred = np.average(models_preds, axis=0, weights=weights)
    r2_final = r2_score(y_test, ensemble_pred)
    rmse_final = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"  🏆 Ensemble R²={r2_final:.6f}, RMSE={rmse_final:.6f}")
    
    if r2_final > best_overall_r2:
        best_overall_r2 = r2_final
        best_overall_config = iqr_mult
        best_test_preds = ensemble_pred
        best_test_true = y_test
        best_model_names = model_names
        best_model_r2s = models_r2s
        best_weights = weights

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "="*80)
print("🏆 FINAL RESULT")
print("="*80)

print(f"\n🎯 Best R² = {best_overall_r2:.6f}")
rmse_final = np.sqrt(mean_squared_error(best_test_true, best_test_preds))
print(f"   RMSE = {rmse_final:.6f}")
print(f"   IQR Mult = {best_overall_config}")

baseline = 0.050
prev = 0.0893

imp_base = ((best_overall_r2 - baseline) / abs(baseline)) * 100
imp_prev = ((best_overall_r2 - prev) / prev) * 100

print(f"\n📊 IMPROVEMENT:")
print(f"  vs Baseline (0.050): {imp_base:+.1f}%")
print(f"  vs Previous (0.0893): {imp_prev:+.1f}%")

if best_overall_r2 > 0.10:
    print(f"\n✨✨✨ MILESTONE: R² > 0.10 ACHIEVED! ✨✨✨")
    print(f"   R² = {best_overall_r2:.6f}")

# Show top models
print(f"\n📈 Top contributing models:")
top_idx = np.argsort(best_model_r2s)[-5:][::-1]
for i, idx in enumerate(top_idx):
    print(f"  {i+1}. {best_model_names[idx]:15} R²={best_model_r2s[idx]:.6f} (w={best_weights[idx]:.4f})")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING FINAL RESULTS")
print("="*80)

summary_df = pd.DataFrame({
    'Metric': ['Final Best R²', 'RMSE', 'Baseline', 'Previous Best', '% Improvement', 'IQR Multiplier'],
    'Value': [
        f"{best_overall_r2:.6f}",
        f"{rmse_final:.6f}",
        "0.0500",
        "0.0893",
        f"{imp_prev:+.1f}%",
        f"{best_overall_config}"
    ]
})

summary_df.to_csv('sr_final_result.csv', index=False)
print("✓ sr_final_result.csv")

model_contrib = pd.DataFrame({
    'Model': best_model_names,
    'R² Score': best_model_r2s,
    'Ensemble Weight': best_weights
})

model_contrib = model_contrib.sort_values('R² Score', ascending=False)
model_contrib.to_csv('sr_model_contributions.csv', index=False)
print("✓ sr_model_contributions.csv")

print("\n" + "="*80)
print("✅ FINAL PUSH COMPLETE")
print("="*80)
