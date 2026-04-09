#!/usr/bin/env python
"""
Supplementary SR Optimization Strategies
=========================================
Provides faster alternative optimization methods to boost SR accuracy
while the main PySR search runs in the background.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SYMBOLIC REGRESSION - OPTIMIZATION BOOST STRATEGIES")
print("="*80)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

print("\n📊 Loading data...")
df = pd.read_csv('Shear_Data_15.csv')

input_features = ['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']
target_variable = 'SHEAR STRESS'

X = df[input_features]
y = df[target_variable]

# Remove outliers
def remove_outliers_iqr(data, features, iqr_multiplier=2.0):
    mask = np.ones(len(data), dtype=bool)
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        mask = mask & ((data[feature] >= lower) & (data[feature] <= upper))
    return data[mask].reset_index(drop=True)

df_clean = remove_outliers_iqr(df, input_features + [target_variable])
X = df_clean[input_features]
y = df_clean[target_variable]

print(f"✓ Data cleaned: {len(df)} → {len(df_clean)} samples")

# Feature engineering
X_eng = X.copy()
for col in input_features:
    if (X_eng[col] > 0).all():
        X_eng[f'log_{col}'] = np.log(X_eng[col])
        X_eng[f'{col}_sqrt'] = np.sqrt(X_eng[col])
    X_eng[f'{col}_sq'] = X_eng[col] ** 2

X_eng['AMP_x_RISE'] = X['AMP'] * X['RISE']
X_eng['ENER_x_DURATION'] = X['ENER'] * X['DURATION']

print(f"✓ Feature engineering: {X.shape[1]} → {X_eng.shape[1]} features")

# Scale data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_eng)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ============================================================================
# STRATEGY 1: GRADIENT BOOSTING (Faster alternative to RF)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 1: GRADIENT BOOSTING REGRESSOR")
print("="*80)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    n_iter_no_change=20,
    validation_fraction=0.1
)

print("\n⏳ Training Gradient Boosting...")
gb.fit(X_train, y_train)
gb_train_r2 = gb.score(X_train, y_train)
gb_test_r2 = gb.score(X_test, y_test)
y_test_pred_gb = gb.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_gb))

print(f"✓ Gradient Boosting Results:")
print(f"  Train R²: {gb_train_r2:.6f}")
print(f"  Test R²:  {gb_test_r2:.6f}")
print(f"  RMSE:     {gb_rmse:.6f}")

# SHAP-style feature importance
feature_importance_gb = gb.feature_importances_
top_features_gb_idx = np.argsort(feature_importance_gb)[-5:]
print(f"\n  Top 5 Features:")
for idx in top_features_gb_idx[::-1]:
    print(f"    {X_eng.columns[idx]}: {feature_importance_gb[idx]:.4f}")

# ============================================================================
# STRATEGY 2: NEURAL NETWORK (Multi-layer Perceptron)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 2: NEURAL NETWORK (MLP REGRESSOR)")
print("="*80)

mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=0
)

print("\n⏳ Training Neural Network...")
mlp.fit(X_train, y_train)
mlp_train_r2 = mlp.score(X_train, y_train)
mlp_test_r2 = mlp.score(X_test, y_test)
y_test_pred_mlp = mlp.predict(X_test)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_mlp))

print(f"✓ Neural Network Results:")
print(f"  Train R²: {mlp_train_r2:.6f}")
print(f"  Test R²:  {mlp_test_r2:.6f}")
print(f"  RMSE:     {mlp_rmse:.6f}")

# ============================================================================
# STRATEGY 3: ENSEMBLE STACKING
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 3: ENSEMBLE STACKING")
print("="*80)

# Create base learners
ridge = Ridge(alpha=10.0)
lasso = Lasso(alpha=0.1, max_iter=5000)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

# Train base models
base_models = {
    'Ridge': ridge,
    'Lasso': lasso,
    'ElasticNet': elastic,
    'RF': rf,
}

print("\n⏳ Training base learners for stacking...")
meta_features = np.zeros((X_train.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f"  Training {name}...", end=" ")
    model.fit(X_train, y_train)
    meta_features[:, i] = model.predict(X_train)
    print(f"✓")

# Train meta-learner on base predictions
print("  Training meta-learner (Ridge)...", end=" ")
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_features, y_train)
print("✓")

# Evaluate stacking
meta_test_features = np.zeros((X_test.shape[0], len(base_models)))
for i, (name, model) in enumerate(base_models.items()):
    meta_test_features[:, i] = model.predict(X_test)

y_test_pred_stack = meta_learner.predict(meta_test_features)
stack_train_r2 = meta_learner.score(meta_features, y_train)
stack_test_r2 = r2_score(y_test, y_test_pred_stack)
stack_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_stack))

print(f"\n✓ Ensemble Stacking Results:")
print(f"  Train R²: {stack_train_r2:.6f}")
print(f"  Test R²:  {stack_test_r2:.6f}")
print(f"  RMSE:     {stack_rmse:.6f}")

# ============================================================================
# STRATEGY 4: SVR WITH MULTIPLE KERNELS
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 4: SUPPORT VECTOR REGRESSION")
print("="*80)

svr_models = {
    'RBF': SVR(kernel='rbf', C=100, gamma='scale'),
    'Poly3': SVR(kernel='poly', degree=3, C=100, gamma='scale'),
    'Sigmoid': SVR(kernel='sigmoid', C=100, gamma='scale'),
}

best_svr = None
best_svr_r2 = -np.inf
best_svr_name = ""

for name, svr in svr_models.items():
    print(f"\n  Training SVR ({name})...", end=" ")
    svr.fit(X_train, y_train)
    y_test_pred_svr = svr.predict(X_test)
    svr_r2 = r2_score(y_test, y_test_pred_svr)
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_svr))
    print(f"✓")
    print(f"    R²: {svr_r2:.6f}, RMSE: {svr_rmse:.6f}")
    
    if svr_r2 > best_svr_r2:
        best_svr = svr
        best_svr_r2 = svr_r2
        best_svr_name = name
        best_svr_rmse = svr_rmse

print(f"\n  Best SVR: {best_svr_name} (R² = {best_svr_r2:.6f})")

# ============================================================================
# COMPARISON AND SUMMARY
# ============================================================================

print("\n" + "="*80)
print("OPTIMIZATION STRATEGIES COMPARISON")
print("="*80)

strategies = {
    'Gradient Boosting': {'r2': gb_test_r2, 'rmse': gb_rmse},
    'Neural Network': {'r2': mlp_test_r2, 'rmse': mlp_rmse},
    'Ensemble Stack': {'r2': stack_test_r2, 'rmse': stack_rmse},
    f'SVR ({best_svr_name})': {'r2': best_svr_r2, 'rmse': best_svr_rmse},
}

results_df = pd.DataFrame({
    'Strategy': list(strategies.keys()),
    'Test R²': [v['r2'] for v in strategies.values()],
    'RMSE': [v['rmse'] for v in strategies.values()],
})

results_df = results_df.sort_values('Test R²', ascending=False)
print("\n" + results_df.to_string(index=False))

best_overall = results_df.iloc[0]
print(f"\n🏆 Best Strategy: {best_overall['Strategy']}")
print(f"   R² = {best_overall['Test R²']:.6f}")
print(f"   RMSE = {best_overall['RMSE']:.6f}")

# Save results
results_df.to_csv('sr_optimization_results.csv', index=False)
print(f"\n✓ Results saved to: sr_optimization_results.csv")

# ============================================================================
# SAVE BEST MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING BEST MODELS")
print("="*80)

import pickle

models_to_save = {
    'gb_model.pkl': gb,
    'mlp_model.pkl': mlp,
    'stack_model.pkl': (meta_learner, base_models),
    'svr_model.pkl': best_svr,
}

for filename, model_obj in models_to_save.items():
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"✓ {filename}")

print("\n" + "="*80)
print("✅ OPTIMIZATION BOOST COMPLETE!")
print("="*80)
print("\n💡 All models ready for evaluation and ensemble combination")
print("   Main PySR search running in background for symbolic equations...")
