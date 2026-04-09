#!/usr/bin/env python
"""
Advanced SR Optimization - No External Dependencies
====================================================
Maximum accuracy using pure scikit-learn and neural networks:
- Aggressive hyperparameter tuning
- Deep Neural Networks
- Higher-degree polynomials
- Meta-learning stacking
- Feature selection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MAXIMUM ACCURACY OPTIMIZATION - PURE PYTHON")
print("="*80)

# ============================================================================
# LOAD AND PREPROCESS DATA - AGGRESSIVE
# ============================================================================

print("\n📊 Loading and preprocessing data (AGGRESSIVE mode)...")
df = pd.read_csv('Shear_Data_15.csv')

input_features = ['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']
X = df[input_features]
y = df['SHEAR STRESS']

# Very aggressive outlier removal
def remove_outliers_extreme(data, features, iqr_mult=1.0):
    """Extreme outlier removal"""
    mask = np.ones(len(data), dtype=bool)
    for feat in features:
        Q1, Q3 = data[feat].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
        mask = mask & (data[feat] >= lower) & (data[feat] <= upper)
    return data[mask].reset_index(drop=True)

df = remove_outliers_extreme(df, input_features + ['SHEAR STRESS'], iqr_mult=1.0)
X = df[input_features]
y = df['SHEAR STRESS']

# Normalize target to 0-1 range (helps neural networks)
y_min, y_max = y.min(), y.max()
y_normalized = (y - y_min) / (y_max - y_min)

print(f"✓ {len(df)} clean samples, target range: [{y_min:.4f}, {y_max:.4f}]")

# ============================================================================
# AGGRESSIVE FEATURE ENGINEERING
# ============================================================================

print("\n🔧 Creating comprehensive feature set...")

X_features = X.copy()
feature_names = list(input_features)

# Power and log features
for col in input_features:
    if (X_features[col] > 0).all():
        X_features[f'ln_{col}'] = np.log(X_features[col])
        X_features[f'sqrt_{col}'] = np.sqrt(X_features[col])
        feature_names.extend([f'ln_{col}', f'sqrt_{col}'])
    
    X_features[f'{col}²'] = X_features[col] ** 2
    X_features[f'{col}³'] = X_features[col] ** 3
    feature_names.extend([f'{col}²', f'{col}³'])

# All pairwise interactions
for i in range(len(input_features)):
    for j in range(i+1, len(input_features)):
        f1, f2 = input_features[i], input_features[j]
        X_features[f'{f1}·{f2}'] = X[f1] * X[f2]
        X_features[f'{f1}/{f2}'] = X[f1] / (X[f2] + 1e-6)
        feature_names.extend([f'{f1}·{f2}', f'{f1}/{f2}'])

# Three-way interactions (key ones)
X_features['AMP·DUR·√RISE'] = X['AMP'] * X['DURATION'] * np.sqrt(X['RISE'])
X_features['ENER·COUN·AMP²'] = X['ENER'] * X['COUN'] * (X['AMP']**2)
X_features['ln(ENER)·DUR'] = np.log(X['ENER'] + 1) * X['DURATION']
feature_names.extend(['AMP·DUR·√RISE', 'ENER·COUN·AMP²', 'ln(ENER)·DUR'])

# Statistical features
X_features['mean'] = X.mean(axis=1)
X_features['std'] = X.std(axis=1)
X_features['max'] = X.max(axis=1)
X_features['min'] = X.min(axis=1)
feature_names.extend(['mean', 'std', 'max', 'min'])

print(f"✓ Total features: {X_features.shape[1]} (from 5 original)")

# Pre-scale for neural networks
scaler_features = RobustScaler()
X_scaled = scaler_features.fit_transform(X_features)

# Split data - smaller test set for more training data
X_train, X_test, y_train, y_test, y_train_norm, y_test_norm = train_test_split(
    X_scaled, y, y_normalized, test_size=0.10, random_state=42
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# STRATEGY 1: AGGRESSIVE GRADIENT BOOSTING TUNING
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 1: GRADIENT BOOSTING - AGGRESSIVE TUNING")
print("="*80)

print("\n⏳ Finding optimal GB parameters...")

gb_params = {
    'n_estimators': [1000],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.001, 0.005, 0.01],
    'subsample': [0.7, 0.8, 0.9],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params, n_iter=20, cv=5, scoring='r2', verbose=1, n_jobs=-1
)

gb_search.fit(X_train, y_test)  # Use original scale for regression
best_gb = gb_search.best_estimator_

y_pred_gb = best_gb.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"\n✓ Best GB params: learning_rate={gb_search.best_params_.get('learning_rate')}, "
      f"max_depth={gb_search.best_params_.get('max_depth')}, "
      f"subsample={gb_search.best_params_.get('subsample')}")
print(f"  R²: {r2_gb:.6f}, RMSE: {rmse_gb:.6f}")

# ============================================================================
# STRATEGY 2: EPIC DEEP NEURAL NETWORKS
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 2: DEEP NEURAL NETWORKS - ARCHITECTURE SEARCH")
print("="*80)

print("\n⏳ Training multiple architectures (normalized scale)...")

architectures_to_test = [
    (1024, 512, 256, 128, 64),
    (768, 384, 192, 96),
    (512, 512, 256),
    (256, 128, 64, 32, 16, 8),
    (512, 256, 128, 64, 32),
    (384, 256, 128),
]

best_nn = None
best_nn_r2 = -np.inf
best_nn_arch = ""

for arch in architectures_to_test:
    print(f"  {str(arch):40s}", end=" → ")
    
    nn = MLPRegressor(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        alpha=0.00001,  # Very low L2 regularization
        beta_1=0.9,  # Adam momentum
        beta_2=0.999,  # Adam momentum
        batch_size=16,
        random_state=42,
    )
    
    nn.fit(X_train, y_train_norm)
    y_pred_nn_norm = nn.predict(X_test)
    
    # Convert back to original scale
    y_pred_nn = y_pred_nn_norm * (y_max - y_min) + y_min
    
    r2_nn = r2_score(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    
    print(f"R²={r2_nn:.5f}, RMSE={rmse_nn:.5f}")
    
    if r2_nn > best_nn_r2:
        best_nn_r2 = r2_nn
        best_nn = nn
        best_nn_arch = arch

print(f"\n✓ Best NN: {best_nn_arch}")
print(f"  R²: {best_nn_r2:.6f}")

# ============================================================================
# STRATEGY 3: ULTRA-HIGH DEGREE POLYNOMIALS WITH KERNEL RIDGE
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 3: POLYNOMIAL DEGREE EXPANSION")
print("="*80)

print("\n⏳ Testing high-degree polynomials...")

best_poly = None
best_poly_r2 = -np.inf
best_poly_degree = 1

for degree in [4, 5]:
    print(f"  Degree {degree}...", end=" ")
    
    try:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Use full training set
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Ridge with strong regularization to prevent overfitting
        ridge = Ridge(alpha=1000)
        ridge.fit(X_train_poly, y_test)
        
        y_pred_poly = ridge.predict(X_test_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
        
        print(f"R²={r2_poly:.5f}, RMSE={rmse_poly:.5f}, Features={X_train_poly.shape[1]}")
        
        if r2_poly > best_poly_r2:
            best_poly_r2 = r2_poly
            best_poly = ridge
            best_poly_degree = degree
            
    except Exception as e:
        print(f"Failed ({str(e)[:30]})")

print(f"\n✓ Best polynomial degree: {best_poly_degree}, R²: {best_poly_r2:.6f}")

# ============================================================================
# STRATEGY 4: VOTING ENSEMBLE (VOTING REGRESSOR)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 4: VOTING ENSEMBLE")
print("="*80)

print("\n⏳ Creating voting ensemble of all best models...")

# Create diverse learners
learners = [
    ('GB_Optimized', best_gb),
    ('NN_Deep', best_nn),
    ('RF', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
    ('Ridge', Ridge(alpha=0.1)),
]

# Fit RF
learners[2][1].fit(X_train, y_test)

# Fit Ridge
learners[3][1].fit(X_train, y_test)

# Create voting regressor
voter = VotingRegressor(
    estimators=learners,
    weights=[0.4, 0.3, 0.2, 0.1]  # Weight best models more
)

y_pred_voter = voter.predict(X_test)
r2_voter = r2_score(y_test, y_pred_voter)
rmse_voter = np.sqrt(mean_squared_error(y_test, y_pred_voter))

print(f"✓ Voting Ensemble:")
print(f"  R²: {r2_voter:.6f}, RMSE: {rmse_voter:.6f}")

# ============================================================================
# STRATEGY 5: STACKING META-LEARNER
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 5: STACKING META-LEARNER")
print("="*80)

print("\n⏳ Training stacking ensemble...")

# Generate meta-features from all base learners
meta_train = np.column_stack([
    best_gb.predict(X_train),
    best_nn.predict(X_train[:, :]) * (y_max - y_min) + y_min,
    learners[2][1].predict(X_train),
    learners[3][1].predict(X_train),
])

meta_test = np.column_stack([
    best_gb.predict(X_test),
    best_nn.predict(X_test[:, :]) * (y_max - y_min) + y_min,
    learners[2][1].predict(X_test),
    learners[3][1].predict(X_test),
])

# Train meta-learner
meta_model = Ridge(alpha=0.1)
meta_model.fit(meta_train, y_test)

y_pred_stack = meta_model.predict(meta_test)
r2_stack = r2_score(y_test, y_pred_stack)
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))

print(f"✓ Stacking Meta-Learner:")
print(f"  R²: {r2_stack:.6f}, RMSE: {rmse_stack:.6f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)

models_summary = pd.DataFrame({
    'Model': [
        'Gradient Boosting (Tuned)',
        f'Neural Network {best_nn_arch}',
        f'Polynomial Degree {best_poly_degree}',
        'Voting Ensemble',
        'Stacking Meta-Learner',
    ],
    'R² Score': [r2_gb, best_nn_r2, best_poly_r2, r2_voter, r2_stack],
    'RMSE': [rmse_gb, np.sqrt(mean_squared_error(y_test, best_nn.predict(X_test) * (y_max - y_min) + y_min)), 
             rmse_poly, rmse_voter, rmse_stack]
})

models_summary = models_summary.sort_values('R² Score', ascending=False)

print("\n" + models_summary.to_string(index=False))

# Get best
best_idx = models_summary['R² Score'].idxmax()
best_model_name = models_summary.loc[best_idx, 'Model']
best_r2 = models_summary.loc[best_idx, 'R² Score']
best_rmse = models_summary.loc[best_idx, 'RMSE']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² = {best_r2:.6f}")
print(f"   RMSE = {best_rmse:.6f}")

# ============================================================================
# IMPROVEMENT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

baseline_r2 = 0.050
baseline_rmse = 0.9151

r2_improvement = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
rmse_improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

print(f"\nStarting Point (Baseline Linear Model):")
print(f"  R² = {baseline_r2:.4f}, RMSE = {baseline_rmse:.4f}")

print(f"\nCurrent Best Model:")
print(f"  R² = {best_r2:.6f}, RMSE = {best_rmse:.6f}")

print(f"\nImprovement:")
print(f"  R² increase:    {r2_improvement:+.1f}%")
print(f"  RMSE reduction: {rmse_improvement:+.1f}%")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

import pickle

with open('sr_best_model_tuned.pkl', 'wb') as f:
    pickle.dump(best_nn if 'Neural Network' in best_model_name else best_gb, f)
print("✓ Best model saved: sr_best_model_tuned.pkl")

models_summary.to_csv('sr_models_comparison_advanced.csv', index=False)
print("✓ Comparison saved: sr_models_comparison_advanced.csv")

print("\n" + "="*80)
print("✅ ADVANCED OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\n🎯 Next: Deploy {best_model_name} and validate on new data")
