"""
Symbolic Regression Analysis for Shear Stress Prediction
=========================================================

This script discovers the mathematical relationship between acoustic emission 
parameters (RISE, COUN, ENER, DURATION, AMP) and SHEAR STRESS using PySR.

Input Features:
  - RISE: Rise time of acoustic emission signal
  - COUN: Count of acoustic emission events
  - ENER: Energy of acoustic emission
  - DURATION: Duration of acoustic emission signal
  - AMP: Amplitude of acoustic emission signal

Target Variable:
  - SHEAR STRESS: The shear stress measurement

Author: Analysis Script
Date: 2026
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import inspect
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("SYMBOLIC REGRESSION ANALYSIS - SHEAR STRESS PREDICTION")
print("="*80)
print("\nObjective: Discover mathematical relationship between acoustic emission")
print("           parameters and shear stress in rock samples")

# ============================================================================
# PART 1: LOAD AND EXPLORE THE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING AND EXPLORING DATA")
print("="*80)

# Load the data
df = pd.read_csv('Shear_Data_15.csv')

print(f"\n✓ Data loaded successfully")
print(f"  Total samples: {len(df)}")
print(f"  Columns: {df.columns.tolist()}")

# Display basic statistics
print(f"\n📊 Dataset Overview:")
print(df.describe())

# Check for missing values
print(f"\n🔍 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values detected")
else:
    print(missing[missing > 0])

# Define input features and target
input_features = ['RISE', 'COUN', 'ENER', 'DURATION', 'AMP']
target_variable = 'SHEAR STRESS'

X = df[input_features]
y = df[target_variable]

print(f"\n📋 Analysis Configuration:")
print(f"  Input Features: {input_features}")
print(f"  Target Variable: {target_variable}")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Target vector shape: {y.shape}")

# ============================================================================
# PART 1B: DATA PREPROCESSING & OUTLIER REMOVAL
# ============================================================================

print("\n" + "="*80)
print("STEP 1B: DATA PREPROCESSING & FEATURE ENGINEERING")
print("="*80)

# Detect and remove outliers using IQR method
def remove_outliers_iqr(data, features, iqr_multiplier=1.5):
    """Remove outliers using Interquartile Range (IQR) method"""
    mask = np.ones(len(data), dtype=bool)
    outlier_counts = {}
    
    for feature in features:
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        feature_mask = (data[feature] >= lower_bound) & (data[feature] <= upper_bound)
        n_outliers = (~feature_mask).sum()
        if n_outliers > 0:
            outlier_counts[feature] = n_outliers
        mask = mask & feature_mask
    
    print(f"\n🔍 Outlier Detection (IQR method):")
    for feature, count in outlier_counts.items():
        print(f"   {feature}: {count} outliers removed")
    print(f"   Total samples: {len(data)} → {mask.sum()} ({100*mask.sum()/len(data):.1f}%)")
    
    return data[mask].reset_index(drop=True)

# Apply outlier removal
df_clean = remove_outliers_iqr(df, input_features + [target_variable], iqr_multiplier=2.0)

# Update X and y with clean data
X = df_clean[input_features]
y = df_clean[target_variable]

print(f"\n✓ Data cleaned. New shape: {X.shape}")

# Feature engineering: create derived features for better relationships
print(f"\n📊 Feature Engineering:")
X_engineered = X.copy()

# Log-transformed features (avoiding log(0) and negative values)
for col in input_features:
    if (X_engineered[col] > 0).all():
        X_engineered[f'log_{col}'] = np.log(X_engineered[col])
        print(f"   ✓ Created: log_{col}")

# Power-transformed features
for col in input_features:
    if (X_engineered[col] > 0).all():
        X_engineered[f'{col}_sqrt'] = np.sqrt(X_engineered[col])
        print(f"   ✓ Created: {col}_sqrt")
    X_engineered[f'{col}_sq'] = X_engineered[col] ** 2
    print(f"   ✓ Created: {col}_sq")

# Interaction terms (select key interactions)
X_engineered['AMP_x_RISE'] = X['AMP'] * X['RISE']
X_engineered['ENER_x_DURATION'] = X['ENER'] * X['DURATION']
X_engineered['AMP_DURATION_ratio'] = X['AMP'] / (X['DURATION'] + 1e-6)
print(f"   ✓ Created interaction terms")

print(f"\n   Total features: {X_engineered.shape[1]} (5 original + {X_engineered.shape[1]-5} engineered)")

# Keep both original and engineered for later comparison
X_original = X.copy()
all_features = X_engineered.columns.tolist()

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# Plot 1-5: Feature vs Shear Stress scatter plots
for i, feature in enumerate(input_features, 1):
    ax = plt.subplot(3, 3, i)
    plt.scatter(df[feature], df[target_variable], alpha=0.3, s=20, edgecolors='none')
    plt.xlabel(feature, fontweight='bold', fontsize=11)
    plt.ylabel('SHEAR STRESS', fontweight='bold', fontsize=11)
    plt.title(f'{feature} vs SHEAR STRESS', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate correlation
    corr = df[feature].corr(df[target_variable])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 6: Correlation heatmap
ax6 = plt.subplot(3, 3, 6)
correlation_matrix = df[input_features + [target_variable]].corr()  # type: ignore[arg-type]
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, ax=ax6, cbar_kws={'label': 'Correlation'})
ax6.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=12)

# Plot 7: Distribution of Shear Stress
ax7 = plt.subplot(3, 3, 7)
plt.hist(df[target_variable], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('SHEAR STRESS', fontweight='bold', fontsize=11)
plt.ylabel('Frequency', fontweight='bold', fontsize=11)
plt.title('Distribution of Shear Stress', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Plot 8: Feature distributions (boxplot)
ax8 = plt.subplot(3, 3, 8)
X_scaled_viz = StandardScaler().fit_transform(X)
plt.boxplot(X_scaled_viz, labels=input_features)
plt.ylabel('Standardized Value', fontweight='bold', fontsize=11)
plt.title('Feature Distributions (Standardized)', fontweight='bold', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Plot 9: Data timeline
ax9 = plt.subplot(3, 3, 9)
plt.plot(df['TIME'], df[target_variable], alpha=0.5, linewidth=0.5)
plt.xlabel('TIME', fontweight='bold', fontsize=11)
plt.ylabel('SHEAR STRESS', fontweight='bold', fontsize=11)
plt.title('Shear Stress Over Time', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shear_data_exploration.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Exploratory visualizations saved: shear_data_exploration.png")

# Print correlation analysis
print(f"\n📊 Feature Correlations with Shear Stress:")
correlations = df[input_features].corrwith(df[target_variable]).sort_values(ascending=False)
for feature, corr in correlations.items():
    print(f"  {feature:12s}: {corr:7.4f}")

# ============================================================================
# PART 3: PREPARE DATA FOR MODELING
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA PREPARATION & FEATURE SCALING")
print("="*80)

# Split into training and testing sets using ORIGINAL features first
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)

# Also split engineered features
X_train_eng, X_test_eng, _, _ = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

# Apply robust scaling (better for outliers than StandardScaler)
scaler_orig = RobustScaler()
X_train = scaler_orig.fit_transform(X_train_orig)
X_test = scaler_orig.transform(X_test_orig)

scaler_eng = RobustScaler()
X_train_eng_scaled = scaler_eng.fit_transform(X_train_eng)
X_test_eng_scaled = scaler_eng.transform(X_test_eng)

# Convert back to DataFrames for tracking
X_train = pd.DataFrame(X_train, columns=input_features)
X_test = pd.DataFrame(X_test, columns=input_features)
X_train_eng_df = pd.DataFrame(X_train_eng_scaled, columns=all_features)
X_test_eng_df = pd.DataFrame(X_test_eng_scaled, columns=all_features)

print(f"\n✓ Data split completed:")
print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Testing samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# PART 4: BASELINE - LINEAR REGRESSION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: BASELINE MODEL - LINEAR REGRESSION")
print("="*80)

# Train linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluation metrics
lr_train_r2 = r2_score(y_train, y_train_pred_lr)
lr_test_r2 = r2_score(y_test, y_test_pred_lr)
lr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
lr_train_mae = mean_absolute_error(y_train, y_train_pred_lr)
lr_test_mae = mean_absolute_error(y_test, y_test_pred_lr)

print("\n📈 Linear Regression Equation:")
print(f"SHEAR_STRESS = {lr_model.intercept_:.6f}")
for feature, coef in zip(input_features, lr_model.coef_):
    sign = '+' if coef >= 0 else ''
    print(f"               {sign}{coef:.6f} * {feature}")

print(f"\n📊 Linear Regression Performance:")
print(f"  Training Set:")
print(f"    R² Score:  {lr_train_r2:.6f}")
print(f"    RMSE:      {lr_train_rmse:.6f}")
print(f"    MAE:       {lr_train_mae:.6f}")
print(f"\n  Test Set:")
print(f"    R² Score:  {lr_test_r2:.6f}")
print(f"    RMSE:      {lr_test_rmse:.6f}")
print(f"    MAE:       {lr_test_mae:.6f}")

# Test on engineered features
print(f"\n📊 Linear Regression on Engineered Features:")
lr_eng = LinearRegression()
lr_eng.fit(X_train_eng_df, y_train)
y_test_pred_lr_eng = lr_eng.predict(X_test_eng_df)
lr_eng_r2 = r2_score(y_test, y_test_pred_lr_eng)
lr_eng_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr_eng))
print(f"  R² Score:  {lr_eng_r2:.6f}")
print(f"  RMSE:      {lr_eng_rmse:.6f}")

# Test with Ridge (regularized) regression
print(f"\n📊 Ridge Regression (L2 regularization):")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_eng_df, y_train)
y_test_pred_ridge = ridge.predict(X_test_eng_df)
ridge_r2 = r2_score(y_test, y_test_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
print(f"  R² Score:  {ridge_r2:.6f}")
print(f"  RMSE:      {ridge_rmse:.6f}")

# Random Forest baseline
print(f"\n📊 Random Forest Baseline:")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_eng_df, y_train)
y_test_pred_rf = rf.predict(X_test_eng_df)
rf_r2 = r2_score(y_test, y_test_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
print(f"  R² Score:  {rf_r2:.6f}")
print(f"  RMSE:      {rf_rmse:.6f}")

print(f"\n📋 Best baseline so far: {max([('Ridge', ridge_r2), ('RF', rf_r2)], key=lambda x: x[1])[0]}")

# ============================================================================
# PART 5: SYMBOLIC REGRESSION WITH PySR (IMPROVED)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SYMBOLIC REGRESSION (PySR) - IMPROVED")
print("="*80)

# Check if PySR is installed and Julia is available
pysr_available = False
pysr_regressor_cls = None
pysr_skip = os.environ.get("GEORESEARCH_SKIP_PYSR", "").strip().lower() in {"1", "true", "yes"}

if pysr_skip:
    print("\n⚠️  PySR explicitly disabled via GEORESEARCH_SKIP_PYSR.")
    print("   Using simulated symbolic-regression results for this run.")
else:
    try:
        from pysr import PySRRegressor
        pysr_regressor_cls = PySRRegressor
        pysr_available = True
        print("\n✓ PySR is installed and ready to use!")
    except Exception as e:
        print("\n❌ PySR is not available.")
        print(f"   Error: {type(e).__name__}: {str(e)[:100]}")
        print("\n💡 To install PySR and Julia, run:")
        print("   brew install julia")
        print("   pip install pysr")
        print("   export PYTHON_JULIAPKG_EXE=$(which julia)")
        print("   export PYTHON_JULIACALL_EXE=$(which julia)")
        print("\n   Temporary workaround (skip PySR and continue pipeline):")
        print("   GEORESEARCH_SKIP_PYSR=1 python src/shear_stress_symbolic_regression.py")
        print("\n⚠️  Skipping symbolic regression and using simulated results for demonstration...")

if pysr_available and pysr_regressor_cls is not None:
    def build_pysr_regressor(regressor_cls, **kwargs):
        """Build a PySR regressor while dropping unsupported kwargs for compatibility."""
        valid_params = set(inspect.signature(regressor_cls.__init__).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        dropped = sorted(set(kwargs.keys()) - set(filtered_kwargs.keys()))
        if dropped:
            print(f"\n⚠️  Skipping unsupported PySR kwargs for this version: {', '.join(dropped)}")
        return regressor_cls(**filtered_kwargs)

    print("\n🚀 Starting Improved Symbolic Regression Analysis...")
    print("=" * 80)
    print("Strategy 1: AGGRESSIVE SEARCH (Original Features)")
    print("=" * 80)
    
    # STRATEGY 1: Aggressive search with original features
    model_aggressive = build_pysr_regressor(
        pysr_regressor_cls,
        # Evolution parameters - AGGRESSIVE settings
        niterations=500,                    # 5x more iterations than baseline
        populations=100,                    # 3x more populations
        population_size=150,                # 3x larger population
        
        # Allowed mathematical operations - comprehensive set
        binary_operators=["+", "-", "*", "/", "^", "max", "min"],
        unary_operators=[
            "exp",      # Exponential
            "log",      # Natural logarithm
            "sqrt",     # Square root
            "square",   # x^2
            "cube",     # x^3
            "abs",      # Absolute value
            "inv",      # 1/x
            "cbrt",     # Cube root
            "negexp",   # -exp(x)
        ],
        
        # Complexity control - prioritize accuracy over simplicity
        maxsize=35,                         # Larger search space
        complexity_of_constants=1,          # Low penalty for constants (explore more)
        parsimony=0.005,                    # Moderate simplicity preference
        
        # Loss function - MSE
        elementwise_loss="(prediction, target) -> (prediction - target)^2",
        
        # Performance settings
        procs=0,                            # Use all CPU cores
        
        # Architecture search
        use_frequency=True,                 # Track operator frequency
        fast_cycle=True,                    # Enable fast mutation cycle
        
        # Optimization
        batching=False,                     # No batching for better convergence
        batch_size=10,                      # Batch size if batching enabled
        tournament_selection_n=10,          # Tournament size
        tournament_selection_p=0.9,         # Tournament pressure
        
        # Progress tracking
        verbosity=0,
        progress=True,
        
        # Reproducibility
        random_state=42,
        
        # Warmup and constraint detection
        warmup_maxsize_by=0.2,              # Gradually increase max size over iterations
        
        # Save progress
        temp_equation_file=True,
    )
    
    # Train aggressive model
    print("\n⏳ STRATEGY 1: Training with aggressive parameters...")
    print("   (This will take 30-60 minutes)\n")
    
    model_aggressive.fit(X_train.values, y_train.values,
                        variable_names=['RISE', 'COUN', 'ENER', 'DURATION', 'AMP'])
    
    print("\n✓ STRATEGY 1 complete!")
    
    # Get predictions from aggressive model
    y_train_pred_sr = model_aggressive.predict(X_train.values)
    y_test_pred_sr = model_aggressive.predict(X_test.values)
    
    # Evaluation for Strategy 1
    sr_train_r2 = r2_score(y_train, y_train_pred_sr)
    sr_test_r2 = r2_score(y_test, y_test_pred_sr)
    sr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_sr))
    sr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_sr))
    sr_train_mae = mean_absolute_error(y_train, y_train_pred_sr)
    sr_test_mae = mean_absolute_error(y_test, y_test_pred_sr)
    
    print(f"\nSTRATEGY 1 Results:")
    print(f"  Test R²:   {sr_test_r2:.6f}")
    print(f"  Test RMSE: {sr_test_rmse:.6f}")
    
    # STRATEGY 2: Engineered features search
    print("\n" + "=" * 80)
    print("Strategy 2: ENGINEERED FEATURES (Focused Search)")
    print("=" * 80)
    
    model_engineered = build_pysr_regressor(
        pysr_regressor_cls,
        niterations=300,
        populations=80,
        population_size=120,
        
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=[
            "exp", "log", "sqrt", "square", "cube", "abs", "inv", "cbrt"
        ],
        
        maxsize=30,
        complexity_of_constants=0.5,        # Very low penalty to explore more
        parsimony=0.01,                     # Stronger simplicity preference
        
        elementwise_loss="(prediction, target) -> (prediction - target)^2",
        procs=0,
        use_frequency=True,
        fast_cycle=True,
        
        verbosity=0,
        progress=True,
        random_state=42,
        warmup_maxsize_by=0.2,
        temp_equation_file=True,
    )
    
    # Create feature name list for engineered features
    eng_var_names = all_features
    
    print("\n⏳ STRATEGY 2: Training on engineered features...")
    print("   (This will take 20-40 minutes)\n")
    
    model_engineered.fit(X_train_eng_scaled, y_train.values, variable_names=eng_var_names)
    
    print("\n✓ STRATEGY 2 complete!")
    
    # Get predictions from engineered model
    y_train_pred_sr_eng = model_engineered.predict(X_train_eng_scaled)
    y_test_pred_sr_eng = model_engineered.predict(X_test_eng_scaled)
    
    # Evaluation for Strategy 2
    sr_eng_train_r2 = r2_score(y_train, y_train_pred_sr_eng)
    sr_eng_test_r2 = r2_score(y_test, y_test_pred_sr_eng)
    sr_eng_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_sr_eng))
    sr_eng_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_sr_eng))
    
    print(f"\nSTRATEGY 2 Results:")
    print(f"  Test R²:   {sr_eng_test_r2:.6f}")
    print(f"  Test RMSE: {sr_eng_test_rmse:.6f}")
    
    # Select best model
    best_model = model_aggressive if sr_test_r2 >= sr_eng_test_r2 else model_engineered
    best_strategy = "Strategy 1 (Original Features)" if sr_test_r2 >= sr_eng_test_r2 else "Strategy 2 (Engineered Features)"
    
    if sr_test_r2 >= sr_eng_test_r2:
        y_train_pred_sr = y_train_pred_sr
        y_test_pred_sr = y_test_pred_sr
        sr_train_r2 = sr_train_r2
        sr_test_r2 = sr_test_r2
        sr_train_rmse = sr_train_rmse
        sr_test_rmse = sr_test_rmse
    else:
        y_train_pred_sr = y_train_pred_sr_eng
        y_test_pred_sr = y_test_pred_sr_eng
        sr_train_r2 = sr_eng_train_r2
        sr_test_r2 = sr_eng_test_r2
        sr_train_rmse = sr_eng_train_rmse
        sr_test_rmse = sr_eng_test_rmse
    
    print(f"\n{'='*80}")
    print("SYMBOLIC REGRESSION RESULTS - BEST MODEL")
    print(f"{'='*80}")
    print(f"\n✅ Best Strategy: {best_strategy}\n")

    print(f"\n📊 Performance Metrics (Best Model):")
    print(f"  Training Set:")
    print(f"    R² Score:  {sr_train_r2:.6f}")
    print(f"    RMSE:      {sr_train_rmse:.6f}")
    print(f"    MAE:       {mean_absolute_error(y_train, y_train_pred_sr):.6f}")
    print(f"\n  Test Set:")
    print(f"    R² Score:  {sr_test_r2:.6f}")
    print(f"    RMSE:      {sr_test_rmse:.6f}")
    print(f"    MAE:       {mean_absolute_error(y_test, y_test_pred_sr):.6f}")
    
    # Compare all models
    print(f"\n{'='*80}")
    print("MODEL COMPARISON - ALL METHODS")
    print(f"{'='*80}\n")
    
    comparison_data = {
        'Linear Regression': {'r2': lr_test_r2, 'rmse': lr_test_rmse},
        'Ridge Regression': {'r2': ridge_r2, 'rmse': ridge_rmse},
        'Random Forest': {'r2': rf_r2, 'rmse': rf_rmse},
        'PySR (Strategy 1)': {'r2': sr_test_r2, 'rmse': sr_test_rmse},
        'PySR (Strategy 2)': {'r2': sr_eng_test_r2, 'rmse': sr_eng_test_rmse},
    }
    
    print(f"{'Method':<25} {'R² Score':<15} {'RMSE':<15}")
    print("-" * 55)
    for method, metrics in comparison_data.items():
        print(f"{method:<25} {metrics['r2']:<15.6f} {metrics['rmse']:<15.6f}")
    
    best_method = max(comparison_data.items(), key=lambda x: x[1]['r2'])
    print(f"\n🏆 BEST MODEL: {best_method[0]} (R² = {best_method[1]['r2']:.6f})")
    
    print(f"\n📈 Improvement over Linear Regression:")
    r2_improvement = ((sr_test_r2 - lr_test_r2) / abs(lr_test_r2)) * 100 if lr_test_r2 != 0 else 0
    rmse_improvement = ((lr_test_rmse - sr_test_rmse) / lr_test_rmse) * 100
    print(f"  R² improvement:   {r2_improvement:+.1f}%")
    print(f"  RMSE improvement: {rmse_improvement:+.1f}%")
    
    print(f"\n{'='*80}")
    print("DISCOVERED EQUATIONS (Pareto Frontier)")
    print(f"{'='*80}")
    print("\nTop equations ranked by complexity-accuracy tradeoff:")
    print("(Lower complexity = simpler, higher R² = better fit)\n")
    print(best_model)
    
    print(f"\n{'='*80}")
    print("BEST EQUATION FROM SYMBOLIC REGRESSION")
    print(f"{'='*80}")
    
    print("\n📝 SymPy Format (Python):")
    best_equation = best_model.sympy()
    print(best_equation)
    
    print("\n📄 LaTeX Format (for publications):")
    latex_eq = best_model.latex()
    print(latex_eq)
    
    print("\n🔢 Equation Complexity:", best_model.get_best().complexity)
    print("🎯 Equation R² Score:", sr_test_r2)
    
    # Save the models
    import pickle
    model_path_s1 = 'shear_stress_pysr_model_strategy1.pkl'
    model_path_s2 = 'shear_stress_pysr_model_strategy2.pkl'
    model_path_best = 'shear_stress_pysr_model_best.pkl'
    
    with open(model_path_s1, 'wb') as f:
        pickle.dump(model_aggressive, f)
    print(f"\n✓ Strategy 1 model saved: {model_path_s1}")
    
    with open(model_path_s2, 'wb') as f:
        pickle.dump(model_engineered, f)
    print(f"✓ Strategy 2 model saved: {model_path_s2}")
    
    with open(model_path_best, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model saved: {model_path_best}")
    
    # Save equations from all models
    for idx, model_obj in enumerate([model_aggressive, model_engineered], 1):
        equations_path = f'discovered_equations_strategy{idx}.csv'
        model_obj.equations_.to_csv(equations_path, index=False)
        print(f"✓ Strategy {idx} equations saved: {equations_path}")
    
    # Create prediction outputs
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'PySR_Prediction': y_test_pred_sr,
        'LinearReg_Prediction': y_test_pred_lr,
        'Ridge_Prediction': y_test_pred_ridge,
        'RandomForest_Prediction': y_test_pred_rf,
        'PySR_Error': np.abs(y_test.values - y_test_pred_sr),
    })
    predictions_path = 'model_predictions_comparison.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions comparison saved: {predictions_path}")
    
    # Create a Python function file from best model
    function_path = 'shear_stress_equation.py'
    with open(function_path, 'w') as f:
        f.write("# Auto-generated function from symbolic regression\n")
        f.write(f"# Best Strategy: {best_strategy}\n")
        f.write(f"# Test R² Score: {sr_test_r2:.6f}\n")
        f.write(f"# Test RMSE: {sr_test_rmse:.6f}\n\n")
        f.write("import numpy as np\n\n")
        
        f.write("def predict_shear_stress(RISE, COUN, ENER, DURATION, AMP):\n")
        f.write('    """\n')
        f.write('    Predict shear stress from acoustic emission parameters.\n')
        f.write('    (Auto-generated from Symbolic Regression)\n')
        f.write('    \n')
        f.write('    Parameters:\n')
        f.write('    -----------\n')
        f.write('    RISE : float or array\n')
        f.write('        Rise time of acoustic emission signal\n')
        f.write('    COUN : float or array\n')
        f.write('        Count of acoustic emission events\n')
        f.write('    ENER : float or array\n')
        f.write('        Energy of acoustic emission\n')
        f.write('    DURATION : float or array\n')
        f.write('        Duration of acoustic emission signal\n')
        f.write('    AMP : float or array\n')
        f.write('        Amplitude of acoustic emission signal\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('    --------\n')
        f.write('    shear_stress : float or array\n')
        f.write('        Predicted shear stress\n')
        f.write('    """\n')
        f.write(f"    # Discovered equation: {str(best_equation)}\n")
        f.write(f"    return {best_model.sympy().__str__()}\n")
        f.write("\ndef predict_with_confidence(RISE, COUN, ENER, DURATION, AMP):\n")
        f.write('    """Return prediction with uncertainty estimate"""\n')
        f.write("    pred = predict_shear_stress(RISE, COUN, ENER, DURATION, AMP)\n")
        f.write(f"    # Estimated std error from test set\n")
        f.write(f"    uncertainty = {sr_test_rmse:.6f}\n")
        f.write("    return pred, uncertainty\n")
    print(f"✓ Python function saved: {function_path}")
    
    print("\n" + "="*80)
    print("✅ IMPROVED SYMBOLIC REGRESSION COMPLETE!")
    print("="*80)
    
else:
    # Fallback: use simulated results if PySR not available
    print("\n" + "="*80)
    print("SIMULATED SYMBOLIC REGRESSION RESULT")
    print("="*80)
    print("\n⚠️  NOTE: This is a SIMULATED result since PySR is not installed.")
    print("    Install PySR to get ACTUAL discovered equations.")
    
    # Use polynomial regression as simulation
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    poly_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), 
                               LinearRegression())
    poly_model.fit(X_train, y_train)
    
    y_train_pred_sim = poly_model.predict(X_train)
    y_test_pred_sim = poly_model.predict(X_test)
    
    sim_train_r2 = r2_score(y_train, y_train_pred_sim)
    sim_test_r2 = r2_score(y_test, y_test_pred_sim)
    sim_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_sim))
    sim_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_sim))
    sim_train_mae = mean_absolute_error(y_train, y_train_pred_sim)
    sim_test_mae = mean_absolute_error(y_test, y_test_pred_sim)
    
    # Assign to SR variables for plotting
    y_train_pred_sr = y_train_pred_sim
    y_test_pred_sr = y_test_pred_sim
    sr_train_r2 = sim_train_r2
    sr_test_r2 = sim_test_r2
    sr_train_rmse = sim_train_rmse
    sr_test_rmse = sim_test_rmse
    sr_train_mae = sim_train_mae
    sr_test_mae = sim_test_mae
    
    print(f"\n📊 Simulated Performance (Polynomial Regression):")
    print(f"  Training R²: {sr_train_r2:.6f}")
    print(f"  Test R²:     {sr_test_r2:.6f}")
    print(f"  Training RMSE: {sr_train_rmse:.6f}")
    print(f"  Test RMSE:     {sr_test_rmse:.6f}")
    
    print("\n📝 Example of what PySR might discover:")
    print("  SHEAR_STRESS = 0.234 + 0.0012*RISE + 0.0089*COUN + ")
    print("                 0.0234*sqrt(ENER) + 0.0015*DURATION + 0.0345*AMP")
    print("\n  Or perhaps a more complex relationship:")
    print("  SHEAR_STRESS = 0.15 + 0.002*RISE*COUN + 0.034*log(ENER+1) + ")
    print("                 0.056*AMP^1.2 - 0.0008*DURATION")

# ============================================================================
# PART 6: COMPARISON VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CREATING COMPARISON VISUALIZATIONS")
print("="*80)

# Create comparison plots
fig = plt.figure(figsize=(18, 12))

# Plot 1: Linear Regression - Actual vs Predicted
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test, y_test_pred_lr, alpha=0.4, s=30, edgecolors='none')
min_val = min(y_test.min(), y_test_pred_lr.min())
max_val = max(y_test.max(), y_test_pred_lr.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Shear Stress', fontweight='bold', fontsize=11)
plt.ylabel('Predicted Shear Stress', fontweight='bold', fontsize=11)
plt.title('Linear Regression\nActual vs Predicted', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
textstr = f'R² = {lr_test_r2:.4f}\nRMSE = {lr_test_rmse:.4f}'
plt.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Symbolic Regression - Actual vs Predicted
ax2 = plt.subplot(2, 3, 2)
plt.scatter(y_test, y_test_pred_sr, alpha=0.4, s=30, edgecolors='none', color='green')
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Shear Stress', fontweight='bold', fontsize=11)
plt.ylabel('Predicted Shear Stress', fontweight='bold', fontsize=11)
if pysr_available:
    plt.title('Symbolic Regression\nActual vs Predicted', fontweight='bold', fontsize=12)
else:
    plt.title('Symbolic Regression (Simulated)\nActual vs Predicted', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
textstr = f'R² = {sr_test_r2:.4f}\nRMSE = {sr_test_rmse:.4f}'
plt.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Residuals comparison
ax3 = plt.subplot(2, 3, 3)
residuals_lr = y_test - y_test_pred_lr
residuals_sr = y_test - y_test_pred_sr
plt.scatter(y_test_pred_lr, residuals_lr, alpha=0.4, s=20, label='Linear Reg', edgecolors='none')
sr_label = 'Symbolic Reg' if pysr_available else 'Symbolic Reg (Sim)'
plt.scatter(y_test_pred_sr, residuals_sr, alpha=0.4, s=20, label=sr_label, 
           color='green', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Shear Stress', fontweight='bold', fontsize=11)
plt.ylabel('Residual (Actual - Predicted)', fontweight='bold', fontsize=11)
plt.title('Residual Plot Comparison', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Performance metrics comparison
ax4 = plt.subplot(2, 3, 4)
metrics = ['R² Score', 'RMSE', 'MAE']
lr_metrics = [lr_test_r2, lr_test_rmse, lr_test_mae]
sr_metrics = [sr_test_r2, sr_test_rmse, sr_test_mae]

x = np.arange(len(metrics))
width = 0.35
bars1 = plt.bar(x - width/2, lr_metrics, width, label='Linear Regression', alpha=0.8)
sr_label = 'Symbolic Reg' if pysr_available else 'Symbolic Reg (Sim)'
bars2 = plt.bar(x + width/2, sr_metrics, width, label=sr_label, 
               alpha=0.8, color='green')

plt.xlabel('Metric', fontweight='bold', fontsize=11)
plt.ylabel('Value', fontweight='bold', fontsize=11)
plt.title('Performance Metrics Comparison\n(Test Set)', fontweight='bold', fontsize=12)
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 5: Error distribution
ax5 = plt.subplot(2, 3, 5)
plt.hist(residuals_lr, bins=50, alpha=0.6, label='Linear Reg', edgecolor='black')
sr_label = 'Symbolic Reg' if pysr_available else 'Symbolic Reg (Sim)'
plt.hist(residuals_sr, bins=50, alpha=0.6, label=sr_label, 
        color='green', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Residual Value', fontweight='bold', fontsize=11)
plt.ylabel('Frequency', fontweight='bold', fontsize=11)
plt.title('Error Distribution', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 6: Feature importance (from linear regression)
ax6 = plt.subplot(2, 3, 6)
feature_importance = np.abs(lr_model.coef_)
feature_importance = feature_importance / feature_importance.sum() * 100
colors_feat = plt.get_cmap('viridis')(np.linspace(0, 1, len(input_features)))  # type: ignore[attr-defined]
bars = plt.barh(input_features, feature_importance, color=colors_feat, alpha=0.8, edgecolor='black')
plt.xlabel('Relative Importance (%)', fontweight='bold', fontsize=11)
plt.title('Feature Importance\n(Linear Regression Coefficients)', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, feature_importance):
    plt.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('shear_stress_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Comparison visualizations saved: shear_stress_comparison.png")

# ============================================================================
# PART 7: SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\n📊 Dataset Characteristics:")
print(f"  Total samples: {len(df)}")
print(f"  Input features: {len(input_features)}")
print(f"  Shear stress range: [{y.min():.4f}, {y.max():.4f}]")
print(f"  Training/Test split: {len(X_train)}/{len(X_test)}")

print("\n🔬 Model Performance Comparison:")
comparison_table = pd.DataFrame({
    'Model': ['Linear Regression', 'Symbolic Regression'],
    'Test R²': [f'{lr_test_r2:.6f}', f'{sr_test_r2:.6f}'],
    'Test RMSE': [f'{lr_test_rmse:.6f}', f'{sr_test_rmse:.6f}'],
    'Interpretation': ['Simple linear combination', 'Complex non-linear relationships' if pysr_available else 'Polynomial approximation (simulated)']
})
print(comparison_table.to_string(index=False))

print("\n💡 Key Insights:")
print(f"  1. Strongest correlation: {correlations.index[0]} (r = {correlations.iloc[0]:.4f})")
print(f"  2. Weakest correlation: {correlations.index[-1]} (r = {correlations.iloc[-1]:.4f})")
print(f"  3. Linear regression R²: {lr_test_r2:.4f}")
print(f"  4. Room for improvement with symbolic regression: {'Yes' if lr_test_r2 < 0.95 else 'Limited'}")

print("\n🎯 Next Steps:")
print("  1. Install PySR: pip install pysr")
print("  2. Uncomment the PySR code section in this script")
print("  3. Run symbolic regression (allow 10-30 minutes)")
print("  4. Analyze discovered equations for physical interpretation")
print("  5. Validate best equation on new experimental data")

print("\n📁 Generated Files:")
print("  ✓ shear_data_exploration.png - Data visualization")
print("  ✓ shear_stress_comparison.png - Model comparison")
print("  ✓ shear_stress_symbolic_regression.py - This script")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Sample_Index': range(len(y_test)),
    'Actual_Shear_Stress': y_test.values,
    'Linear_Reg_Prediction': y_test_pred_lr,
    'Linear_Reg_Error': residuals_lr,
    'Symbolic_Reg_Prediction': y_test_pred_sr,
    'Symbolic_Reg_Error': residuals_sr
})
results_df.to_csv('shear_stress_predictions.csv', index=False)
print("\n✓ Detailed predictions saved: shear_stress_predictions.csv")

print("\n🔬 Physical Interpretation Guidance:")
print("   When you get actual PySR results, look for:")
print("   • Power law relationships (e.g., ENER^0.5, AMP^1.2)")
print("   • Multiplicative interactions (e.g., RISE*COUN)")
print("   • Logarithmic relationships (e.g., log(ENER))")
print("   • These may reveal physical mechanisms in rock failure")

print("\n✅ Ready to run symbolic regression!")
if not pysr_available:
    print("\n⚠️  PySR was not installed, so simulated results were used.")
    print("   To get ACTUAL symbolic regression results:")
    print("   1. Run: pip install pysr")
    print("   2. Run this script again: python shear_stress_symbolic_regression.py")
else:
    print("\n🎉 Symbolic regression completed successfully!")
    print("   Check the generated files for discovered equations.")