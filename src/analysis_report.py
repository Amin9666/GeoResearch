#!/usr/bin/env python
"""
Analyze the symbolic regression results and validate equations.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load the results
equations_df = pd.read_csv('discovered_equations.csv')
predictions_df = pd.read_csv('shear_stress_predictions.csv')

print("="*90)
print("SYMBOLIC REGRESSION RESULTS - COMPREHENSIVE ANALYSIS")
print("="*90)

# 1. Equation Discovery Summary
print("\n📊 EQUATION DISCOVERY:")
print(f"   Total equations discovered: {len(equations_df)}")
print(f"   Complexity range: {int(equations_df['complexity'].min())} - {int(equations_df['complexity'].max())}")
print(f"   Loss range: {equations_df['loss'].min():.6f} - {equations_df['loss'].max():.6f}")

# 2. Best Equations
print("\n🏆 TOP 5 BEST EQUATIONS (by score - lower is better):")
top5 = equations_df.nsmallest(5, 'score')
for idx, (i, row) in enumerate(top5.iterrows(), 1):
    print(f"\n   {idx}. Complexity: {int(row['complexity'])}, Loss: {row['loss']:.6f}, Score: {row['score']:.2e}")
    print(f"      {row['equation']}")

# 3. Prediction Performance
print("\n\n📈 PREDICTION PERFORMANCE METRICS:")
actual = predictions_df['Actual_Shear_Stress'].values
lr_pred = predictions_df['Linear_Reg_Prediction'].values
sr_pred = predictions_df['Symbolic_Reg_Prediction'].values

actual_mean = actual.mean()
actual_std = actual.std()

print(f"\n   Actual Shear Stress Statistics:")
print(f"      Mean: {actual_mean:.6f}")
print(f"      Std:  {actual_std:.6f}")
print(f"      Range: [{actual.min():.6f}, {actual.max():.6f}]")

# Linear Regression
lr_rmse = np.sqrt(mean_squared_error(actual, lr_pred))
lr_mae = mean_absolute_error(actual, lr_pred)
lr_r2 = r2_score(actual, lr_pred)

# Symbolic Regression
sr_rmse = np.sqrt(mean_squared_error(actual, sr_pred))
sr_mae = mean_absolute_error(actual, sr_pred)
sr_r2 = r2_score(actual, sr_pred)

print(f"\n   Linear Regression (Baseline):")
print(f"      RMSE: {lr_rmse:.6f}")
print(f"      MAE:  {lr_mae:.6f}")
print(f"      R²:   {lr_r2:.6f}")

print(f"\n   Symbolic Regression:")
print(f"      RMSE: {sr_rmse:.6f}")
print(f"      MAE:  {sr_mae:.6f}")
print(f"      R²:   {sr_r2:.6f}")

# Improvements
rmse_imp = ((lr_rmse - sr_rmse) / lr_rmse) * 100
mae_imp = ((lr_mae - sr_mae) / lr_mae) * 100
r2_imp = ((sr_r2 - lr_r2) / abs(lr_r2)) * 100 if lr_r2 != 0 else 0

print(f"\n   🎯 IMPROVEMENT (Symbolic vs Linear):")
print(f"      RMSE: {rmse_imp:+.2f}%")
print(f"      MAE:  {mae_imp:+.2f}%")
print(f"      R²:   {r2_imp:+.2f}%")

# 4. Physical Interpretation
print("\n\n🔬 PHYSICAL INTERPRETATION OF TOP EQUATIONS:")

print("\n   Key Features in Discovered Equations:")
print("   • ENER (Energy): Appears in exponential form (0.996^ENER)")
print("     → Energy has weak exponential decay relationship")
print("   • AMP (Amplitude): Appears in square/sqrt/inv forms")
print("     → Strong non-linear relationship with shear stress")
print("   • DURATION: Appears in logarithmic forms")
print("     → Logarithmic damping of duration effect")
print("   • RISE: Combined with other parameters in interactions")
print("     → Multiplicative relationships suggest coupling effects")

print("\n   Equation Characteristics:")
for i, row in top5.iloc[:3].iterrows():
    eq = row['equation']
    print(f"\n   Eq {int(row['complexity'])}: {eq}")
    if 'log' in eq:
        print("      → Contains logarithmic relationships")
    if 'sqrt' in eq:
        print("      → Non-linear via square root")
    if '^' in eq or '**' in eq:
        print("      → Power law relationships present")

# 5. Recommendations
print("\n\n💡 RECOMMENDATIONS:")
print("   1. Model Performance: Currently predicting near mean value")
print("      → Consider feature scaling or engineering")
print("   2. Best Balance: Eq complexity 17-20 balance accuracy/simplicity")
print("   3. Feature Usage: Focus on AMP, ENER, DURATION interactions")
print("   4. Validation: Test on new experimental rock samples")
print("   5. Physics Check: Verify discovered relationships match rock mechanics theory")

print("\n" + "="*90)
