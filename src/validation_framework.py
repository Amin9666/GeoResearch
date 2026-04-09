#!/usr/bin/env python
"""
Validation and Testing Framework for Shear Stress Prediction
=============================================================

This script provides tools to validate the discovered symbolic regression
equation on new experimental data and assess prediction accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import the prediction function
from shear_stress_equation import predict_shear_stress, predict_with_confidence

print("="*80)
print("SYMBOLIC REGRESSION VALIDATION FRAMEWORK")
print("="*80)

# ============================================================================
# PART 1: VALIDATION ON EXISTING TEST SET
# ============================================================================

print("\n" + "="*80)
print("PART 1: VALIDATION ON EXISTING TEST SET")
print("="*80)

# Load predictions
predictions_df = pd.read_csv('shear_stress_predictions.csv')
actual_stress = predictions_df['Actual_Shear_Stress'].values
sr_predictions = predictions_df['Symbolic_Reg_Prediction'].values

# Recalculate with our function
computed_predictions = predict_shear_stress(
    predictions_df['RISE'].values if 'RISE' in predictions_df.columns 
    else np.ones_like(actual_stress) * 12.5,
    predictions_df['COUN'].values if 'COUN' in predictions_df.columns 
    else np.ones_like(actual_stress) * 450,
    predictions_df['ENER'].values if 'ENER' in predictions_df.columns 
    else np.ones_like(actual_stress) * 1250,
    predictions_df['DURATION'].values if 'DURATION' in predictions_df.columns 
    else np.ones_like(actual_stress) * 85.3,
    predictions_df['AMP'].values if 'AMP' in predictions_df.columns 
    else np.ones_like(actual_stress) * 45.2,
)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual_stress, sr_predictions))
mae = mean_absolute_error(actual_stress, sr_predictions)
r2 = r2_score(actual_stress, sr_predictions)

print(f"\n📊 Test Set Performance:")
print(f"   RMSE (Root Mean Squared Error): {rmse:.6f} MPa")
print(f"   MAE (Mean Absolute Error):      {mae:.6f} MPa")
print(f"   R² Score:                        {r2:.6f}")

# Error statistics
errors = np.abs(actual_stress - sr_predictions)
print(f"\n📈 Prediction Error Statistics:")
print(f"   Mean Error:     {np.mean(errors):.6f} MPa")
print(f"   Std Dev Error:  {np.std(errors):.6f} MPa")
print(f"   Min Error:      {np.min(errors):.6f} MPa")
print(f"   Max Error:      {np.max(errors):.6f} MPa")
print(f"   Median Error:   {np.median(errors):.6f} MPa")

# Error distribution
within_1std = np.sum(errors <= rmse) / len(errors) * 100
within_2std = np.sum(errors <= 2*rmse) / len(errors) * 100

print(f"\n📉 Error Distribution:")
print(f"   Predictions within 1σ: {within_1std:.1f}%")
print(f"   Predictions within 2σ: {within_2std:.1f}%")

# ============================================================================
# PART 2: EXAMPLE - SYNTHETIC NEW EXPERIMENTAL DATA
# ============================================================================

print("\n" + "="*80)
print("PART 2: VALIDATION ON NEW EXPERIMENTAL DATA (SYNTHETIC)")
print("="*80)

# Simulate new experimental samples
np.random.seed(42)
n_samples = 50

new_data = pd.DataFrame({
    'RISE': np.random.uniform(5, 30, n_samples),
    'COUN': np.random.uniform(200, 800, n_samples),
    'ENER': np.random.uniform(500, 2500, n_samples),
    'DURATION': np.random.uniform(40, 150, n_samples),
    'AMP': np.random.uniform(30, 60, n_samples),
})

# Generate synthetic "true" values using a simple underlying model
# (In practice, these would be measured from new experiments)
underlying_stress = (
    0.5 * new_data['AMP'] + 
    0.1 * np.log(new_data['ENER'] + 1) + 
    0.2 * np.sqrt(new_data['DURATION']) - 
    0.05 * new_data['RISE']
)
new_data['True_Stress'] = underlying_stress + np.random.normal(0, 0.3, n_samples)

# Make predictions
new_data['Predicted_Stress'] = predict_shear_stress(
    new_data['RISE'].values,
    new_data['COUN'].values,
    new_data['ENER'].values,
    new_data['DURATION'].values,
    new_data['AMP'].values,
)

# Calculate validation metrics
new_rmse = np.sqrt(mean_squared_error(new_data['True_Stress'], new_data['Predicted_Stress']))
new_mae = mean_absolute_error(new_data['True_Stress'], new_data['Predicted_Stress'])
new_r2 = r2_score(new_data['True_Stress'], new_data['Predicted_Stress'])

print(f"\n🔬 New Data Validation (50 synthetic samples):")
print(f"   RMSE: {new_rmse:.6f} MPa")
print(f"   MAE:  {new_mae:.6f} MPa")
print(f"   R²:   {new_r2:.6f}")

print(f"\n   Sample Predictions:")
print(f"   {'Index':<8}{'RISE':<10}{'AMP':<10}{'True':<10}{'Predicted':<12}{'Error':<10}")
print("   " + "-"*58)
for i in range(min(10, len(new_data))):
    row = new_data.iloc[i]
    error = row['True_Stress'] - row['Predicted_Stress']
    print(f"   {i+1:<8}{row['RISE']:<10.2f}{row['AMP']:<10.2f}{row['True_Stress']:<10.2f}{row['Predicted_Stress']:<12.2f}{error:<10.2f}")

# ============================================================================
# PART 3: HOW TO USE WITH NEW EXPERIMENTAL DATA
# ============================================================================

print("\n" + "="*80)
print("PART 3: USAGE GUIDE FOR NEW EXPERIMENTAL DATA")
print("="*80)

print("""
To validate on your own new experimental data:

1. PREPARE YOUR DATA:
   - Load measured acoustic emission parameters
   - Ensure features match: RISE, COUN, ENER, DURATION, AMP
   - Units should match training data (see feature descriptions)
   
2. MAKE PREDICTIONS:
   
   from shear_stress_equation import predict_shear_stress
   import pandas as pd
   
   # Load your data
   new_data = pd.read_csv('your_new_experiments.csv')
   
   # Generate predictions
   predictions = predict_shear_stress(
       new_data['RISE'],
       new_data['COUN'],
       new_data['ENER'],
       new_data['DURATION'],
       new_data['AMP']
   )
   
3. EVALUATE PREDICTIONS:
   
   from sklearn.metrics import r2_score, mean_squared_error
   
   if 'Shear_Stress' in new_data.columns:
       rmse = np.sqrt(mean_squared_error(new_data['Shear_Stress'], predictions))
       r2 = r2_score(new_data['Shear_Stress'], predictions)
       print(f'RMSE: {rmse:.4f}, R²: {r2:.4f}')

4. WITH CONFIDENCE INTERVALS:
   
   from shear_stress_equation import predict_with_confidence
   
   result = predict_with_confidence(
       rise=12.5, coun=450, ener=1250, 
       duration=85.3, amp=45.2
   )
   print(f"Prediction: {result['prediction']:.2f} ± {result['model_rmse']:.2f} MPa")
""")

# ============================================================================
# PART 4: MODEL LIMITATIONS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 4: MODEL LIMITATIONS & RECOMMENDATIONS")
print("="*80)

print("""
⚠️  LIMITATIONS:

1. GENERALIZATION:
   - Model was trained on rock shear test data from specific conditions
   - May not generalize to different rock types or testing configurations
   - Out-of-distribution predictions should be treated with caution

2. EQUATION FORM:
   - Discovered equation uses complex non-linear combinations
   - May not align with established rock mechanics theory
   - Physical interpretation unclear (black-box relationship)

3. DATA REQUIREMENTS:
   - All 5 input features (RISE, COUN, ENER, DURATION, AMP) must be provided
   - Features must be in same units as training data
   - No missing values allowed

4. PREDICTION ACCURACY:
   - Current R² ≈ -11.29 (worse than baseline linear model)
   - Large prediction errors (RMSE ≈ 3.53 MPa)
   - Suggests features have limited predictive power for shear stress

💡 RECOMMENDATIONS FOR IMPROVEMENT:

1. FEATURE ENGINEERING:
   ✓ Create interaction terms: AMP*RISE, ENER/DURATION, etc.
   ✓ Add domain-specific features: acoustic quality factor, peak frequency
   ✓ Normalize/scale features to [-1,1] or [0,1] range
   ✓ Create ratio features: ENER/AMP, DURATION/RISE

2. DATA COLLECTION:
   ✓ Increase training data size (more diverse rock samples)
   ✓ Include additional parameters: confining pressure, temperature
   ✓ Ensure consistent measurement conditions
   ✓ Validate measurement calibration

3. MODEL REFINEMENT:
   ✓ Try different PySR parameters:
     - Increase population size and iterations
     - Add more unary operators (sin, cos, exp, etc.)
     - Reduce maxsize complexity constraint
   ✓ Use multi-objective optimization (accuracy vs complexity)
   ✓ Implement ensemble models combining multiple equations

4. VALIDATION STRATEGY:
   ✓ Use k-fold cross-validation (not just train/test split)
   ✓ Test on rock types not in training set
   ✓ Validate against published experimental data
   ✓ Compare with physics-based models

5. PHYSICS-INFORMED APPROACH:
   ✓ Incorporate known relationships from rock mechanics
   ✓ Constrain solution space using domain knowledge
   ✓ Use symbolic regression with physical constraints
""")

print("\n" + "="*80)
print("VALIDATION FRAMEWORK COMPLETE")
print("="*80)

# Save new test data
new_data.to_csv('validation_predictions_new_data.csv', index=False)
print(f"\n✓ New validation data saved to: validation_predictions_new_data.csv")
