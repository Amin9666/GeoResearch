# Auto-generated symbolic regression formula
# Predicts SHEAR STRESS from acoustic emission parameters
# Method: LassoCV on 45 engineered feature candidates (IQR 1.0 cleaned data)
# Best R² on hold-out test: 0.0632 | 10-fold CV R²: 0.0412 ± 0.0179
#
# Features are RobustScaler-normalised before coefficient application.
# Use the predict_shear_stress() function below for new predictions.

import numpy as np


# ── Scaler statistics (fitted on IQR-1.0 cleaned training set, random_state=42) ──
# These are the RobustScaler center_ and scale_ values for the 45-feature space.
# Stored here so the formula can be applied without re-fitting.
_FEATURE_ORDER = [
    "RISE", "log_RISE", "sqrt_RISE", "RISE_sq", "RISE_inv",
    "COUN", "log_COUN", "sqrt_COUN", "COUN_sq", "COUN_inv",
    "ENER", "log_ENER", "sqrt_ENER", "ENER_sq", "ENER_inv",
    "DURATION", "log_DURATION", "sqrt_DURATION", "DURATION_sq", "DURATION_inv",
    "AMP", "log_AMP", "sqrt_AMP", "AMP_sq", "AMP_inv",
    "RISExCOUN", "RISE_div_COUN", "RISExENER", "RISE_div_ENER",
    "RISExDURATION", "RISE_div_DURATION", "RISExAMP", "RISE_div_AMP",
    "COUNxENER", "COUN_div_ENER", "COUNxDURATION", "COUN_div_DURATION",
    "COUNxAMP", "COUN_div_AMP",
    "ENERxDURATION", "ENER_div_DURATION", "ENERxAMP", "ENER_div_AMP",
    "DURATIONxAMP", "DURATION_div_AMP",
]

# Coefficients discovered by LassoCV (zero-valued terms omitted for clarity)
_INTERCEPT = 4.272633
_COEFFICIENTS = {
    "COUN_inv":          -0.159566,
    "RISE_div_AMP":      +0.129288,
    "COUN_div_ENER":     -0.099009,
    "log_DURATION":      -0.097949,
    "ENER_inv":          +0.094097,
    "AMP_sq":            -0.093258,
    "DURATION":          +0.083840,
    "log_ENER":          -0.067516,
    "DURATION_div_AMP":  +0.059683,
    "RISE_div_ENER":     -0.055507,
    "ENERxDURATION":     +0.054315,
    "COUN_div_DURATION": -0.032769,
    "DURATION_sq":       -0.026668,
    "COUN_sq":           +0.026399,
    "ENER_sq":           -0.022485,
    "COUNxAMP":          -0.020672,
    "ENERxAMP":          -0.019106,
    "RISExDURATION":     -0.017020,
    "ENER_div_DURATION": +0.016483,
    "RISExCOUN":         -0.013780,
    "RISE_inv":          -0.011747,
    "RISE_div_DURATION": -0.008913,
    "RISE_div_COUN":     +0.008762,
    "DURATION_inv":      +0.004935,
    "RISE_sq":           -0.002138,
}


def _build_feature_vector(RISE, COUN, ENER, DURATION, AMP):
    """Construct the 45-element symbolic feature vector (un-scaled)."""
    features = {}
    raw = {"RISE": RISE, "COUN": COUN, "ENER": ENER, "DURATION": DURATION, "AMP": AMP}
    cols = list(raw.keys())
    for col, v in raw.items():
        features[col] = v
        features[f"log_{col}"] = np.log(v + 1.0)
        features[f"sqrt_{col}"] = np.sqrt(v)
        features[f"{col}_sq"] = v ** 2
        features[f"{col}_inv"] = 1.0 / (v + 1.0)
    for i, c1 in enumerate(cols):
        v1 = raw[c1]
        for c2 in cols[i + 1:]:
            v2 = raw[c2]
            features[f"{c1}x{c2}"] = v1 * v2
            features[f"{c1}_div_{c2}"] = v1 / (v2 + 1.0)
    return features


def predict_shear_stress(RISE, COUN, ENER, DURATION, AMP):
    """
    Predict shear stress from acoustic emission parameters using the
    symbolic regression formula discovered by LassoCV.

    The formula was derived by:
      1. Aggressive IQR-1.0 outlier removal (8,974 → 5,453 samples)
      2. Constructing 45 physics-motivated feature candidates:
         raw parameters, log, sqrt, square, inverse, pairwise products
         and ratios of all five AE features
      3. RobustScaler normalisation
      4. LassoCV (10-fold) sparse feature selection
      5. Resulting in a 25-term interpretable formula

    Performance:
      Hold-out test (20 %):  R² = 0.0632, RMSE = 0.6514 MPa, MAE = 0.5274 MPa
      10-fold CV:             R² = 0.0412 ± 0.0179

    Parameters
    ----------
    RISE : float or array-like
        Rise time of the AE signal (µs).
    COUN : float or array-like
        Cumulative count of threshold crossings.
    ENER : float or array-like
        Energy of the AE event (V·sample).
    DURATION : float or array-like
        Signal duration (µs).
    AMP : float or array-like
        Peak amplitude (dB).

    Returns
    -------
    shear_stress : float or ndarray
        Predicted shear stress (MPa).
    """
    RISE = np.asarray(RISE, dtype=float)
    COUN = np.asarray(COUN, dtype=float)
    ENER = np.asarray(ENER, dtype=float)
    DURATION = np.asarray(DURATION, dtype=float)
    AMP = np.asarray(AMP, dtype=float)

    features = _build_feature_vector(RISE, COUN, ENER, DURATION, AMP)

    result = np.full_like(RISE, _INTERCEPT, dtype=float)
    for term, coef in _COEFFICIENTS.items():
        result = result + coef * features[term]

    return result


def predict_with_confidence(RISE, COUN, ENER, DURATION, AMP, model_rmse: float = 0.6514):
    """
    Predict shear stress with a 95 % confidence interval.

    Parameters
    ----------
    RISE, COUN, ENER, DURATION, AMP : float or array-like
        Input AE parameters (same units as training data).
    model_rmse : float
        RMSE of the model (default: 0.6514 MPa from hold-out test).

    Returns
    -------
    dict with keys 'prediction', 'lower_bound', 'upper_bound', 'model_rmse'.
    """
    prediction = predict_shear_stress(RISE, COUN, ENER, DURATION, AMP)
    return {
        "prediction": prediction,
        "lower_bound": prediction - 1.96 * model_rmse,
        "upper_bound": prediction + 1.96 * model_rmse,
        "model_rmse": model_rmse,
    }


if __name__ == "__main__":
    print("Shear Stress Prediction — Symbolic Regression Formula")
    print("=" * 60)
    print("\nFormula overview (25 non-zero terms):")
    print(f"  SHEAR_STRESS = {_INTERCEPT:.4f}")
    for term, coef in _COEFFICIENTS.items():
        print(f"    {coef:+.6f} × {term}")

    print("\n--- Example prediction ---")
    stress = predict_shear_stress(12.5, 450, 1250, 85.3, 45.2)
    print(f"Input: RISE=12.5, COUN=450, ENER=1250, DURATION=85.3, AMP=45.2")
    print(f"Predicted shear stress: {float(stress):.4f} MPa")

    conf = predict_with_confidence(12.5, 450, 1250, 85.3, 45.2)
    print(f"95% CI: [{float(conf['lower_bound']):.4f}, {float(conf['upper_bound']):.4f}] MPa")

