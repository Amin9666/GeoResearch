# Auto-generated symbolic regression formula
# Predicts SHEAR STRESS from acoustic emission parameters
# Method: LassoCV on 45 engineered feature candidates (IQR 1.0 cleaned data)
# Best R² on hold-out test: 0.0632 | 10-fold CV R²: 0.0412 ± 0.0179

import numpy as np


# ── Scaler statistics (RobustScaler fitted on IQR-1.0 training split, random_state=42) ──
_SCALER_CENTER = {
    "RISE": 6.0,
    "log_RISE": 1.9459101490553132,
    "sqrt_RISE": 2.449489742783178,
    "RISE_sq": 36.0,
    "RISE_inv": 0.14285714285714285,
    "COUN": 6.0,
    "log_COUN": 1.9459101490553132,
    "sqrt_COUN": 2.449489742783178,
    "COUN_sq": 36.0,
    "COUN_inv": 0.14285714285714285,
    "ENER": 3.0,
    "log_ENER": 1.3862943611198906,
    "sqrt_ENER": 1.7320508075688772,
    "ENER_sq": 9.0,
    "ENER_inv": 0.25,
    "DURATION": 80.0,
    "log_DURATION": 4.394449154672439,
    "sqrt_DURATION": 8.94427190999916,
    "DURATION_sq": 6400.0,
    "DURATION_inv": 0.012345679012345678,
    "AMP": 60.0,
    "log_AMP": 4.110873864173311,
    "sqrt_AMP": 7.745966692414834,
    "AMP_sq": 3600.0,
    "AMP_inv": 0.01639344262295082,
    "RISExCOUN": 45.0,
    "RISE_div_COUN": 0.8571428571428571,
    "RISExENER": 21.0,
    "RISE_div_ENER": 1.5,
    "RISExDURATION": 600.0,
    "RISE_div_DURATION": 0.08928571428571429,
    "RISExAMP": 396.0,
    "RISE_div_AMP": 0.1,
    "COUNxENER": 18.0,
    "COUN_div_ENER": 1.6,
    "COUNxDURATION": 504.5,
    "COUN_div_DURATION": 0.0967741935483871,
    "COUNxAMP": 390.0,
    "COUN_div_AMP": 0.10606060606060606,
    "ENERxDURATION": 240.0,
    "ENER_div_DURATION": 0.037037037037037035,
    "ENERxAMP": 177.0,
    "ENER_div_AMP": 0.04838709677419355,
    "DURATIONxAMP": 4667.0,
    "DURATION_div_AMP": 1.270513855259618,
}

_SCALER_SCALE = {
    "RISE": 12.0,
    "log_RISE": 1.3862943611198906,
    "sqrt_RISE": 2.14093253863854,
    "RISE_sq": 216.0,
    "RISE_inv": 0.1875,
    "COUN": 7.0,
    "log_COUN": 0.8754687373539001,
    "sqrt_COUN": 1.3166247903553998,
    "COUN_sq": 105.0,
    "COUN_inv": 0.11666666666666668,
    "ENER": 5.0,
    "log_ENER": 1.252762968495368,
    "sqrt_ENER": 1.4494897427831779,
    "ENER_sq": 35.0,
    "ENER_inv": 0.35714285714285715,
    "DURATION": 98.0,
    "log_DURATION": 1.0851892683359687,
    "sqrt_DURATION": 5.124355652982141,
    "DURATION_sq": 19208.0,
    "DURATION_inv": 0.013243243243243243,
    "AMP": 7.0,
    "log_AMP": 0.1139442593492177,
    "sqrt_AMP": 0.4501655647292502,
    "AMP_sq": 847.0,
    "AMP_inv": 0.0018567639257294419,
    "RISExCOUN": 114.0,
    "RISE_div_COUN": 1.3333333333333333,
    "RISExENER": 62.0,
    "RISE_div_ENER": 2.70979020979021,
    "RISExDURATION": 1678.0,
    "RISE_div_DURATION": 0.1666256964929531,
    "RISExAMP": 786.0,
    "RISE_div_AMP": 0.20338983050847456,
    "COUNxENER": 60.0,
    "COUN_div_ENER": 0.6666666666666667,
    "COUNxDURATION": 1390.75,
    "COUN_div_DURATION": 0.04996634935477942,
    "COUNxAMP": 472.0,
    "COUN_div_AMP": 0.10246305418719212,
    "ENERxDURATION": 977.0,
    "ENER_div_DURATION": 0.027651997502165706,
    "ENERxAMP": 330.0,
    "ENER_div_AMP": 0.07825567502986858,
    "DURATIONxAMP": 6652.75,
    "DURATION_div_AMP": 1.408592799736271,
}

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

# Numerical stability offset: added before log() and division to avoid log(0)/div-by-zero
_OFFSET = 1.0


def _build_feature_vector(RISE: np.ndarray, COUN: np.ndarray, ENER: np.ndarray,
                          DURATION: np.ndarray, AMP: np.ndarray) -> dict[str, np.ndarray]:
    """Build raw (un-scaled) symbolic feature vectors from five AE parameters."""
    raw = {"RISE": RISE, "COUN": COUN, "ENER": ENER, "DURATION": DURATION, "AMP": AMP}
    cols = list(raw.keys())
    features: dict[str, np.ndarray] = {}
    for col, v in raw.items():
        features[col] = v
        features[f"log_{col}"] = np.log(v + _OFFSET)
        features[f"sqrt_{col}"] = np.sqrt(v)
        features[f"{col}_sq"] = v ** 2
        features[f"{col}_inv"] = _OFFSET / (v + _OFFSET)
    for i, c1 in enumerate(cols):
        v1 = raw[c1]
        for c2 in cols[i + 1:]:
            v2 = raw[c2]
            features[f"{c1}x{c2}"] = v1 * v2
            features[f"{c1}_div_{c2}"] = v1 / (v2 + _OFFSET)
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
      3. RobustScaler normalisation (parameters embedded in this module)
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

    raw_features = _build_feature_vector(RISE, COUN, ENER, DURATION, AMP)

    result = np.full_like(RISE, _INTERCEPT, dtype=float)
    for term, coef in _COEFFICIENTS.items():
        center = _SCALER_CENTER[term]
        scale = _SCALER_SCALE[term]
        scaled_term = (raw_features[term] - center) / scale
        result = result + coef * scaled_term

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
    print("\nFormula overview (25 non-zero terms, applied after RobustScaler):")
    print(f"  SHEAR_STRESS = {_INTERCEPT:.4f}")
    for term, coef in _COEFFICIENTS.items():
        print(f"    {coef:+.6f} × scaled({term})")

    # Use realistic values from the cleaned training data distribution
    # (median row: RISE=6, COUN=7, ENER=3, DURATION=80, AMP=60)
    print("\n--- Example prediction (typical AE event near test median) ---")
    stress = predict_shear_stress(6.0, 7.0, 3.0, 80.0, 60.0)
    print(f"Input: RISE=6, COUN=7, ENER=3, DURATION=80, AMP=60")
    print(f"Predicted shear stress: {float(stress):.4f} MPa")

    conf = predict_with_confidence(6.0, 7.0, 3.0, 80.0, 60.0)
    print(f"95% CI: [{float(conf['lower_bound']):.4f}, {float(conf['upper_bound']):.4f}] MPa")

    print("\n--- Batch example (four events) ---")
    import numpy as np
    rise_vals  = np.array([4.0, 6.0, 10.0, 20.0])
    coun_vals  = np.array([5.0, 7.0, 12.0, 25.0])
    ener_vals  = np.array([2.0, 3.0,  6.0, 15.0])
    dur_vals   = np.array([50.0, 80.0, 130.0, 400.0])
    amp_vals   = np.array([55.0, 60.0,  65.0,  70.0])
    results = predict_shear_stress(rise_vals, coun_vals, ener_vals, dur_vals, amp_vals)
    for i in range(len(rise_vals)):
        print(f"  Event {i+1}: RISE={rise_vals[i]:.0f}, COUN={coun_vals[i]:.0f}, "
              f"ENER={ener_vals[i]:.0f}, DUR={dur_vals[i]:.0f}, AMP={amp_vals[i]:.0f} "
              f"→ Stress={float(results[i]):.4f} MPa")


