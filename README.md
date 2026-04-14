# GeoResearch

Machine learning and symbolic regression models applied to rock shear stress prediction and iris flower classification/regression.

## Project structure

```
GeoResearch/
├── data/
│   └── Shear_Data_15.csv          # Acoustic emission dataset for rock shear stress
├── rock_shear_stress/
│   ├── symbolic_regression.py     # Symbolic regression (LassoCV on engineered features)
│   ├── comparison.py              # Linear Regression, MLP, Random Forest & Symbolic Regression
│   ├── shear_stress_equation.py   # Auto-generated symbolic formula module
│   └── comparison_graphs/         # Pre-generated comparison plots
├── iris/
│   ├── symbolic_regression.py     # Symbolic regression on the Iris dataset
│   ├── linear_regression.py       # Linear regression on the Iris dataset
│   ├── mlp.py                     # MLP regressor on the Iris dataset
│   ├── random_forest.py           # Random forest on the Iris dataset
│   └── comparison.py              # Master comparison of all four algorithms
├── requirements.txt
└── README.md
```

## Quick setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Rock shear stress – all four algorithms + comparison plots
python rock_shear_stress/comparison.py

# Rock shear stress – symbolic regression only
python rock_shear_stress/symbolic_regression.py

# Iris – individual algorithms
python iris/linear_regression.py
python iris/mlp.py
python iris/random_forest.py
python iris/symbolic_regression.py

# Iris – master comparison
python iris/comparison.py
```

## Notes
- `pysr` is included in `requirements.txt`. It may require Julia installed on your system to run symbolic regression.

## PySR / Julia troubleshooting (macOS)

If you see an error like `CalledProcessError` from a `.../pyjuliapkg/.../julia` command, Julia auto-provisioning failed.

```bash
brew install julia
export PYTHON_JULIAPKG_EXE=$(which julia)
export PYTHON_JULIACALL_EXE=$(which julia)
python rock_shear_stress/symbolic_regression.py
```

If you want to run the pipeline without PySR (uses simulated symbolic-regression output):

```bash
GEORESEARCH_SKIP_PYSR=1 python rock_shear_stress/symbolic_regression.py
```
