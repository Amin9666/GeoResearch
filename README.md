# GeoResearch

This repo contains a shear-stress modeling workflow using symbolic regression and baseline ML models.

## Project structure

- `src/` → Python source scripts
- `data/raw/` → raw datasets
- `outputs/csv/` → generated tabular results
- `outputs/models/` → serialized model artifacts (`.pkl`)
- `outputs/figures/` → generated plots
- `outputs/logs/` → run logs
- `docs/reports/` → archived reports and summaries

## Quick setup (local virtual environment)

```zsh
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Run

```zsh
python src/shear_stress_symbolic_regression.py
```

## Notes
- `pysr` is included in `requirements.txt`. It may require Julia installed on your system to run symbolic regression.
- A compatibility symlink `Shear_Data_15.csv` is kept at repo root so existing scripts that read this relative path still work.

## PySR / Julia troubleshooting (macOS)

If you see an error like `CalledProcessError` from a `.../pyjuliapkg/.../julia` command, Julia auto-provisioning failed.

```zsh
brew install julia
export PYTHON_JULIAPKG_EXE=$(which julia)
export PYTHON_JULIACALL_EXE=$(which julia)
python src/shear_stress_symbolic_regression.py
```

If you want to run the full pipeline without PySR for now (uses simulated symbolic-regression output):

```zsh
GEORESEARCH_SKIP_PYSR=1 python src/shear_stress_symbolic_regression.py
```
