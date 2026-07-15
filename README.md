# RiceEWS

RiceEWS implements Modules 1–3 of Kinnear & Mazumdar (2023) for integrated rice-price causality analysis.

## Configuration philosophy

All configurable runtime parameters are centralized in `src/config.py`.

Sections include:
- Project / Input / Output / Execution
- Module 1 / Module 2 / Module 3
- Visualization / Logging / CLI

`run_pipeline.py` and module entry points consume these values directly.

## AUTO / MANUAL modes

`src/config.py` supports AUTO and MANUAL controls, including:
- Lag selection (`M2_LAG_SELECTION_MODE`)
- Significance level (`M2_SIGNIFICANCE_MODE`)
- Benjamini–Hochberg toggle (`M2_APPLY_BH_CORRECTION`)
- Differencing workflow (`M1_DIFFERENCING_MODE`)
- Duplicate handling (`M1_DUPLICATE_STRATEGY`)

## Preprocessing workflow (Module 1)

Module 1 is assumption-driven and configurable:

1. Load raw prices
2. Apply dataset filters (market / grade / date)
3. Optional missing-value handling (`M1_MISSING_VALUE_MODE`, default `disabled`)
4. Optional outlier handling (`M1_OUTLIER_MODE`)
5. Optional log transform (`M1_LOG_TRANSFORM`)
6. Stationarity testing (`M1_STATIONARITY_TEST`: `ADF`, `KPSS`, `ADF_KPSS`)
7. If non-stationary, difference and retest up to `M1_MAX_DIFFERENCING_ORDER`
8. Optional standardization (`M1_STANDARDIZATION_ENABLED`)

Module 1 no longer blindly differences every series.

## Output reproducibility / overwrite behavior

Default execution is fresh and reproducible:
- modules execute even when output files already exist
- outputs are regenerated and overwritten with the same filenames
- modules overwrite only their own output folders

Skip behavior is disabled by default (`SKIP_EXISTING_OUTPUTS = False`) and only used when explicitly enabled.

## CLI filters

Supported in:
- `run_pipeline.py`
- `run_module1.py`
- `run_module2.py`
- `run_module3.py`

Options:
- `--grade` (single, repeated, comma-separated)
- `--market` (single, range, comma-separated)
- `--date-range` (`YYYY-MM-DD:YYYY-MM-DD`)

Examples:

```bash
python run_pipeline.py --all --grade low1 --market 101-120
python run_pipeline.py --module 2 --grade low1,med1 --market 101,103,108
python run_pipeline.py --module 1 --date-range 2021-01-01:2023-12-31
```

## Output folders

- `data/processed/module_01/`
- `data/processed/module_02/`
- `data/processed/module_03/`

## Requirements

```bash
pip install -r requirements.txt
```

## Reference

R. J. Kinnear and R. R. Mazumdar (2023). *Exact recovery of Granger causality graphs with unconditional pairwise tests.* Network Science 11(3), 431–457.
