# RiceEWS

RiceEWS implements Modules 1–3 of the Kinnear & Mazumdar (2023) workflow for Indonesian rice price data.

## Methodology

### Integrated causal discovery

The pipeline now uses **one integrated workflow** across all rice grades.

- **Module 1** preprocesses the raw Excel dataset(s).
- **Module 2** runs unconditional pairwise Granger tests over integrated nodes `(market_id, grade)` and produces **one ancestor matrix `W`**.
- **Module 3** recovers **one integrated direct-edge graph** from `W` using **Algorithm 1 only**.

No direct-edge thresholding and no post-recovery shortcut heuristic is used in Module 3.

### Node representation

Integrated nodes are encoded as:

- `M101_low1`
- `M101_low2`
- `M101_med1`
- `M101_med2`

This allows both:

- within-grade edges
- cross-grade edges

when supported by the data and the pairwise tests.

## Output folders

Modules 1–3 write to dedicated output folders under `data/processed/`:

- `data/processed/module_01/`
- `data/processed/module_02/`
- `data/processed/module_03/`

Key outputs:

- **Module 1**: `preprocessed_pilot_data.csv`, `preprocessed_pilot_data_report.json`
- **Module 2**: `granger_results.json`, `granger_pairwise.csv`, `granger_matrices.npz`, `granger_results.xlsx`, `granger_results.md`
- **Module 3**: `network_edges.csv`, `network_summary.csv`, `network_inference_results.json`, `market_leaders_*.csv`

## Multiple dataset support

`run_full_preprocessing_pipeline(..., input_file=...)` accepts:

- `None`
- `Path`
- `str`
- `list[Path]`
- `list[str]`

Behavior:

- `input_file=None` automatically loads all `*.xlsx` files in `data/raw/`
- all input files must have identical schemas
- duplicate `(date, market_id, grade)` rows are handled by `duplicate_strategy`

Supported duplicate strategies:

- `error` — raise an informative error
- `keep_first` — keep the first occurrence in input order
- `keep_last` — keep the last occurrence in input order

## Execution order

Run the modules in this order:

1. Module 1 preprocessing
2. Module 2 integrated pairwise Granger testing
3. Module 3 integrated graph recovery

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The requirements file includes all dependencies used by Modules 1–3, exports, visualizations, and tests.

## Reference

R. J. Kinnear and R. R. Mazumdar (2023). *Exact recovery of Granger causality graphs with unconditional pairwise tests.* Network Science 11(3), 431–457. https://doi.org/10.1017/nws.2023.11
