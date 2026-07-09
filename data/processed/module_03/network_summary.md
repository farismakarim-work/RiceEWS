# MODUL 3 - Network Inference Summary

**Algorithm**: Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery

> One integrated graph is recovered over nodes `(market_id, grade)`.
> No direct edge thresholding or post-Algorithm-1 cycle-breaking heuristic is applied.
> Ancestor relations that still form directed cycles are dropped before Algorithm 1
> because they violate the ancestor partial-order assumption required by the paper.

Input: `/home/runner/work/RiceEWS/RiceEWS/data/processed/module_02/granger_results.json`
Output dir: `/home/runner/work/RiceEWS/RiceEWS/data/processed/module_03`

## Per Grade Summary

| Grade | Nodes | Incident Edges | Out Edges | In Edges | Cross-Grade Edges | DAG | Polytree | Top Leader |
|---|---:|---:|---:|---:|---:|:---:|:---:|---|
| low1 | 6 | 3 | 1 | 2 | 3 | True | False | M106_low1 |
| low2 | 6 | 10 | 2 | 10 | 8 | True | False | M106_low2 |
| med1 | 6 | 13 | 13 | 3 | 10 | True | False | M101_med1 |
| med2 | 6 | 7 | 3 | 4 | 7 | True | False | M106_med2 |