# MODUL 3 - Network Inference Summary

**Algorithm**: Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery

> **Methodology Notes:**
> - Edges are recovered via *transitive reduction* (Algorithm 1), not direct
>   thresholding of pairwise test results.
> - Bidirectional pairs are excluded as confounders (Corollary 2.3).
> - `is_polytree` checks whether the DAG's undirected skeleton is a forest
>   (correct Definition 2.9), not whether each node has ≤1 parent.
> - The exact-recovery guarantee requires the Strongly Causal Graph (SCG)
>   assumption.  Real-world results are heuristic estimates.

Input: `/home/runner/work/RiceEWS/RiceEWS/data/processed/granger_results.json`
Output dir: `/home/runner/work/RiceEWS/RiceEWS/data/processed/module_03`
DAG enforcement: `True`

## Per Grade Summary

| Grade | Nodes | E (edges) | W (ancestors) | Bidir excluded | Transitive removed | DAG cycles removed | DAG | Polytree | Top Leader | Leader Score |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|
| low1 | 0 | 0 | 0 | 15 | 0 | 0 | True | True | None | 0.0000 |
| low2 | 0 | 0 | 0 | 15 | 0 | 0 | True | True | None | 0.0000 |
| med1 | 0 | 0 | 0 | 15 | 0 | 0 | True | True | None | 0.0000 |
| med2 | 0 | 0 | 0 | 15 | 0 | 0 | True | True | None | 0.0000 |

## Excluded Bidirectional Pairs (Confounders, Corollary 2.3)

| Grade | Market A | Market B |
|---|---|---|
| low1 | M102 | M103 |
| low1 | M105 | M106 |
| low1 | M101 | M106 |
| low1 | M102 | M106 |
| low1 | M103 | M105 |
| low1 | M103 | M104 |
| low1 | M104 | M106 |
| low1 | M101 | M102 |
| low1 | M103 | M106 |
| low1 | M104 | M105 |
| low1 | M102 | M104 |
| low1 | M101 | M105 |
| low1 | M101 | M104 |
| low1 | M102 | M105 |
| low1 | M101 | M103 |
| low2 | M102 | M103 |
| low2 | M105 | M106 |
| low2 | M101 | M106 |
| low2 | M102 | M106 |
| low2 | M103 | M105 |
| low2 | M103 | M104 |
| low2 | M104 | M106 |
| low2 | M101 | M102 |
| low2 | M103 | M106 |
| low2 | M104 | M105 |
| low2 | M102 | M104 |
| low2 | M101 | M105 |
| low2 | M101 | M104 |
| low2 | M102 | M105 |
| low2 | M101 | M103 |
| med1 | M102 | M103 |
| med1 | M105 | M106 |
| med1 | M101 | M106 |
| med1 | M102 | M106 |
| med1 | M103 | M105 |
| med1 | M103 | M104 |
| med1 | M104 | M106 |
| med1 | M101 | M102 |
| med1 | M103 | M106 |
| med1 | M104 | M105 |
| med1 | M102 | M104 |
| med1 | M101 | M105 |
| med1 | M101 | M104 |
| med1 | M102 | M105 |
| med1 | M101 | M103 |
| med2 | M102 | M103 |
| med2 | M105 | M106 |
| med2 | M101 | M106 |
| med2 | M102 | M106 |
| med2 | M103 | M105 |
| med2 | M103 | M104 |
| med2 | M104 | M106 |
| med2 | M101 | M102 |
| med2 | M103 | M106 |
| med2 | M104 | M105 |
| med2 | M102 | M104 |
| med2 | M101 | M105 |
| med2 | M101 | M104 |
| med2 | M102 | M105 |
| med2 | M101 | M103 |