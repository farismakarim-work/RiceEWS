# RiceEWS
Rice Early Warning System - A research framework for identifying market leaders in Indonesian rice price networks using Granger Causality and DAG analysis

## Overview

RiceEWS implements the methodology of **Kinnear & Mazumdar (2023)** — *"Exact recovery of Granger causality graphs with unconditional pairwise tests"* (Network Science 11(3), pp. 431–457) — applied to Indonesian rice market price data.

## Methodology

### Pipeline

| Module | Description |
|--------|-------------|
| 1 | Data preprocessing (log transform, differencing, stationarity test) |
| 2 | Pairwise Granger causality testing with Benjamini-Hochberg FDR correction |
| 3 | Network inference via Algorithm 1 (transitive reduction / Pairwise Recovery) |
| 4 | Shock propagation simulation |
| 5 | Intervention analysis |
| 6 | Robustness validation |
| 7 | Policy recommendations |

### Key Methodological Notes

1. **Module 2 — Pairwise Ancestor Set W**: The output of pairwise Granger tests is the *ancestor set W*, not the direct-edge graph. W may contain transitive false positives (e.g., if 1→3→4, then 1→4 also appears in W). Module 3 eliminates these.

2. **Module 3 — Algorithm 1 (Pairwise Recovery)**: Direct edges are recovered via transitive reduction using the level-partition algorithm of Kinnear & Mazumdar (2023). Bidirectional pairs are excluded as confounders (Corollary 2.3).

3. **Strongly Causal Graph (SCG) / Polytree**: A DAG whose *undirected skeleton* is a forest (Definition 2.9). This is *not* equivalent to "each node has at most one parent" — multi-parent v-structures are valid SCG nodes.

4. **FDR Correction**: Benjamini-Hochberg correction is applied across all N(N-1) pairwise tests per grade, consistent with Section 3 of Kinnear & Mazumdar (2023).

5. **Preprocessing**: Linear detrending is disabled by default when order-1 differencing is active, since differencing already removes a linear trend (applying both causes over-differencing).

### Theoretical Assumptions

The exact-recovery guarantee (Theorem 2.2) requires:
- The true causal graph is a **strongly causal DAG** (Definition 2.9)
- **Assumption 2.1**: Noise covariance matrix Σ_v is diagonal (no instantaneous causality)
- No-cancellation (Assumption 2.1) and persistence (Definition 2.11) conditions

For real-world rice market data these assumptions may hold approximately rather than exactly. Results should be interpreted as **heuristic estimates** of the causal structure, not as guaranteed exact recovery unless the assumptions are verified.

## Reference

R. J. Kinnear and R. R. Mazumdar (2023). "Exact recovery of Granger causality graphs with unconditional pairwise tests." *Network Science* 11(3), 431–457. https://doi.org/10.1017/nws.2023.11
