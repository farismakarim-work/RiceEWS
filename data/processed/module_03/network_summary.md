# MODUL 3 - Network Inference Summary

Input: `/home/runner/work/RiceEWS/RiceEWS/data/processed/granger_results.json`
Output dir: `/home/runner/work/RiceEWS/RiceEWS/data/processed/module_03`
DAG enforcement: `True`

## Per Grade Summary

| Grade | Nodes | Edges | Edges Removed | DAG | Polytree | Top Leader | Leader Score | Visualization |
|---|---:|---:|---:|:---:|:---:|---|---:|---|
| low1 | 6 | 11 | 19 | True | False | M106 | 113.3490 | network_graph_low1.html |
| low2 | 6 | 14 | 16 | True | False | M106 | 194.6991 | network_graph_low2.html |
| med1 | 6 | 15 | 15 | True | False | M102 | 53.6304 | network_graph_med1.html |
| med2 | 6 | 10 | 20 | True | False | M103 | 35.7966 | network_graph_med2.html |

## Edges Removed for DAG Enforcement

| Grade | Source | Target |
|---|---|---|
| low1 | M101 | M102 |
| low1 | M103 | M101 |
| low1 | M103 | M102 |
| low1 | M104 | M101 |
| low1 | M102 | M101 |
| low1 | M104 | M102 |
| low1 | M104 | M103 |
| low1 | M105 | M101 |
| low1 | M103 | M104 |
| low1 | M105 | M102 |
| low1 | M103 | M105 |
| low1 | M106 | M101 |
| low1 | M106 | M102 |
| low1 | M106 | M103 |
| low1 | M104 | M105 |
| low1 | M106 | M104 |
| low1 | M105 | M103 |
| low1 | M105 | M104 |
| low1 | M105 | M106 |
| low2 | M102 | M101 |
| low2 | M103 | M101 |
| low2 | M102 | M103 |
| low2 | M102 | M104 |
| low2 | M102 | M105 |
| low2 | M102 | M106 |
| low2 | M104 | M101 |
| low2 | M103 | M104 |
| low2 | M105 | M101 |
| low2 | M103 | M105 |
| low2 | M106 | M101 |
| low2 | M106 | M103 |
| low2 | M106 | M104 |
| low2 | M105 | M103 |
| low2 | M105 | M104 |
| low2 | M105 | M106 |
| med1 | M102 | M101 |
| med1 | M103 | M101 |
| med1 | M103 | M102 |
| med1 | M104 | M101 |
| med1 | M104 | M102 |
| med1 | M103 | M104 |
| med1 | M103 | M105 |
| med1 | M103 | M106 |
| med1 | M105 | M101 |
| med1 | M104 | M105 |
| med1 | M104 | M106 |
| med1 | M102 | M105 |
| med1 | M106 | M101 |
| med1 | M106 | M102 |
| med1 | M106 | M105 |
| med2 | M102 | M101 |
| med2 | M101 | M102 |
| med2 | M101 | M103 |
| med2 | M104 | M101 |
| med2 | M101 | M104 |
| med2 | M105 | M101 |
| med2 | M105 | M102 |
| med2 | M101 | M105 |
| med2 | M106 | M101 |
| med2 | M106 | M102 |
| med2 | M103 | M101 |
| med2 | M102 | M103 |
| med2 | M104 | M102 |
| med2 | M104 | M103 |
| med2 | M104 | M105 |
| med2 | M104 | M106 |
| med2 | M102 | M105 |
| med2 | M106 | M103 |
| med2 | M106 | M105 |
| med2 | M103 | M105 |