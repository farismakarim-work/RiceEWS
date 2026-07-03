# MODUL 5 - Intervention Analysis Summary

Num propagation steps: 5  
Shock magnitude: 1.0  
Node attenuation factor: 0.0  
Edge attenuation factor: 0.0  
Top-k control: 1

---

## Grade: low1

**Baseline cumulative impact:** 30.000000

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M101 | 5.000000 | 6 |
| 2 | M102 | 5.000000 | 6 |
| 3 | M103 | 5.000000 | 6 |
| 4 | M104 | 5.000000 | 6 |
| 5 | M105 | 5.000000 | 6 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M103 | 10.188085 |
| 2 | M105 | 8.711962 |
| 3 | M106 | 6.797022 |
| 4 | M104 | 2.585155 |
| 5 | M102 | 0.958852 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 22.0230 | 1.0000 |
|  | topk_source_control | 22.0230 | 1.0000 |
|  | edge_attenuation | -0.0000 | 0.0000 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 23.393088
- Impact reduction: **22.0230%**
- Affected nodes reduction (avg): 1.0000

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 30.000000
- Impact reduction: **-0.0000%**
- Affected nodes reduction (avg): 0.0000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 23.393088
- Impact reduction: **22.0230%**
- Affected nodes reduction (avg): 1.0000

---

## Grade: low2

**Baseline cumulative impact:** 30.000000

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M106 | 5.000000 | 6 |
| 2 | M101 | 5.000000 | 6 |
| 3 | M102 | 5.000000 | 6 |
| 4 | M103 | 5.000000 | 6 |
| 5 | M104 | 5.000000 | 6 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M103 | 9.533674 |
| 2 | M105 | 8.248088 |
| 3 | M106 | 5.695048 |
| 4 | M102 | 2.482234 |
| 5 | M104 | 2.338091 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 47.4695 | 1.0000 |
|  | topk_source_control | 47.4695 | 1.0000 |
|  | edge_attenuation | 0.0000 | 0.0000 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 15.759145
- Impact reduction: **47.4695%**
- Affected nodes reduction (avg): 1.0000

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 30.000000
- Impact reduction: **0.0000%**
- Affected nodes reduction (avg): 0.0000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 15.759145
- Impact reduction: **47.4695%**
- Affected nodes reduction (avg): 1.0000

---

## Grade: med1

**Baseline cumulative impact:** 30.000000

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M106 | 5.000000 | 6 |
| 2 | M104 | 5.000000 | 6 |
| 3 | M101 | 5.000000 | 6 |
| 4 | M102 | 5.000000 | 6 |
| 5 | M103 | 5.000000 | 6 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M103 | 10.240528 |
| 2 | M102 | 7.325890 |
| 3 | M104 | 6.596629 |
| 4 | M106 | 2.923399 |
| 5 | M101 | 1.911168 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 35.6224 | 1.0000 |
|  | topk_source_control | 35.6224 | 1.0000 |
|  | edge_attenuation | -0.0000 | 0.0000 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 19.313278
- Impact reduction: **35.6224%**
- Affected nodes reduction (avg): 1.0000

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 30.000000
- Impact reduction: **-0.0000%**
- Affected nodes reduction (avg): 0.0000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 19.313278
- Impact reduction: **35.6224%**
- Affected nodes reduction (avg): 1.0000

---

## Grade: med2

**Baseline cumulative impact:** 30.000000

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M105 | 5.000000 | 6 |
| 2 | M101 | 5.000000 | 6 |
| 3 | M106 | 5.000000 | 6 |
| 4 | M102 | 5.000000 | 6 |
| 5 | M103 | 5.000000 | 6 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M103 | 8.263296 |
| 2 | M102 | 6.804384 |
| 3 | M104 | 6.009289 |
| 4 | M106 | 4.990205 |
| 5 | M105 | 2.786916 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 33.2080 | 1.0000 |
|  | topk_source_control | 33.2080 | 1.0000 |
|  | edge_attenuation | -0.0000 | 0.0000 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 20.037591
- Impact reduction: **33.2080%**
- Affected nodes reduction (avg): 1.0000

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 30.000000
- Impact reduction: **-0.0000%**
- Affected nodes reduction (avg): 0.0000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 30.000000
- Intervention cumulative impact: 20.037591
- Impact reduction: **33.2080%**
- Affected nodes reduction (avg): 1.0000

---
