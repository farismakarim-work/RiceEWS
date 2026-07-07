# MODUL 5 - Intervention Analysis Summary

Num propagation steps: 5  
Shock magnitude: 1.0  
Node attenuation factor: 0.0  
Edge attenuation factor: 0.0  
Top-k control: 1

---

## Grade: low1

**Baseline cumulative impact:** 10.835137

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M102 | 2.927788 | 4 |
| 2 | M101 | 2.907350 | 4 |
| 3 | M103 | 2.000000 | 2 |
| 4 | M104 | 2.000000 | 2 |
| 5 | M106 | 1.000000 | 1 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M105 | 5.000000 |
| 2 | M106 | 3.950542 |
| 3 | M103 | 1.464701 |
| 4 | M104 | 0.419894 |
| 5 | M101 | 0.000000 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | edge_attenuation | 45.6897 | 0.5000 |
|  | node_attenuation | 27.0212 | 0.6667 |
|  | topk_source_control | 27.0212 | 0.6667 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 10.835137
- Intervention cumulative impact: 7.907350
- Impact reduction: **27.0212%**
- Affected nodes reduction (avg): 0.6667

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 10.835137
- Intervention cumulative impact: 5.884596
- Impact reduction: **45.6897%**
- Affected nodes reduction (avg): 0.5000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 10.835137
- Intervention cumulative impact: 7.907350
- Impact reduction: **27.0212%**
- Affected nodes reduction (avg): 0.6667

---

## Grade: low2

**Baseline cumulative impact:** 10.551486

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M103 | 2.565723 | 3 |
| 2 | M104 | 2.541873 | 4 |
| 3 | M101 | 2.449027 | 5 |
| 4 | M106 | 1.994862 | 2 |
| 5 | M105 | 1.000000 | 1 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M102 | 5.000000 |
| 2 | M105 | 3.149545 |
| 3 | M106 | 1.701166 |
| 4 | M103 | 0.556058 |
| 5 | M104 | 0.144718 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 37.8374 | 0.5000 |
|  | topk_source_control | 37.8374 | 0.5000 |
|  | edge_attenuation | 25.4683 | 0.3333 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 10.551486
- Intervention cumulative impact: 6.559073
- Impact reduction: **37.8374%**
- Affected nodes reduction (avg): 0.5000

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 10.551486
- Intervention cumulative impact: 7.864198
- Impact reduction: **25.4683%**
- Affected nodes reduction (avg): 0.3333

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 10.551486
- Intervention cumulative impact: 6.559073
- Impact reduction: **37.8374%**
- Affected nodes reduction (avg): 0.5000

---

## Grade: med1

**Baseline cumulative impact:** 7.976851

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M101 | 2.179016 | 5 |
| 2 | M105 | 1.970394 | 4 |
| 3 | M102 | 1.544092 | 3 |
| 4 | M106 | 1.283349 | 2 |
| 5 | M104 | 1.000000 | 1 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M103 | 5.000000 |
| 2 | M104 | 1.216397 |
| 3 | M106 | 1.023855 |
| 4 | M102 | 0.693146 |
| 5 | M105 | 0.043452 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 27.3167 | 0.8333 |
|  | topk_source_control | 27.3167 | 0.8333 |
|  | edge_attenuation | -12.2866 | 0.0000 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 7.976851
- Intervention cumulative impact: 5.797835
- Impact reduction: **27.3167%**
- Affected nodes reduction (avg): 0.8333

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 7.976851
- Intervention cumulative impact: 8.956938
- Impact reduction: **-12.2866%**
- Affected nodes reduction (avg): 0.0000

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 7.976851
- Intervention cumulative impact: 5.797835
- Impact reduction: **27.3167%**
- Affected nodes reduction (avg): 0.8333

---

## Grade: med2

**Baseline cumulative impact:** 8.959216

### Baseline – Most Influential Sources

| Rank | Node | Total Impact | Affected Nodes |
|---:|---|---:|---:|
| 1 | M105 | 2.528232 | 4 |
| 2 | M103 | 2.048985 | 3 |
| 3 | M101 | 2.000000 | 2 |
| 4 | M102 | 1.381999 | 2 |
| 5 | M106 | 1.000000 | 1 |

### Baseline – Most Vulnerable Nodes

| Rank | Node | Cumulative Received Impact |
|---:|---|---:|
| 1 | M104 | 5.000000 |
| 2 | M106 | 2.330104 |
| 3 | M102 | 1.124517 |
| 4 | M103 | 0.504595 |
| 5 | M101 | 0.000000 |

### Scenario Ranking (best to worst)

| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |
|---:|---|---:|---:|
|  | node_attenuation | 28.2193 | 0.6667 |
|  | topk_source_control | 28.2193 | 0.6667 |
|  | edge_attenuation | 16.5464 | 0.3333 |

### Scenario: `node_attenuation`

_Attenuate outgoing propagation of top-1 source nodes by factor 0.0_

- Baseline cumulative impact: 8.959216
- Intervention cumulative impact: 6.430984
- Impact reduction: **28.2193%**
- Affected nodes reduction (avg): 0.6667

### Scenario: `edge_attenuation`

_Attenuate top-1 highest-weight edges by factor 0.0_

- Baseline cumulative impact: 8.959216
- Intervention cumulative impact: 7.476784
- Impact reduction: **16.5464%**
- Affected nodes reduction (avg): 0.3333

### Scenario: `topk_source_control`

_Remove all outgoing edges from top-1 most influential sources_

- Baseline cumulative impact: 8.959216
- Intervention cumulative impact: 6.430984
- Impact reduction: **28.2193%**
- Affected nodes reduction (avg): 0.6667

---
