# MODUL 6 - Validation & Robustness Summary

Propagation steps sweep: [3, 5, 7]  
Attenuation factors sweep: [0.3, 0.5, 0.7]  
High stability threshold: 0.7  
Low stability threshold: 0.4

---

## Per-Grade Summary

| Grade | Nodes | Edges | Influential Consistency | Vulnerable Consistency | Impact CoV | Stability Score | Confidence |
|---|---:|---:|---:|---:|---:|---:|:---|
| low1 | 6 | 30 | 0.8889 | 1.0000 | 0.2309 | 0.8790 | **HIGH** |
| low2 | 6 | 30 | 0.8000 | 1.0000 | 0.2309 | 0.8376 | **HIGH** |
| med1 | 6 | 30 | 0.7556 | 1.0000 | 0.2309 | 0.8170 | **HIGH** |
| med2 | 6 | 30 | 0.8222 | 1.0000 | 0.2309 | 0.8480 | **HIGH** |

---

## Grade: low1

**Confidence Level: HIGH**  
Stability Score: 0.8790  
Influential Consistency: 0.8889  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.2309

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 22.02%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 18.000000 | M106, M101, M102 | M103, M105, M106 |
| propagation_steps | 5 | 30.000000 | M101, M102, M103 | M103, M105, M106 |
| propagation_steps | 7 | 42.000000 | M101, M102, M103 | M103, M105, M106 |
| attenuation_factor | 0.3 | 30.000000 | M101, M102, M103 | M103, M105, M106 |
| attenuation_factor | 0.5 | 30.000000 | M101, M102, M103 | M103, M105, M106 |
| attenuation_factor | 0.7 | 30.000000 | M101, M102, M103 | M103, M105, M106 |

---

## Grade: low2

**Confidence Level: HIGH**  
Stability Score: 0.8376  
Influential Consistency: 0.8000  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.2309

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 47.47%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 18.000000 | M105, M106, M103 | M103, M105, M106 |
| propagation_steps | 5 | 30.000000 | M106, M101, M102 | M103, M105, M106 |
| propagation_steps | 7 | 42.000000 | M106, M102, M103 | M103, M105, M106 |
| attenuation_factor | 0.3 | 30.000000 | M106, M101, M102 | M103, M105, M106 |
| attenuation_factor | 0.5 | 30.000000 | M106, M101, M102 | M103, M105, M106 |
| attenuation_factor | 0.7 | 30.000000 | M101, M106, M102 | M103, M105, M106 |

---

## Grade: med1

**Confidence Level: HIGH**  
Stability Score: 0.8170  
Influential Consistency: 0.7556  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.2309

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 35.62%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 18.000000 | M104, M103, M106 | M103, M102, M104 |
| propagation_steps | 5 | 30.000000 | M106, M104, M101 | M103, M102, M104 |
| propagation_steps | 7 | 42.000000 | M106, M104, M103 | M103, M102, M104 |
| attenuation_factor | 0.3 | 30.000000 | M101, M102, M103 | M103, M102, M104 |
| attenuation_factor | 0.5 | 30.000000 | M106, M104, M101 | M103, M102, M104 |
| attenuation_factor | 0.7 | 30.000000 | M101, M102, M103 | M103, M102, M104 |

---

## Grade: med2

**Confidence Level: HIGH**  
Stability Score: 0.8480  
Influential Consistency: 0.8222  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.2309

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 33.21%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 18.000000 | M102, M105, M103 | M103, M102, M104 |
| propagation_steps | 5 | 30.000000 | M105, M101, M106 | M103, M102, M104 |
| propagation_steps | 7 | 42.000000 | M105, M102, M103 | M103, M102, M104 |
| attenuation_factor | 0.3 | 30.000000 | M105, M101, M106 | M103, M102, M104 |
| attenuation_factor | 0.5 | 30.000000 | M105, M101, M106 | M103, M102, M104 |
| attenuation_factor | 0.7 | 30.000000 | M105, M106, M101 | M103, M102, M104 |

---
