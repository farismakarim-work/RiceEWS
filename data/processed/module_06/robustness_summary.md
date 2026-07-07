# MODUL 6 - Validation & Robustness Summary

Propagation steps sweep: [3, 5, 7]  
Attenuation factors sweep: [0.3, 0.5, 0.7]  
High stability threshold: 0.7  
Low stability threshold: 0.4

---

## Per-Grade Summary

| Grade | Nodes | Edges | Influential Consistency | Vulnerable Consistency | Impact CoV | Stability Score | Confidence |
|---|---:|---:|---:|---:|---:|---:|:---|
| low1 | 6 | 11 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | **HIGH** |
| low2 | 6 | 14 | 1.0000 | 1.0000 | 0.0180 | 0.9946 | **HIGH** |
| med1 | 6 | 15 | 1.0000 | 1.0000 | 0.0024 | 0.9993 | **HIGH** |
| med2 | 6 | 10 | 1.0000 | 1.0000 | 0.0060 | 0.9982 | **HIGH** |

---

## Grade: low1

**Confidence Level: HIGH**  
Stability Score: 1.0000  
Influential Consistency: 1.0000  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.0000

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 45.69%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 10.835137 | M102, M101, M103 | M105, M106, M103 |
| propagation_steps | 5 | 10.835137 | M102, M101, M103 | M105, M106, M103 |
| propagation_steps | 7 | 10.835137 | M102, M101, M103 | M105, M106, M103 |
| attenuation_factor | 0.3 | 10.835137 | M102, M101, M103 | M105, M106, M103 |
| attenuation_factor | 0.5 | 10.835137 | M102, M101, M103 | M105, M106, M103 |
| attenuation_factor | 0.7 | 10.835137 | M102, M101, M103 | M105, M106, M103 |

---

## Grade: low2

**Confidence Level: HIGH**  
Stability Score: 0.9946  
Influential Consistency: 1.0000  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.0180

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 37.84%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 10.045988 | M103, M104, M101 | M102, M105, M106 |
| propagation_steps | 5 | 10.551486 | M103, M104, M101 | M102, M105, M106 |
| propagation_steps | 7 | 10.551486 | M103, M104, M101 | M102, M105, M106 |
| attenuation_factor | 0.3 | 10.551486 | M103, M104, M101 | M102, M105, M106 |
| attenuation_factor | 0.5 | 10.551486 | M103, M104, M101 | M102, M105, M106 |
| attenuation_factor | 0.7 | 10.551486 | M103, M104, M101 | M102, M105, M106 |

---

## Grade: med1

**Confidence Level: HIGH**  
Stability Score: 0.9993  
Influential Consistency: 1.0000  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.0024

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 27.32%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 7.925101 | M101, M105, M102 | M103, M104, M106 |
| propagation_steps | 5 | 7.976851 | M101, M105, M102 | M103, M104, M106 |
| propagation_steps | 7 | 7.976851 | M101, M105, M102 | M103, M104, M106 |
| attenuation_factor | 0.3 | 7.976851 | M101, M105, M102 | M103, M104, M106 |
| attenuation_factor | 0.5 | 7.976851 | M101, M105, M102 | M103, M104, M106 |
| attenuation_factor | 0.7 | 7.976851 | M101, M105, M102 | M103, M104, M106 |

---

## Grade: med2

**Confidence Level: HIGH**  
Stability Score: 0.9982  
Influential Consistency: 1.0000  
Vulnerable Consistency: 1.0000  
Impact CoV: 0.0060

### Cross-Module Consistency

- Module 4 ↔ Module 5 influential Jaccard: 1.0000
- Best intervention impact reduction: 28.22%
- Cross-module consistent: True

### Sensitivity Sweep Results

| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |
|---|---|---:|---|---|
| propagation_steps | 3 | 8.815153 | M105, M103, M101 | M104, M106, M102 |
| propagation_steps | 5 | 8.959216 | M105, M103, M101 | M104, M106, M102 |
| propagation_steps | 7 | 8.959216 | M105, M103, M101 | M104, M106, M102 |
| attenuation_factor | 0.3 | 8.959216 | M105, M103, M101 | M104, M106, M102 |
| attenuation_factor | 0.5 | 8.959216 | M105, M103, M101 | M104, M106, M102 |
| attenuation_factor | 0.7 | 8.959216 | M105, M103, M101 | M104, M106, M102 |

---
