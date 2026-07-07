"""
MODUL 6 - Validation & Robustness
===================================
Mengevaluasi kestabilan hasil analisis shock-propagation dan intervensi
terhadap variasi parameter kunci.

Analisis yang dilakukan:
1. Sensitivity analysis terhadap:
   - propagation_steps  (default: [3, 5, 7])
   - attenuation_factor (default: [0.3, 0.5, 0.7])
2. Stability metrics per grade:
   - konsistensi top influential nodes (Jaccard similarity lintas parameter)
   - konsistensi top vulnerable nodes  (Jaccard similarity lintas parameter)
   - variance of cumulative impact     (std / mean ratio = CoV)
3. Confidence level per grade: HIGH / MEDIUM / LOW berbasis stability score.

Output:
- data/processed/module_06/robustness_results.json
- data/processed/module_06/robustness_summary.csv
- data/processed/module_06/robustness_summary.md
- data/processed/module_06/robustness_graph_{grade}.html  (per grade)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    "shock_magnitude": 1.0,
    "impact_threshold": 1e-9,
    "top_n": 5,
    # sensitivity sweep ranges
    "propagation_steps_range": [3, 5, 7],
    "attenuation_factors_range": [0.3, 0.5, 0.7],
    # thresholds for confidence assignment
    "high_stability_threshold": 0.70,
    "low_stability_threshold": 0.40,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Format {label} tidak valid (harus dict).")
    return data


def _load_network_edges(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["grade", "source", "target", "weight"])
    df = pd.read_csv(path)
    if "weight" not in df.columns:
        df["weight"] = df["f_statistic"] if "f_statistic" in df.columns else 1.0
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Propagation helpers (self-contained, mirrors Module 4/5 logic)
# ---------------------------------------------------------------------------

def _build_propagation_matrix(nodes: List[str], edge_df: pd.DataFrame) -> np.ndarray:
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    P = np.zeros((n, n), dtype=float)
    for _, row in edge_df.iterrows():
        s = node_idx.get(str(row["source"]))
        t = node_idx.get(str(row["target"]))
        if s is not None and t is not None and s != t:
            P[s, t] += float(row["weight"])
    row_sums = P.sum(axis=1, keepdims=True)
    non_zero = row_sums.ravel() > 0
    P[non_zero] = P[non_zero] / row_sums[non_zero]
    return P


def _simulate_propagation(
    nodes: List[str],
    P: np.ndarray,
    num_steps: int,
    shock_magnitude: float,
    impact_threshold: float,
    top_n: int,
) -> Dict:
    """
    Run full shock propagation and return summary metrics.
    Returns:
      total_impact_per_source, cumulative_received,
      most_influential_sources (list), most_vulnerable_nodes (list),
      cumulative_impact_total
    """
    n = len(nodes)
    simulation: Dict[str, float] = {}
    cumulative_received = np.zeros(n, dtype=float)

    for src_idx, src_node in enumerate(nodes):
        current = np.zeros(n, dtype=float)
        current[src_idx] = shock_magnitude
        total_vec = np.zeros(n, dtype=float)
        for _ in range(num_steps):
            current = P.T @ current
            total_vec += current
        cumulative_received += total_vec
        simulation[src_node] = float(total_vec.sum())

    ranked_sources = sorted(simulation.items(), key=lambda x: x[1], reverse=True)
    most_influential = [
        {"node": node, "total_impact": impact}
        for node, impact in ranked_sources[:top_n]
    ]

    vuln_ranked = sorted(enumerate(cumulative_received), key=lambda x: x[1], reverse=True)
    most_vulnerable = [
        {"node": nodes[i], "cumulative_received_impact": float(v)}
        for i, v in vuln_ranked[:top_n]
    ]

    return {
        "total_impact_per_source": simulation,
        "most_influential_sources": most_influential,
        "most_vulnerable_nodes": most_vulnerable,
        "cumulative_impact_total": float(cumulative_received.sum()),
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _run_sensitivity_sweep(
    nodes: List[str],
    edge_df: pd.DataFrame,
    steps_range: List[int],
    attenuation_range: List[float],
    shock_magnitude: float,
    impact_threshold: float,
    top_n: int,
) -> Dict:
    """
    Run propagation for each (steps, attenuation_factor) combination.
    attenuation_factor modifies outgoing edge weights of the single
    most-influential node (mirrors Module 5 node_attenuation logic).
    Returns dict keyed by parameter label.
    """
    if not nodes or edge_df.empty:
        return {}

    P_base = _build_propagation_matrix(nodes, edge_df)
    results: Dict[str, Dict] = {}

    # --- sweep propagation_steps (baseline attenuation = 1.0, no change) ---
    for steps in steps_range:
        label = f"steps={steps}"
        sim = _simulate_propagation(
            nodes, P_base, steps, shock_magnitude, impact_threshold, top_n
        )
        results[label] = {
            "param_type": "propagation_steps",
            "param_value": steps,
            **sim,
        }

    # --- sweep attenuation_factor on top-1 influential node ---
    # First identify the top-1 source from base simulation (steps = middle value)
    mid_steps = steps_range[len(steps_range) // 2]
    base_sim = _simulate_propagation(
        nodes, P_base, mid_steps, shock_magnitude, impact_threshold, top_n
    )
    top_source = (
        base_sim["most_influential_sources"][0]["node"]
        if base_sim["most_influential_sources"]
        else None
    )

    for att in attenuation_range:
        label = f"attenuation={att}"
        if top_source is not None and not edge_df.empty:
            mod_df = edge_df.copy()
            mask = mod_df["source"].astype(str) == str(top_source)
            mod_df.loc[mask, "weight"] = mod_df.loc[mask, "weight"] * att
            P_mod = _build_propagation_matrix(nodes, mod_df)
        else:
            P_mod = P_base

        sim = _simulate_propagation(
            nodes, P_mod, mid_steps, shock_magnitude, impact_threshold, top_n
        )
        results[label] = {
            "param_type": "attenuation_factor",
            "param_value": att,
            "attenuated_node": top_source,
            **sim,
        }

    return results


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union)


def _pairwise_jaccard_mean(node_sets: List[set]) -> float:
    """Average Jaccard similarity across all pairs in node_sets."""
    if len(node_sets) < 2:
        return 1.0
    scores: List[float] = []
    n = len(node_sets)
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(_jaccard(node_sets[i], node_sets[j]))
    return float(np.mean(scores))


def _compute_stability_metrics(
    sweep_results: Dict,
    top_n: int,
) -> Dict:
    """
    Given sweep_results from _run_sensitivity_sweep, compute:
    - influential_consistency  : mean pairwise Jaccard of top-n influential sets
    - vulnerable_consistency   : mean pairwise Jaccard of top-n vulnerable sets
    - impact_cov               : coefficient of variation of cumulative_impact_total
    - stability_score          : composite score in [0, 1]
    """
    if not sweep_results:
        return {
            "influential_consistency": 1.0,
            "vulnerable_consistency": 1.0,
            "impact_cov": 0.0,
            "stability_score": 1.0,
        }

    influential_sets: List[set] = []
    vulnerable_sets: List[set] = []
    impact_totals: List[float] = []

    for res in sweep_results.values():
        inf_nodes = {e["node"] for e in res.get("most_influential_sources", [])}
        vul_nodes = {e["node"] for e in res.get("most_vulnerable_nodes", [])}
        influential_sets.append(inf_nodes)
        vulnerable_sets.append(vul_nodes)
        impact_totals.append(res.get("cumulative_impact_total", 0.0))

    inf_consistency = _pairwise_jaccard_mean(influential_sets)
    vul_consistency = _pairwise_jaccard_mean(vulnerable_sets)

    impact_arr = np.array(impact_totals, dtype=float)
    # Guard against NaN/Inf values that could propagate through statistics
    impact_arr = np.where(np.isfinite(impact_arr), impact_arr, 0.0)
    mean_impact = float(np.mean(impact_arr))
    std_impact = float(np.std(impact_arr))
    impact_cov = std_impact / mean_impact if mean_impact > 1e-12 else 0.0
    # Clamp in case of numerical edge-cases (e.g. all-zero impacts)
    impact_cov = float(np.nan_to_num(impact_cov, nan=0.0, posinf=0.0, neginf=0.0))

    # Stability score: average of consistency scores minus CoV penalty (capped at 1)
    cov_penalty = min(impact_cov, 1.0)
    stability_score = float(np.clip(
        np.nan_to_num(
            (inf_consistency + vul_consistency) / 2.0 * (1.0 - 0.3 * cov_penalty),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    ))

    return {
        "influential_consistency": round(inf_consistency, 4),
        "vulnerable_consistency": round(vul_consistency, 4),
        "impact_cov": round(impact_cov, 4),
        "stability_score": round(stability_score, 4),
    }


# ---------------------------------------------------------------------------
# Confidence assignment
# ---------------------------------------------------------------------------

def _assign_confidence(stability_score: float, high_thr: float, low_thr: float) -> str:
    if stability_score >= high_thr:
        return "HIGH"
    if stability_score >= low_thr:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Cross-module consistency check
# ---------------------------------------------------------------------------

def _cross_module_consistency(
    grade: str,
    shock_grade_data: Dict,
    intervention_grade_data: Dict,
) -> Dict:
    """
    Check whether Module 5 best intervention actually reduces impact
    compared to Module 4 baseline, and that top influential nodes are
    consistent between Module 4 and Module 5 baseline.
    """
    # Module 4 top influential
    m4_influential = [
        e["node"] for e in shock_grade_data.get("most_influential_sources", [])
    ]
    # Module 5 baseline top influential
    m5_baseline = intervention_grade_data.get("baseline", {})
    m5_influential = [
        e["node"] for e in m5_baseline.get("most_influential_sources", [])
    ]
    inf_match = _jaccard(set(m4_influential), set(m5_influential))

    # Module 5 best scenario impact reduction
    ranking = intervention_grade_data.get("ranking", [])
    best_reduction = ranking[0].get("impact_reduction_pct", 0.0) if ranking else 0.0

    return {
        "m4_m5_influential_jaccard": round(inf_match, 4),
        "best_intervention_impact_reduction_pct": round(best_reduction, 4),
        "cross_module_consistent": inf_match >= 0.5,
    }


# ---------------------------------------------------------------------------
# Per-grade robustness runner
# ---------------------------------------------------------------------------

def _run_robustness_for_grade(
    grade: str,
    shock_grade_data: Dict,
    intervention_grade_data: Dict,
    edge_df: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    nodes: List[str] = shock_grade_data.get("nodes", [])
    steps_range: List[int] = [int(s) for s in cfg["propagation_steps_range"]]
    att_range: List[float] = [float(a) for a in cfg["attenuation_factors_range"]]
    shock_mag = float(cfg["shock_magnitude"])
    impact_thr = float(cfg["impact_threshold"])
    top_n = int(cfg["top_n"])

    sweep = _run_sensitivity_sweep(
        nodes=nodes,
        edge_df=edge_df,
        steps_range=steps_range,
        attenuation_range=att_range,
        shock_magnitude=shock_mag,
        impact_threshold=impact_thr,
        top_n=top_n,
    )

    stability = _compute_stability_metrics(sweep, top_n)
    confidence = _assign_confidence(
        stability["stability_score"],
        float(cfg["high_stability_threshold"]),
        float(cfg["low_stability_threshold"]),
    )

    cross = _cross_module_consistency(grade, shock_grade_data, intervention_grade_data)

    return {
        "nodes": nodes,
        "edges_count": int(len(edge_df)),
        "sweep_results": sweep,
        "stability_metrics": stability,
        "confidence_level": confidence,
        "cross_module_consistency": cross,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _build_summary_rows(grades: List[str], per_grade: Dict) -> List[Dict]:
    rows = []
    for grade in grades:
        gd = per_grade.get(grade, {})
        sm = gd.get("stability_metrics", {})
        cross = gd.get("cross_module_consistency", {})
        rows.append(
            {
                "grade": grade,
                "nodes": len(gd.get("nodes", [])),
                "edges": gd.get("edges_count", 0),
                "influential_consistency": sm.get("influential_consistency", 0.0),
                "vulnerable_consistency": sm.get("vulnerable_consistency", 0.0),
                "impact_cov": sm.get("impact_cov", 0.0),
                "stability_score": sm.get("stability_score", 0.0),
                "confidence_level": gd.get("confidence_level", "LOW"),
                "m4_m5_influential_jaccard": cross.get("m4_m5_influential_jaccard", 0.0),
                "best_intervention_reduction_pct": cross.get(
                    "best_intervention_impact_reduction_pct", 0.0
                ),
            }
        )
    return rows


def _write_markdown(
    grades: List[str],
    per_grade: Dict,
    out_dir: Path,
    cfg: Dict,
) -> None:
    lines = [
        "# MODUL 6 - Validation & Robustness Summary",
        "",
        f"Propagation steps sweep: {cfg['propagation_steps_range']}  ",
        f"Attenuation factors sweep: {cfg['attenuation_factors_range']}  ",
        f"High stability threshold: {cfg['high_stability_threshold']}  ",
        f"Low stability threshold: {cfg['low_stability_threshold']}",
        "",
        "---",
        "",
        "## Per-Grade Summary",
        "",
        "| Grade | Nodes | Edges | Influential Consistency | Vulnerable Consistency |"
        " Impact CoV | Stability Score | Confidence |",
        "|---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for grade in grades:
        gd = per_grade.get(grade, {})
        sm = gd.get("stability_metrics", {})
        lines.append(
            f"| {grade} | {len(gd.get('nodes', []))} | {gd.get('edges_count', 0)}"
            f" | {sm.get('influential_consistency', 0.0):.4f}"
            f" | {sm.get('vulnerable_consistency', 0.0):.4f}"
            f" | {sm.get('impact_cov', 0.0):.4f}"
            f" | {sm.get('stability_score', 0.0):.4f}"
            f" | **{gd.get('confidence_level', 'LOW')}** |"
        )
    lines += ["", "---", ""]

    for grade in grades:
        gd = per_grade.get(grade, {})
        sm = gd.get("stability_metrics", {})
        cross = gd.get("cross_module_consistency", {})
        sweep = gd.get("sweep_results", {})

        lines += [
            f"## Grade: {grade}",
            "",
            f"**Confidence Level: {gd.get('confidence_level', 'LOW')}**  ",
            f"Stability Score: {sm.get('stability_score', 0.0):.4f}  ",
            f"Influential Consistency: {sm.get('influential_consistency', 0.0):.4f}  ",
            f"Vulnerable Consistency: {sm.get('vulnerable_consistency', 0.0):.4f}  ",
            f"Impact CoV: {sm.get('impact_cov', 0.0):.4f}",
            "",
            "### Cross-Module Consistency",
            "",
            f"- Module 4 ↔ Module 5 influential Jaccard: "
            f"{cross.get('m4_m5_influential_jaccard', 0.0):.4f}",
            f"- Best intervention impact reduction: "
            f"{cross.get('best_intervention_impact_reduction_pct', 0.0):.2f}%",
            f"- Cross-module consistent: {cross.get('cross_module_consistent', False)}",
            "",
        ]

        if sweep:
            lines += [
                "### Sensitivity Sweep Results",
                "",
                "| Parameter | Value | Cumulative Impact | Top Influential | Top Vulnerable |",
                "|---|---|---:|---|---|",
            ]
            for label, res in sweep.items():
                top_inf = ", ".join(
                    e["node"] for e in res.get("most_influential_sources", [])[:3]
                )
                top_vul = ", ".join(
                    e["node"] for e in res.get("most_vulnerable_nodes", [])[:3]
                )
                lines.append(
                    f"| {res.get('param_type', '')} | {res.get('param_value', '')} |"
                    f" {res.get('cumulative_impact_total', 0.0):.6f} |"
                    f" {top_inf} | {top_vul} |"
                )
            lines.append("")

        lines += ["---", ""]

    (out_dir / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _save_html_visualization(
    grade: str,
    nodes: List[str],
    edge_df: pd.DataFrame,
    grade_result: Dict,
    out_dir: Path,
) -> Path:
    html_path = out_dir / f"robustness_graph_{grade}.html"

    sweep = grade_result.get("sweep_results", {})
    sm = grade_result.get("stability_metrics", {})
    confidence = grade_result.get("confidence_level", "LOW")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # --- subplot 1: stability bar chart ---
        metric_names = [
            "Influential\nConsistency",
            "Vulnerable\nConsistency",
            "1 - Impact CoV",
            "Stability\nScore",
        ]
        metric_vals = [
            sm.get("influential_consistency", 0.0),
            sm.get("vulnerable_consistency", 0.0),
            max(0.0, 1.0 - sm.get("impact_cov", 0.0)),
            sm.get("stability_score", 0.0),
        ]
        bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        bar_trace = go.Bar(
            x=metric_names,
            y=metric_vals,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in metric_vals],
            textposition="outside",
            name="Stability Metrics",
        )

        # --- subplot 2: sensitivity line chart (cumulative impact per sweep param) ---
        steps_labels: List[str] = []
        steps_impacts: List[float] = []
        att_labels: List[str] = []
        att_impacts: List[float] = []

        for label, res in sweep.items():
            if res.get("param_type") == "propagation_steps":
                steps_labels.append(str(res["param_value"]))
                steps_impacts.append(res.get("cumulative_impact_total", 0.0))
            elif res.get("param_type") == "attenuation_factor":
                att_labels.append(str(res["param_value"]))
                att_impacts.append(res.get("cumulative_impact_total", 0.0))

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                f"Stability Metrics – {grade}",
                f"Impact vs Propagation Steps – {grade}",
                f"Impact vs Attenuation Factor – {grade}",
            ],
        )

        fig.add_trace(bar_trace, row=1, col=1)

        if steps_labels:
            fig.add_trace(
                go.Scatter(
                    x=steps_labels,
                    y=steps_impacts,
                    mode="lines+markers",
                    name="Steps sweep",
                    line=dict(color="#2196F3", width=2),
                    marker=dict(size=8),
                ),
                row=1,
                col=2,
            )

        if att_labels:
            fig.add_trace(
                go.Scatter(
                    x=att_labels,
                    y=att_impacts,
                    mode="lines+markers",
                    name="Attenuation sweep",
                    line=dict(color="#FF9800", width=2),
                    marker=dict(size=8),
                ),
                row=1,
                col=3,
            )

        stability_score = sm.get("stability_score", 0.0)
        conf_color = {"HIGH": "#4CAF50", "MEDIUM": "#FF9800", "LOW": "#F44336"}.get(
            confidence, "#9E9E9E"
        )

        fig.update_layout(
            title=(
                f"Module 6 Robustness Analysis – {grade}<br>"
                f"<sup>Confidence: <b style='color:{conf_color}'>{confidence}</b> "
                f"| Stability Score: {stability_score:.4f}</sup>"
            ),
            showlegend=False,
            template="plotly_white",
            yaxis=dict(range=[0, 1.1]),
        )

        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path

    except Exception:
        # Fallback minimal HTML
        metrics_html = "".join(
            f"<li>{k}: {v}</li>"
            for k, v in sm.items()
        )
        sweep_html = ""
        for label, res in sweep.items():
            sweep_html += (
                f"<li>{label}: cumulative_impact = "
                f"{res.get('cumulative_impact_total', 0.0):.4f}</li>"
            )
        if not sweep_html:
            sweep_html = "<li>(no sweep data)</li>"

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Module 6 Robustness Graph - {grade}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #666; }}
    .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px;
              font-weight: bold; color: #fff;
              background: {'#4CAF50' if confidence == 'HIGH'
                           else '#FF9800' if confidence == 'MEDIUM' else '#F44336'}; }}
  </style>
</head>
<body>
  <h1>Module 6 Robustness Analysis - {grade}</h1>
  <p>Confidence Level: <span class="badge">{confidence}</span></p>
  <p class="muted">Fallback HTML visualization (plotly unavailable).</p>
  <h2>Stability Metrics</h2>
  <ul>{metrics_html}</ul>
  <h2>Sensitivity Sweep</h2>
  <ul>{sweep_html}</ul>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_module6_validation_robustness(
    network_results_path: str,
    shock_results_path: str,
    intervention_results_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Jalankan MODUL 6 dari output MODUL 3, 4, dan 5.

    Parameters
    ----------
    network_results_path : str
        Path ke `network_inference_results.json` dari MODUL 3.
    shock_results_path : str
        Path ke `shock_propagation_results.json` dari MODUL 4.
    intervention_results_path : str
        Path ke `intervention_results.json` dari MODUL 5.
    output_dir : str
        Folder output khusus MODUL 6, contoh: data/processed/module_06
    config : dict | None
        Konfigurasi opsional. Kunci yang didukung:
        - shock_magnitude (float, default 1.0)
        - impact_threshold (float, default 1e-9)
        - top_n (int, default 5)
        - propagation_steps_range (list[int], default [3,5,7])
        - attenuation_factors_range (list[float], default [0.3,0.5,0.7])
        - high_stability_threshold (float, default 0.70)
        - low_stability_threshold (float, default 0.40)

    Returns
    -------
    dict
        Hasil validasi dan robustness lengkap per grade.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    net_path = Path(network_results_path)
    shock_path = Path(shock_results_path)
    intervention_path = Path(intervention_results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    net_data = _load_json(net_path, "network_inference_results.json")
    shock_data = _load_json(shock_path, "shock_propagation_results.json")
    intervention_data = _load_json(intervention_path, "intervention_results.json")

    # Load edges (from Module 3 dir, neighbour of network_results_path)
    edges_candidate = net_path.parent / "network_edges.csv"
    edges_all = _load_network_edges(edges_candidate if edges_candidate.exists() else None)

    grades: List[str] = shock_data.get("grades", [])
    shock_per_grade: Dict = shock_data.get("per_grade", {})
    intervention_per_grade: Dict = intervention_data.get("per_grade", {})

    per_grade_results: Dict[str, Dict] = {}
    visualization_paths: Dict[str, str] = {}

    for grade in grades:
        shock_gd = shock_per_grade.get(grade, {})
        intervention_gd = intervention_per_grade.get(grade, {})

        if not edges_all.empty and "grade" in edges_all.columns:
            edge_df = edges_all[edges_all["grade"] == grade].copy().reset_index(drop=True)
        else:
            edge_df = pd.DataFrame(columns=["grade", "source", "target", "weight"])

        grade_result = _run_robustness_for_grade(
            grade=grade,
            shock_grade_data=shock_gd,
            intervention_grade_data=intervention_gd,
            edge_df=edge_df,
            cfg=cfg,
        )
        per_grade_results[grade] = grade_result

        nodes: List[str] = shock_gd.get("nodes", [])
        viz_path = _save_html_visualization(
            grade=grade,
            nodes=nodes,
            edge_df=edge_df,
            grade_result=grade_result,
            out_dir=out_dir,
        )
        visualization_paths[grade] = str(viz_path)

    # --- robustness_results.json ---
    result = {
        "module": "module_06_validation_robustness",
        "input_network_results": str(net_path),
        "input_shock_results": str(shock_path),
        "input_intervention_results": str(intervention_path),
        "output_dir": str(out_dir),
        "grades": grades,
        "config": cfg,
        "per_grade": per_grade_results,
        "visualizations": visualization_paths,
    }

    with (out_dir / "robustness_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # --- robustness_summary.csv ---
    summary_rows = _build_summary_rows(grades, per_grade_results)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "robustness_summary.csv", index=False)

    # --- robustness_summary.md ---
    _write_markdown(grades=grades, per_grade=per_grade_results, out_dir=out_dir, cfg=cfg)

    print(f"[MODUL 6] Output tersimpan di: {out_dir}")
    for grade in grades:
        gd = per_grade_results.get(grade, {})
        sm = gd.get("stability_metrics", {})
        print(
            f"  Grade={grade}: stability={sm.get('stability_score', 0.0):.4f}, "
            f"confidence={gd.get('confidence_level', 'LOW')}"
        )

    return result
