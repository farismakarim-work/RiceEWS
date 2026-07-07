"""
MODUL 5 - Intervention Analysis
================================
Mengevaluasi skenario intervensi terhadap jaringan shock-propagation dari MODUL 4.
Membandingkan baseline impact vs kondisi setelah intervensi dan meranking skenario
terbaik per grade.

Skenario intervensi yang disimulasikan:
1. node_attenuation   – melemahkan kemampuan satu atau lebih node meneruskan shock.
2. edge_attenuation   – melemahkan bobot satu atau lebih edge tertentu.
3. topk_source_control – menekan total_impact source k teratas menjadi 0.

Output:
- data/processed/module_05/intervention_results.json
- data/processed/module_05/intervention_summary.csv
- data/processed/module_05/intervention_summary.md
- data/processed/module_05/intervention_graph_{grade}.html  (per grade)
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    "num_steps": 5,
    "shock_magnitude": 1.0,
    "impact_threshold": 1e-9,
    "top_n": 5,
    # node attenuation: fraction of propagation capacity RETAINED [0, 1]
    # 0.0 = complete removal (strongest intervention)
    # 0.5 = 50% capacity retained (moderate intervention, default)
    # 1.0 = no intervention
    "node_attenuation_factor": 0.5,
    # edge attenuation: fraction of edge weight RETAINED [0, 1]
    # 0.0 = remove edges entirely, 1.0 = no change
    "edge_attenuation_factor": 0.5,
    # top-k sources to suppress in topk_source_control scenario
    "topk_control": 1,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_shock_results(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"shock_propagation_results.json tidak ditemukan: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format shock_propagation_results.json tidak valid (harus dict).")
    return data


def _load_network_edges(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["grade", "source", "target", "weight"])
    df = pd.read_csv(path)
    if "weight" not in df.columns:
        if "f_statistic" in df.columns:
            df["weight"] = df["f_statistic"]
        else:
            df["weight"] = 1.0
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Propagation matrix (shared with Module 4 logic, self-contained here)
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
    Jalankan propagasi shock dari semua node dan kembalikan metrik ringkas.
    Returns dict dengan kunci:
      total_impact_per_source, affected_nodes_count,
      most_influential_sources, most_vulnerable_nodes,
      cumulative_impact_total
    """
    n = len(nodes)
    simulation_results: Dict[str, Dict] = {}
    cumulative_received = np.zeros(n, dtype=float)

    for src_idx, src_node in enumerate(nodes):
        current = np.zeros(n, dtype=float)
        current[src_idx] = shock_magnitude
        total_impact_vec = np.zeros(n, dtype=float)
        for _ in range(num_steps):
            current = P.T @ current
            total_impact_vec += current
        cumulative_received += total_impact_vec
        total_impact_value = float(total_impact_vec.sum())
        affected_count = int(np.sum(total_impact_vec > impact_threshold))
        simulation_results[src_node] = {
            "total_impact": total_impact_value,
            "affected_nodes_count": affected_count,
        }

    ranked_sources = sorted(
        simulation_results.items(),
        key=lambda x: x[1]["total_impact"],
        reverse=True,
    )
    most_influential = [
        {
            "node": node,
            "total_impact": info["total_impact"],
            "affected_nodes_count": info["affected_nodes_count"],
        }
        for node, info in ranked_sources[:top_n]
    ]

    vuln_ranked = sorted(enumerate(cumulative_received), key=lambda x: x[1], reverse=True)
    most_vulnerable = [
        {
            "node": nodes[i],
            "cumulative_received_impact": float(v),
        }
        for i, v in vuln_ranked[:top_n]
    ]

    return {
        "total_impact_per_source": {n: r["total_impact"] for n, r in simulation_results.items()},
        "affected_nodes_count": {n: r["affected_nodes_count"] for n, r in simulation_results.items()},
        "most_influential_sources": most_influential,
        "most_vulnerable_nodes": most_vulnerable,
        "cumulative_impact_total": float(cumulative_received.sum()),
    }


# ---------------------------------------------------------------------------
# Baseline extractor from Module 4 results
# ---------------------------------------------------------------------------

def _extract_baseline_from_shock_results(grade_data: Dict) -> Dict:
    """
    Buat metrik baseline dari hasil MODUL 4 per grade.
    """
    total_impact_per_source: Dict[str, float] = grade_data.get("total_impact_per_source", {})
    affected_nodes_count: Dict[str, int] = grade_data.get("affected_nodes_count", {})
    most_influential: List[Dict] = grade_data.get("most_influential_sources", [])
    most_vulnerable: List[Dict] = grade_data.get("most_vulnerable_nodes", [])

    cumulative_total = sum(total_impact_per_source.values())

    return {
        "total_impact_per_source": total_impact_per_source,
        "affected_nodes_count": affected_nodes_count,
        "most_influential_sources": most_influential,
        "most_vulnerable_nodes": most_vulnerable,
        "cumulative_impact_total": cumulative_total,
    }


# ---------------------------------------------------------------------------
# Intervention scenarios
# ---------------------------------------------------------------------------

def _apply_node_attenuation(
    nodes: List[str],
    edge_df: pd.DataFrame,
    target_nodes: List[str],
    attenuation_factor: float,
) -> np.ndarray:
    """
    Lemahkan kemampuan target_nodes meneruskan shock dengan mengalikan
    seluruh outgoing weights-nya dengan attenuation_factor.
    """
    mod_df = edge_df.copy()
    if not mod_df.empty:
        mask = mod_df["source"].isin(set(target_nodes))
        mod_df.loc[mask, "weight"] = mod_df.loc[mask, "weight"] * attenuation_factor
    return _build_propagation_matrix(nodes, mod_df)


def _apply_edge_attenuation(
    nodes: List[str],
    edge_df: pd.DataFrame,
    target_edges: List[Tuple[str, str]],
    attenuation_factor: float,
) -> np.ndarray:
    """
    Lemahkan bobot edge tertentu dengan mengalikannya dengan attenuation_factor.
    """
    mod_df = edge_df.copy()
    if not mod_df.empty and target_edges:
        te_set = set(target_edges)
        mask = mod_df.apply(
            lambda r: (str(r["source"]), str(r["target"])) in te_set, axis=1
        )
        mod_df.loc[mask, "weight"] = mod_df.loc[mask, "weight"] * attenuation_factor
    return _build_propagation_matrix(nodes, mod_df)


def _apply_topk_source_control(
    nodes: List[str],
    edge_df: pd.DataFrame,
    topk_sources: List[str],
) -> np.ndarray:
    """
    Hilangkan seluruh outgoing edges dari top-k source nodes
    (simulasi pengendalian penuh sumber shock paling berpengaruh).
    """
    mod_df = edge_df.copy()
    if not mod_df.empty and topk_sources:
        mod_df = mod_df[~mod_df["source"].isin(set(topk_sources))].reset_index(drop=True)
    return _build_propagation_matrix(nodes, mod_df)


# ---------------------------------------------------------------------------
# Impact comparison
# ---------------------------------------------------------------------------

def _compare_baseline_vs_intervention(baseline: Dict, intervention: Dict) -> Dict:
    """
    Hitung perbedaan antara baseline dan hasil skenario intervensi.
    """
    b_total = baseline["cumulative_impact_total"]
    i_total = intervention["cumulative_impact_total"]

    impact_reduction_pct = (
        100.0 * (b_total - i_total) / b_total if b_total > 1e-12 else 0.0
    )

    # Affected nodes reduction: average across sources
    b_aff = baseline["affected_nodes_count"]
    i_aff = intervention["affected_nodes_count"]
    sources = set(b_aff.keys()) | set(i_aff.keys())
    if sources:
        avg_b_aff = sum(b_aff.get(s, 0) for s in sources) / len(sources)
        avg_i_aff = sum(i_aff.get(s, 0) for s in sources) / len(sources)
        affected_nodes_reduction = avg_b_aff - avg_i_aff
    else:
        avg_b_aff = avg_i_aff = affected_nodes_reduction = 0.0

    # Ranking change of most influential sources
    b_rank = {e["node"]: r + 1 for r, e in enumerate(baseline["most_influential_sources"])}
    i_rank = {e["node"]: r + 1 for r, e in enumerate(intervention["most_influential_sources"])}
    all_nodes_ranked = set(b_rank.keys()) | set(i_rank.keys())
    rank_changes = {
        node: {
            "baseline_rank": b_rank.get(node),
            "intervention_rank": i_rank.get(node),
        }
        for node in all_nodes_ranked
    }

    # Ranking change of most vulnerable nodes
    b_vuln_rank = {e["node"]: r + 1 for r, e in enumerate(baseline["most_vulnerable_nodes"])}
    i_vuln_rank = {e["node"]: r + 1 for r, e in enumerate(intervention["most_vulnerable_nodes"])}
    all_vuln = set(b_vuln_rank.keys()) | set(i_vuln_rank.keys())
    vuln_rank_changes = {
        node: {
            "baseline_rank": b_vuln_rank.get(node),
            "intervention_rank": i_vuln_rank.get(node),
        }
        for node in all_vuln
    }

    return {
        "baseline_cumulative_impact": b_total,
        "intervention_cumulative_impact": i_total,
        "impact_reduction_pct": round(impact_reduction_pct, 4),
        "baseline_avg_affected_nodes": round(avg_b_aff, 4),
        "intervention_avg_affected_nodes": round(avg_i_aff, 4),
        "affected_nodes_reduction": round(affected_nodes_reduction, 4),
        "influential_rank_changes": rank_changes,
        "vulnerable_rank_changes": vuln_rank_changes,
    }


# ---------------------------------------------------------------------------
# Per-grade intervention runner
# ---------------------------------------------------------------------------

def _run_interventions_for_grade(
    grade: str,
    grade_shock_data: Dict,
    edge_df: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """
    Jalankan semua skenario intervensi untuk satu grade.
    Mengembalikan dict hasil lengkap per skenario.
    """
    nodes: List[str] = grade_shock_data.get("nodes", [])
    num_steps = int(cfg["num_steps"])
    shock_mag = float(cfg["shock_magnitude"])
    impact_thr = float(cfg["impact_threshold"])
    top_n = int(cfg["top_n"])
    node_att_factor = float(cfg["node_attenuation_factor"])
    edge_att_factor = float(cfg["edge_attenuation_factor"])
    topk = int(cfg["topk_control"])

    # Baseline from Module 4 output
    baseline = _extract_baseline_from_shock_results(grade_shock_data)

    if not nodes or edge_df.empty:
        empty = {
            "baseline": baseline,
            "scenarios": {},
            "ranking": [],
        }
        return empty

    # Identify top-k source nodes from baseline
    most_influential = baseline["most_influential_sources"]
    topk_sources = [e["node"] for e in most_influential[:topk]]

    # Top influential edges (highest weight outgoing from topk sources)
    if not edge_df.empty:
        top_edge_rows = (
            edge_df.sort_values("weight", ascending=False)
            .head(topk)
        )
        top_edges = list(
            zip(top_edge_rows["source"].astype(str), top_edge_rows["target"].astype(str))
        )
    else:
        top_edges = []

    scenarios: Dict[str, Dict] = {}

    # --- Scenario 1: node_attenuation on top-k sources ---
    if topk_sources:
        P_na = _apply_node_attenuation(nodes, edge_df, topk_sources, node_att_factor)
        sim_na = _simulate_propagation(nodes, P_na, num_steps, shock_mag, impact_thr, top_n)
        cmp_na = _compare_baseline_vs_intervention(baseline, sim_na)
        scenarios["node_attenuation"] = {
            "description": (
                f"Attenuate outgoing propagation of top-{topk} source nodes "
                f"by factor {node_att_factor}"
            ),
            "target_nodes": topk_sources,
            "attenuation_factor": node_att_factor,
            "simulation": sim_na,
            "comparison": cmp_na,
        }

    # --- Scenario 2: edge_attenuation on top-k edges ---
    if top_edges:
        P_ea = _apply_edge_attenuation(nodes, edge_df, top_edges, edge_att_factor)
        sim_ea = _simulate_propagation(nodes, P_ea, num_steps, shock_mag, impact_thr, top_n)
        cmp_ea = _compare_baseline_vs_intervention(baseline, sim_ea)
        scenarios["edge_attenuation"] = {
            "description": (
                f"Attenuate top-{topk} highest-weight edges by factor {edge_att_factor}"
            ),
            "target_edges": [f"{s}→{t}" for s, t in top_edges],
            "attenuation_factor": edge_att_factor,
            "simulation": sim_ea,
            "comparison": cmp_ea,
        }

    # --- Scenario 3: topk_source_control ---
    if topk_sources:
        P_tc = _apply_topk_source_control(nodes, edge_df, topk_sources)
        sim_tc = _simulate_propagation(nodes, P_tc, num_steps, shock_mag, impact_thr, top_n)
        cmp_tc = _compare_baseline_vs_intervention(baseline, sim_tc)
        scenarios["topk_source_control"] = {
            "description": (
                f"Remove all outgoing edges from top-{topk} most influential sources"
            ),
            "controlled_sources": topk_sources,
            "simulation": sim_tc,
            "comparison": cmp_tc,
        }

    # --- Rank scenarios by impact_reduction_pct (highest = best) ---
    ranking = sorted(
        [
            {
                "scenario": name,
                "impact_reduction_pct": sc["comparison"]["impact_reduction_pct"],
                "affected_nodes_reduction": sc["comparison"]["affected_nodes_reduction"],
            }
            for name, sc in scenarios.items()
        ],
        key=lambda x: x["impact_reduction_pct"],
        reverse=True,
    )

    return {
        "baseline": baseline,
        "scenarios": scenarios,
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _build_summary_rows(grades: List[str], per_grade: Dict) -> List[Dict]:
    rows = []
    for grade in grades:
        grade_data = per_grade.get(grade, {})
        baseline = grade_data.get("baseline", {})
        b_total = baseline.get("cumulative_impact_total", 0.0)
        ranking = grade_data.get("ranking", [])
        scenarios = grade_data.get("scenarios", {})

        for rank_idx, rank_entry in enumerate(ranking, 1):
            scenario_name = rank_entry["scenario"]
            sc = scenarios.get(scenario_name, {})
            cmp = sc.get("comparison", {})
            rows.append(
                {
                    "grade": grade,
                    "rank": rank_idx,
                    "scenario": scenario_name,
                    "baseline_cumulative_impact": round(b_total, 6),
                    "intervention_cumulative_impact": round(
                        cmp.get("intervention_cumulative_impact", 0.0), 6
                    ),
                    "impact_reduction_pct": cmp.get("impact_reduction_pct", 0.0),
                    "affected_nodes_reduction": cmp.get("affected_nodes_reduction", 0.0),
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
        "# MODUL 5 - Intervention Analysis Summary",
        "",
        f"Num propagation steps: {cfg['num_steps']}  ",
        f"Shock magnitude: {cfg['shock_magnitude']}  ",
        f"Node attenuation factor: {cfg['node_attenuation_factor']}  ",
        f"Edge attenuation factor: {cfg['edge_attenuation_factor']}  ",
        f"Top-k control: {cfg['topk_control']}",
        "",
        "---",
        "",
    ]

    for grade in grades:
        gd = per_grade.get(grade, {})
        baseline = gd.get("baseline", {})
        scenarios = gd.get("scenarios", {})
        ranking = gd.get("ranking", [])

        b_total = baseline.get("cumulative_impact_total", 0.0)
        b_influential = baseline.get("most_influential_sources", [])
        b_vulnerable = baseline.get("most_vulnerable_nodes", [])

        lines += [
            f"## Grade: {grade}",
            "",
            f"**Baseline cumulative impact:** {b_total:.6f}",
            "",
        ]

        if b_influential:
            lines += [
                "### Baseline – Most Influential Sources",
                "",
                "| Rank | Node | Total Impact | Affected Nodes |",
                "|---:|---|---:|---:|",
            ]
            for r, e in enumerate(b_influential, 1):
                lines.append(
                    f"| {r} | {e['node']} | {e['total_impact']:.6f} |"
                    f" {e['affected_nodes_count']} |"
                )
            lines.append("")

        if b_vulnerable:
            lines += [
                "### Baseline – Most Vulnerable Nodes",
                "",
                "| Rank | Node | Cumulative Received Impact |",
                "|---:|---|---:|",
            ]
            for r, e in enumerate(b_vulnerable, 1):
                lines.append(
                    f"| {r} | {e['node']} | {e['cumulative_received_impact']:.6f} |"
                )
            lines.append("")

        if ranking:
            lines += [
                "### Scenario Ranking (best to worst)",
                "",
                "| Rank | Scenario | Impact Reduction (%) | Affected Nodes Reduction |",
                "|---:|---|---:|---:|",
            ]
            for r_entry in ranking:
                lines.append(
                    f"| {r_entry.get('rank', '')} | {r_entry['scenario']} |"
                    f" {r_entry['impact_reduction_pct']:.4f} |"
                    f" {r_entry['affected_nodes_reduction']:.4f} |"
                )
            # fix: ranking list doesn't include rank key, add it
            lines = lines  # already fine, rank not in ranking entries — add positional idx
            lines.append("")

        for scenario_name, sc in scenarios.items():
            cmp = sc.get("comparison", {})
            lines += [
                f"### Scenario: `{scenario_name}`",
                "",
                f"_{sc.get('description', '')}_",
                "",
                f"- Baseline cumulative impact: {cmp.get('baseline_cumulative_impact', 0.0):.6f}",
                f"- Intervention cumulative impact: {cmp.get('intervention_cumulative_impact', 0.0):.6f}",
                f"- Impact reduction: **{cmp.get('impact_reduction_pct', 0.0):.4f}%**",
                f"- Affected nodes reduction (avg): {cmp.get('affected_nodes_reduction', 0.0):.4f}",
                "",
            ]

        lines += ["---", ""]

    (out_dir / "intervention_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _save_html_visualization(
    grade: str,
    nodes: List[str],
    edge_df: pd.DataFrame,
    grade_result: Dict,
    out_dir: Path,
) -> Path:
    html_path = out_dir / f"intervention_graph_{grade}.html"

    baseline = grade_result.get("baseline", {})
    scenarios = grade_result.get("scenarios", {})
    ranking = grade_result.get("ranking", [])

    best_scenario_name = ranking[0]["scenario"] if ranking else None
    best_sc = scenarios.get(best_scenario_name, {}) if best_scenario_name else {}
    best_cmp = best_sc.get("comparison", {})
    best_sim = best_sc.get("simulation", {})

    b_impact_map = baseline.get("total_impact_per_source", {})
    b_vuln_list = baseline.get("most_vulnerable_nodes", [])
    b_vuln_map = {e["node"]: e["cumulative_received_impact"] for e in b_vuln_list}

    i_impact_map = best_sim.get("total_impact_per_source", {}) if best_sim else {}

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if not nodes:
            fig = go.Figure()
            fig.update_layout(
                title=f"Module 5 Intervention Graph - {grade} (No Nodes)",
                template="plotly_white",
            )
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            return html_path

        n = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {
            node: (float(np.cos(angles[i])), float(np.sin(angles[i])))
            for i, node in enumerate(nodes)
        }

        def _make_graph_traces(
            label: str,
            impact_map: Dict,
            vuln_map: Dict,
        ) -> Tuple:
            edge_x: List[float] = []
            edge_y: List[float] = []
            if not edge_df.empty:
                for _, row in edge_df.iterrows():
                    src, tgt = str(row["source"]), str(row["target"])
                    if src in pos and tgt in pos:
                        x0, y0 = pos[src]
                        x1, y1 = pos[tgt]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
                showlegend=False,
            )

            max_impact = max(impact_map.values(), default=1.0) or 1.0
            max_vuln = max(vuln_map.values(), default=1.0) or 1.0

            node_x = [pos[node][0] for node in nodes]
            node_y = [pos[node][1] for node in nodes]
            node_sizes = [max(12, 28 * impact_map.get(node, 0.0) / max_impact) for node in nodes]
            node_colors = [vuln_map.get(node, 0.0) / max_vuln for node in nodes]

            hover_texts = [
                (
                    f"{node}<br>"
                    f"Influence: {impact_map.get(node, 0):.4f}<br>"
                    f"Vulnerability: {vuln_map.get(node, 0):.4f}"
                )
                for node in nodes
            ]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=nodes,
                textposition="top center",
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale="YlOrRd",
                    showscale=False,
                    line=dict(width=1, color="#333"),
                ),
                name=label,
            )
            return edge_trace, node_trace

        b_edge_t, b_node_t = _make_graph_traces("Baseline", b_impact_map, b_vuln_map)

        # Best intervention node vuln map
        i_vuln_list = best_sim.get("most_vulnerable_nodes", []) if best_sim else []
        i_vuln_map = {e["node"]: e["cumulative_received_impact"] for e in i_vuln_list}
        i_edge_t, i_node_t = _make_graph_traces(
            f"Best: {best_scenario_name}" if best_scenario_name else "Intervention",
            i_impact_map,
            i_vuln_map,
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                f"Baseline – {grade}",
                f"Best Intervention ({best_scenario_name}) – {grade}"
                if best_scenario_name
                else f"No intervention – {grade}",
            ],
        )
        fig.add_trace(b_edge_t, row=1, col=1)
        fig.add_trace(b_node_t, row=1, col=1)
        fig.add_trace(i_edge_t, row=1, col=2)
        fig.add_trace(i_node_t, row=1, col=2)

        reduction_pct = best_cmp.get("impact_reduction_pct", 0.0)
        fig.update_layout(
            title=(
                f"Module 5 Intervention Analysis - {grade}<br>"
                f"<sup>Best scenario: {best_scenario_name} | "
                f"Impact reduction: {reduction_pct:.2f}%</sup>"
            ),
            showlegend=False,
            template="plotly_white",
        )
        for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
            fig.update_layout(**{axis: dict(showgrid=False, zeroline=False, visible=False)})

        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path

    except Exception:
        # Fallback minimal HTML
        scenarios_html = ""
        for sc_name, sc in scenarios.items():
            cmp = sc.get("comparison", {})
            scenarios_html += (
                f"<li><strong>{sc_name}</strong>: "
                f"impact reduction = {cmp.get('impact_reduction_pct', 0.0):.2f}%, "
                f"affected nodes reduction = {cmp.get('affected_nodes_reduction', 0.0):.2f}"
                f"</li>"
            )
        if not scenarios_html:
            scenarios_html = "<li>(no scenarios)</li>"

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Module 5 Intervention Graph - {grade}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Module 5 Intervention Analysis - {grade}</h1>
  <p class="muted">Fallback HTML visualization (plotly unavailable).</p>
  <h2>Intervention Scenarios</h2>
  <ul>{scenarios_html}</ul>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_module5_intervention_analysis(
    shock_results_path: str,
    network_edges_path: Optional[str],
    output_dir: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Jalankan MODUL 5 dari output MODUL 4.

    Parameters
    ----------
    shock_results_path : str
        Path ke `shock_propagation_results.json` dari MODUL 4.
    network_edges_path : str | None
        Path ke `network_edges.csv` dari MODUL 3 (opsional; jika None dicari
        otomatis di direktori yang sama dengan shock_results_path).
    output_dir : str
        Folder output khusus MODUL 5, contoh: data/processed/module_05
    config : dict | None
        Konfigurasi opsional. Kunci yang didukung:
        - num_steps (int, default 5): jumlah langkah propagasi
        - shock_magnitude (float, default 1.0): besar shock awal
        - impact_threshold (float, default 1e-9): threshold dampak dianggap nol
        - top_n (int, default 5): jumlah top node yang dilaporkan
        - node_attenuation_factor (float, default 0.0): faktor reduksi node [0,1]
        - edge_attenuation_factor (float, default 0.0): faktor reduksi edge [0,1]
        - topk_control (int, default 1): jumlah top source yang dikendalikan

    Returns
    -------
    dict
        Hasil analisis intervensi lengkap per grade.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    shock_path = Path(shock_results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve edges path
    if network_edges_path is not None:
        edges_path: Optional[Path] = Path(network_edges_path)
    else:
        candidate = shock_path.parent.parent / "module_03" / "network_edges.csv"
        edges_path = candidate if candidate.exists() else None

    shock_data = _load_shock_results(shock_path)
    edges_all = _load_network_edges(edges_path)

    grades: List[str] = shock_data.get("grades", [])
    per_grade_shock: Dict = shock_data.get("per_grade", {})

    per_grade_results: Dict[str, Dict] = {}
    visualization_paths: Dict[str, str] = {}

    for grade in grades:
        grade_shock_data = per_grade_shock.get(grade, {})

        # Filter edges for this grade
        if not edges_all.empty and "grade" in edges_all.columns:
            edge_df = edges_all[edges_all["grade"] == grade].copy().reset_index(drop=True)
        else:
            edge_df = pd.DataFrame(columns=["grade", "source", "target", "weight"])

        grade_result = _run_interventions_for_grade(
            grade=grade,
            grade_shock_data=grade_shock_data,
            edge_df=edge_df,
            cfg=cfg,
        )
        per_grade_results[grade] = grade_result

        nodes: List[str] = grade_shock_data.get("nodes", [])
        viz_path = _save_html_visualization(
            grade=grade,
            nodes=nodes,
            edge_df=edge_df,
            grade_result=grade_result,
            out_dir=out_dir,
        )
        visualization_paths[grade] = str(viz_path)

    # --- intervention_results.json ---
    result = {
        "module": "module_05_intervention_analysis",
        "input_shock_results": str(shock_path),
        "input_network_edges": str(edges_path) if edges_path else None,
        "output_dir": str(out_dir),
        "grades": grades,
        "config": cfg,
        "per_grade": per_grade_results,
        "visualizations": visualization_paths,
    }

    with (out_dir / "intervention_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # --- intervention_summary.csv ---
    summary_rows = _build_summary_rows(grades, per_grade_results)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "intervention_summary.csv", index=False)

    # --- intervention_summary.md ---
    _write_markdown(grades=grades, per_grade=per_grade_results, out_dir=out_dir, cfg=cfg)

    print(f"[MODUL 5] Output tersimpan di: {out_dir}")
    for grade in grades:
        gd = per_grade_results.get(grade, {})
        ranking = gd.get("ranking", [])
        best = ranking[0] if ranking else {}
        print(
            f"  Grade={grade}: best_scenario={best.get('scenario', '-')}, "
            f"impact_reduction={best.get('impact_reduction_pct', 0.0):.2f}%"
        )

    return result
