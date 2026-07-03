"""
MODUL 4 - Shock Propagation Simulation
=======================================

Simulasi penyebaran shock pada graph berarah berbobot hasil MODUL 3.
Mengidentifikasi sumber shock paling berpengaruh dan node paling rentan.

Algoritma:
- Bangun matriks propagasi P dari edge berbobot (row-normalized per source).
- Untuk setiap node sebagai sumber shock, jalankan multi-step propagation.
- Hitung metrik: total_impact, vulnerability, reachability.

Output:
- shock_propagation_results.json
- shock_summary.csv
- shock_summary.md
- shock_graph_{grade}.html (per grade)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_network_results(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"network_inference_results.json tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format network_inference_results.json tidak valid (harus dict).")
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
    # Ensure weight is numeric and non-negative
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _build_propagation_matrix(
    nodes: List[str],
    edge_df: pd.DataFrame,
) -> np.ndarray:
    """
    Bangun matriks propagasi P[i][j] = bobot ternormalisasi dari node i ke node j.
    Row-normalisasi agar setiap sumber menyebarkan tepat 1 unit (bounded).
    """
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    P = np.zeros((n, n), dtype=float)

    for _, row in edge_df.iterrows():
        s = node_idx.get(str(row["source"]))
        t = node_idx.get(str(row["target"]))
        if s is not None and t is not None and s != t:
            P[s, t] += float(row["weight"])

    # Row-normalize: sum of outgoing weights from each source = 1
    row_sums = P.sum(axis=1, keepdims=True)
    non_zero = row_sums.ravel() > 0
    P[non_zero] = P[non_zero] / row_sums[non_zero]

    return P


def _simulate_grade(
    nodes: List[str],
    edge_df: pd.DataFrame,
    num_steps: int,
    shock_magnitude: float,
    impact_threshold: float,
    top_n: int,
) -> Dict:
    """
    Jalankan simulasi shock propagation untuk satu grade.

    Returns dict dengan kunci:
    - nodes, edges_count
    - simulation_results (per source)
    - most_influential_sources
    - most_vulnerable_nodes
    - total_impact_per_source
    - affected_nodes_count
    """
    n = len(nodes)

    if n == 0 or edge_df.empty:
        return _empty_grade_result(nodes, edge_df)

    P = _build_propagation_matrix(nodes, edge_df)

    simulation_results: Dict[str, Dict] = {}
    # Accumulate received impact across all simulations to find vulnerable nodes
    cumulative_received = np.zeros(n, dtype=float)

    for src_idx, src_node in enumerate(nodes):
        current = np.zeros(n, dtype=float)
        current[src_idx] = shock_magnitude

        total_impact_vec = np.zeros(n, dtype=float)
        step_impacts: List[float] = []

        for _step in range(num_steps):
            current = P.T @ current
            total_impact_vec += current
            step_impacts.append(float(current.sum()))

        cumulative_received += total_impact_vec

        # Impact per node (exclude self by convention)
        impact_per_node = {
            nodes[j]: float(total_impact_vec[j])
            for j in range(n)
        }
        affected_count = int(np.sum(total_impact_vec > impact_threshold))
        total_impact_value = float(total_impact_vec.sum())

        simulation_results[src_node] = {
            "total_impact": total_impact_value,
            "affected_nodes_count": affected_count,
            "impact_per_node": impact_per_node,
            "step_impacts": step_impacts,
        }

    # Rank sources by total impact
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

    # Rank nodes by cumulative received impact (vulnerability)
    vulnerability_ranked = sorted(
        enumerate(cumulative_received), key=lambda x: x[1], reverse=True
    )
    most_vulnerable = [
        {
            "node": nodes[i],
            "cumulative_received_impact": float(v),
        }
        for i, v in vulnerability_ranked[:top_n]
    ]

    total_impact_per_source = {
        node: info["total_impact"] for node, info in simulation_results.items()
    }
    affected_nodes_count = {
        node: info["affected_nodes_count"] for node, info in simulation_results.items()
    }

    return {
        "nodes": nodes,
        "edges_count": int(len(edge_df)),
        "simulation_results": simulation_results,
        "total_impact_per_source": total_impact_per_source,
        "affected_nodes_count": affected_nodes_count,
        "most_influential_sources": most_influential,
        "most_vulnerable_nodes": most_vulnerable,
    }


def _empty_grade_result(nodes: List[str], edge_df: pd.DataFrame) -> Dict:
    return {
        "nodes": nodes,
        "edges_count": int(len(edge_df)),
        "simulation_results": {},
        "total_impact_per_source": {},
        "affected_nodes_count": {},
        "most_influential_sources": [],
        "most_vulnerable_nodes": [],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _build_summary_rows(grades: List[str], per_grade: Dict) -> List[Dict]:
    rows = []
    for grade in grades:
        grade_res = per_grade.get(grade, {})
        for entry in grade_res.get("most_influential_sources", []):
            rows.append(
                {
                    "grade": grade,
                    "source_node": entry["node"],
                    "total_impact": entry["total_impact"],
                    "affected_nodes_count": entry["affected_nodes_count"],
                }
            )
    return rows


def _write_markdown(
    grades: List[str],
    per_grade: Dict,
    out_dir: Path,
    config: Dict,
) -> None:
    lines = [
        "# MODUL 4 - Shock Propagation Simulation Summary",
        "",
        f"Num propagation steps: {config['num_steps']}  ",
        f"Shock magnitude: {config['shock_magnitude']}",
        "",
        "---",
        "",
    ]

    for grade in grades:
        gr = per_grade.get(grade, {})
        lines += [
            f"## Grade: {grade}",
            "",
            f"- Nodes: {len(gr.get('nodes', []))}",
            f"- Edges: {gr.get('edges_count', 0)}",
            "",
        ]

        influential = gr.get("most_influential_sources", [])
        if influential:
            lines += [
                "### Most Influential Sources (highest shock impact)",
                "",
                "| Rank | Node | Total Impact | Affected Nodes |",
                "|---:|---|---:|---:|",
            ]
            for rank, entry in enumerate(influential, 1):
                lines.append(
                    f"| {rank} | {entry['node']} | {entry['total_impact']:.6f} |"
                    f" {entry['affected_nodes_count']} |"
                )
            lines.append("")

        vulnerable = gr.get("most_vulnerable_nodes", [])
        if vulnerable:
            lines += [
                "### Most Vulnerable Nodes (highest cumulative received impact)",
                "",
                "| Rank | Node | Cumulative Received Impact |",
                "|---:|---|---:|",
            ]
            for rank, entry in enumerate(vulnerable, 1):
                lines.append(
                    f"| {rank} | {entry['node']} |"
                    f" {entry['cumulative_received_impact']:.6f} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    (out_dir / "shock_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _save_html_visualization(
    grade: str,
    nodes: List[str],
    edge_df: pd.DataFrame,
    per_grade_result: Dict,
    out_dir: Path,
) -> Path:
    html_path = out_dir / f"shock_graph_{grade}.html"

    # Map node → metrics for coloring/sizing
    impact_map = per_grade_result.get("total_impact_per_source", {})
    vuln_list = per_grade_result.get("most_vulnerable_nodes", [])
    vuln_map = {entry["node"]: entry["cumulative_received_impact"] for entry in vuln_list}

    try:
        import plotly.graph_objects as go

        if not nodes:
            fig = go.Figure()
            fig.update_layout(
                title=f"Module 4 Shock Graph - {grade} (No Nodes)",
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

        # Edges
        edge_x: List[float] = []
        edge_y: List[float] = []
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
            name="edges",
        )

        # Node size proportional to influence (source impact)
        max_impact = max(impact_map.values(), default=1.0) or 1.0
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        node_sizes = [
            max(14, 30 * impact_map.get(node, 0.0) / max_impact) for node in nodes
        ]
        # Node color proportional to vulnerability
        max_vuln = max(vuln_map.values(), default=1.0) or 1.0
        node_colors = [vuln_map.get(node, 0.0) / max_vuln for node in nodes]

        hover_texts = [
            (
                f"{node}<br>"
                f"Influence (total impact as source): {impact_map.get(node, 0):.4f}<br>"
                f"Vulnerability (received impact): {vuln_map.get(node, 0):.4f}"
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
                showscale=True,
                colorbar=dict(title="Vulnerability<br>(normalized)"),
                line=dict(width=1, color="#333"),
            ),
            name="markets",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Module 4 Shock Propagation - {grade}<br>"
                  "<sup>Node size = influence | Node color = vulnerability</sup>",
            showlegend=False,
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path

    except Exception:
        # Fallback minimal HTML
        nodes_html = "".join(
            [
                f"<li>{n} — influence: {impact_map.get(n, 0):.4f},"
                f" vulnerability: {vuln_map.get(n, 0):.4f}</li>"
                for n in nodes
            ]
        ) or "<li>(no nodes)</li>"

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Module 4 Shock Graph - {grade}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Module 4 Shock Propagation - {grade}</h1>
  <p class="muted">Fallback HTML visualization summary (plotly unavailable).</p>
  <h2>Node Influence &amp; Vulnerability</h2>
  <ul>{nodes_html}</ul>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_module4_shock_propagation(
    network_results_path: str,
    network_edges_path: Optional[str],
    output_dir: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Jalankan MODUL 4 dari output MODUL 3.

    Parameters
    ----------
    network_results_path : str
        Path ke `network_inference_results.json` dari MODUL 3.
    network_edges_path : str | None
        Path ke `network_edges.csv` dari MODUL 3 (opsional; jika None dicari
        otomatis di direktori yang sama dengan network_results_path).
    output_dir : str
        Folder output khusus MODUL 4, contoh: data/processed/module_04
    config : dict | None
        Konfigurasi opsional. Kunci yang didukung:
        - num_steps (int, default 5): jumlah langkah propagasi
        - shock_magnitude (float, default 1.0): besar shock awal
        - impact_threshold (float, default 1e-9): threshold dampak dianggap nol
        - top_n (int, default 5): jumlah top node yang dilaporkan

    Returns
    -------
    dict
        Hasil simulasi lengkap per grade.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    net_path = Path(network_results_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve edges path
    if network_edges_path is not None:
        edges_path: Optional[Path] = Path(network_edges_path)
    else:
        candidate = net_path.parent / "network_edges.csv"
        edges_path = candidate if candidate.exists() else None

    net_data = _load_network_results(net_path)
    edges_all = _load_network_edges(edges_path)

    grades: List[str] = net_data.get("grades", [])

    per_grade_results: Dict[str, Dict] = {}
    visualization_paths: Dict[str, str] = {}

    for grade in grades:
        # Filter edges for this grade
        if not edges_all.empty and "grade" in edges_all.columns:
            edge_df = edges_all[edges_all["grade"] == grade].copy().reset_index(drop=True)
        else:
            edge_df = pd.DataFrame(columns=["grade", "source", "target", "weight"])

        # Collect all nodes for this grade
        if not edge_df.empty:
            nodes = sorted(
                set(edge_df["source"].astype(str).tolist())
                | set(edge_df["target"].astype(str).tolist())
            )
        else:
            nodes = []

        grade_result = _simulate_grade(
            nodes=nodes,
            edge_df=edge_df,
            num_steps=int(cfg["num_steps"]),
            shock_magnitude=float(cfg["shock_magnitude"]),
            impact_threshold=float(cfg["impact_threshold"]),
            top_n=int(cfg["top_n"]),
        )
        per_grade_results[grade] = grade_result

        viz_path = _save_html_visualization(
            grade=grade,
            nodes=nodes,
            edge_df=edge_df,
            per_grade_result=grade_result,
            out_dir=out_dir,
        )
        visualization_paths[grade] = str(viz_path)

    # --- shock_propagation_results.json ---
    result = {
        "module": "module_04_shock_propagation",
        "input_network_results": str(net_path),
        "input_network_edges": str(edges_path) if edges_path else None,
        "output_dir": str(out_dir),
        "grades": grades,
        "config": cfg,
        "per_grade": per_grade_results,
        "visualizations": visualization_paths,
    }

    with (out_dir / "shock_propagation_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # --- shock_summary.csv ---
    summary_rows = _build_summary_rows(grades, per_grade_results)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["rank"] = (
            summary_df.groupby("grade")["total_impact"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        summary_df = summary_df.sort_values(["grade", "rank"]).reset_index(drop=True)
    summary_df.to_csv(out_dir / "shock_summary.csv", index=False)

    # --- shock_summary.md ---
    _write_markdown(grades=grades, per_grade=per_grade_results, out_dir=out_dir, config=cfg)

    print(f"[MODUL 4] Output tersimpan di: {out_dir}")
    for grade in grades:
        gr = per_grade_results.get(grade, {})
        top = gr.get("most_influential_sources", [{}])
        top_node = top[0].get("node", "-") if top else "-"
        top_impact = top[0].get("total_impact", 0.0) if top else 0.0
        vuln = gr.get("most_vulnerable_nodes", [{}])
        vuln_node = vuln[0].get("node", "-") if vuln else "-"
        print(
            f"  Grade={grade}: nodes={len(gr.get('nodes', []))}, "
            f"edges={gr.get('edges_count', 0)}, "
            f"top_source={top_node} (impact={top_impact:.4f}), "
            f"top_vuln={vuln_node}"
        )

    return result
