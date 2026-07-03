"""
MODUL 3 - Network Inference
===========================

Membangun struktur jaringan dari output Granger (MODUL 2), mengecek properti
DAG/polytree, dan menghasilkan ringkasan kandidat market leader berbasis
metrik graph-level sederhana.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_granger_json(granger_json_path: Path) -> Dict:
    if not granger_json_path.exists():
        raise FileNotFoundError(f"File granger_results.json tidak ditemukan: {granger_json_path}")
    with granger_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format granger_results.json tidak valid (harus object/dict).")
    return data


def _extract_edges_for_grade(grade_data: Dict, grade: str) -> pd.DataFrame:
    pairwise = grade_data.get("pairwise_tests", {})
    rows: List[Dict] = []

    for relation, result in pairwise.items():
        # expected format relation: "M102→M101" or similar
        rel = str(relation).replace("->", "→")
        if "→" in rel:
            source, target = rel.split("→", 1)
        else:
            # fallback: try underscore format
            parts = rel.replace(" ", "").split("_")
            if len(parts) >= 2:
                source, target = parts[0], parts[1]
            else:
                continue

        granger_causes = bool(result.get("granger_causes", False))
        f_stat = float(result.get("f_statistic", 0.0) or 0.0)
        p_val = float(result.get("p_value", 1.0) or 1.0)

        if granger_causes:
            rows.append(
                {
                    "grade": grade,
                    "source": source,
                    "target": target,
                    "f_statistic": f_stat,
                    "p_value": p_val,
                    "weight": f_stat,
                }
            )

    return pd.DataFrame(rows)


def _detect_cycle(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    graph = {n: [] for n in nodes}
    for s, t in edges:
        graph.setdefault(s, []).append(t)

    visited = set()
    visiting = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nei in graph.get(node, []):
            if dfs(nei):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for n in nodes:
        if dfs(n):
            return True
    return False


def _is_polytree(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    # Polytree sederhana: DAG + in-degree tiap node <= 1
    if _detect_cycle(nodes, edges):
        return False

    indeg = {n: 0 for n in nodes}
    for _, t in edges:
        indeg[t] = indeg.get(t, 0) + 1
    return all(v <= 1 for v in indeg.values())


def _build_node_metrics(nodes: List[str], edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            {
                "node": nodes,
                "out_degree": [0] * len(nodes),
                "in_degree": [0] * len(nodes),
                "weighted_out_degree": [0.0] * len(nodes),
                "leader_score": [0.0] * len(nodes),
            }
        )

    out_degree = edges.groupby("source").size().to_dict()
    in_degree = edges.groupby("target").size().to_dict()
    weighted_out = edges.groupby("source")["weight"].sum().to_dict()

    rows = []
    for n in nodes:
        od = int(out_degree.get(n, 0))
        idg = int(in_degree.get(n, 0))
        wod = float(weighted_out.get(n, 0.0))
        # Leader score sederhana: weighted_out + out_degree - in_degree
        leader_score = wod + od - idg
        rows.append(
            {
                "node": n,
                "out_degree": od,
                "in_degree": idg,
                "weighted_out_degree": wod,
                "leader_score": leader_score,
            }
        )

    df = pd.DataFrame(rows).sort_values("leader_score", ascending=False).reset_index(drop=True)
    return df


def _save_html_visualization(grade: str, nodes: List[str], edge_df: pd.DataFrame, out_dir: Path) -> Path:
    html_path = out_dir / f"network_graph_{grade}.html"

    node_count = len(nodes)

    # Safety: restrict edge_df to edges whose endpoints are both in the provided nodes list.
    # This prevents any shared/global DataFrame from bleeding across grades.
    if not edge_df.empty:
        node_set = set(nodes)
        edge_df = edge_df[
            edge_df["source"].isin(node_set) & edge_df["target"].isin(node_set)
        ].copy()

    edge_count = int(len(edge_df))
    title = f"Module 3 Network Graph - {grade} | Nodes={node_count} | Edges={edge_count}"
    debug_text = f"Debug Metadata | grade={grade} | node_count={node_count} | edge_count={edge_count}"

    try:
        import plotly.graph_objects as go

        debug_annotation = dict(
            text=debug_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.05,
            showarrow=False,
            font=dict(size=10, color="#666"),
            align="center",
        )

        if not nodes:
            fig = go.Figure()
            fig.update_layout(
                title=title,
                template="plotly_white",
                annotations=[debug_annotation],
            )
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            return html_path

        n = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {
            node: (float(np.cos(angles[i])), float(np.sin(angles[i])))
            for i, node in enumerate(nodes)
        }

        edge_x: List[float] = []
        edge_y: List[float] = []
        for _, row in edge_df.iterrows():
            x0, y0 = pos[row["source"]]
            x1, y1 = pos[row["target"]]
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

        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]

        out_degree = edge_df.groupby("source").size().to_dict() if not edge_df.empty else {}
        in_degree = edge_df.groupby("target").size().to_dict() if not edge_df.empty else {}
        labels = [
            f"{node}<br>Out-degree: {int(out_degree.get(node, 0))}<br>In-degree: {int(in_degree.get(node, 0))}"
            for node in nodes
        ]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            hovertext=labels,
            hoverinfo="text",
            marker=dict(size=20, color="#1f77b4", line=dict(width=1, color="#0d3b66")),
            name="markets",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            annotations=[debug_annotation],
        )
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path

    except Exception:
        # Fallback minimal HTML, tetap memenuhi syarat minimal satu output visualisasi
        nodes_html = "".join([f"<li>{nd}</li>" for nd in nodes]) or "<li>(no nodes)</li>"
        edges_html = ""
        if edge_df.empty:
            edges_html = "<li>(no significant edges)</li>"
        else:
            edges_html = "".join(
                [
                    f"<li>{row['source']} → {row['target']} (F={row['f_statistic']:.4f}, p={row['p_value']:.4f})</li>"
                    for _, row in edge_df.iterrows()
                ]
            )

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #666; }}
    .debug-meta {{ background: #f5f5f5; border: 1px solid #ddd; padding: 12px; margin-bottom: 16px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="muted">Fallback HTML visualization summary (plotly unavailable).</p>

  <div class="debug-meta">
    <strong>Debug Metadata</strong><br/>
    grade: {grade}<br/>
    node_count: {node_count}<br/>
    edge_count: {edge_count}
  </div>

  <h2>Nodes</h2>
  <ul>{nodes_html}</ul>

  <h2>Significant Directed Edges</h2>
  <ul>{edges_html}</ul>
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


def run_module3_network_inference(
    granger_json_path: str,
    output_dir: str,
) -> Dict:
    """
    Jalankan MODUL 3 dari output MODUL 2.

    Parameters
    ----------
    granger_json_path : str
        Path ke `granger_results.json` dari MODUL 2.
    output_dir : str
        Folder output khusus MODUL 3, contoh: data/processed/module_03
    """
    granger_path = Path(granger_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_granger_json(granger_path)

    # grades expected as top-level keys
    grades = list(data.keys())

    all_edges = []
    leaders_per_grade: Dict[str, List[Dict]] = {}
    summary_rows = []
    visualization_paths: Dict[str, str] = {}

    for grade in grades:
        grade_data = data.get(grade, {})
        edge_df = _extract_edges_for_grade(grade_data, grade)
        all_edges.append(edge_df)

        nodes = sorted(set(edge_df["source"].tolist() + edge_df["target"].tolist())) if not edge_df.empty else []
        edge_pairs = list(zip(edge_df["source"], edge_df["target"])) if not edge_df.empty else []

        is_dag = not _detect_cycle(nodes, edge_pairs) if nodes else True
        is_poly = _is_polytree(nodes, edge_pairs) if nodes else True

        metrics = _build_node_metrics(nodes, edge_df)
        top_leaders = metrics.head(5).to_dict(orient="records")
        leaders_per_grade[grade] = top_leaders

        summary_rows.append(
            {
                "grade": grade,
                "nodes": len(nodes),
                "edges": int(len(edge_df)),
                "is_dag": is_dag,
                "is_polytree": is_poly,
                "top_leader": top_leaders[0]["node"] if top_leaders else None,
                "top_leader_score": float(top_leaders[0]["leader_score"]) if top_leaders else 0.0,
            }
        )

        metrics.to_csv(out_dir / f"market_leaders_{grade}.csv", index=False)
        viz_path = _save_html_visualization(grade=grade, nodes=nodes, edge_df=edge_df, out_dir=out_dir)
        visualization_paths[grade] = str(viz_path)

    edges_df = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame(
        columns=["grade", "source", "target", "f_statistic", "p_value", "weight"]
    )
    edges_df.to_csv(out_dir / "network_edges.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "network_summary.csv", index=False)

    result = {
        "module": "module_03_network_inference",
        "input": str(granger_path),
        "output_dir": str(out_dir),
        "grades": grades,
        "summary": summary_rows,
        "leaders_per_grade": leaders_per_grade,
        "visualizations": visualization_paths,
    }

    with (out_dir / "network_inference_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md_lines = [
        "# MODUL 3 - Network Inference Summary",
        "",
        f"Input: `{granger_path}`",
        f"Output dir: `{out_dir}`",
        "",
        "## Per Grade Summary",
        "",
        "| Grade | Nodes | Edges | DAG | Polytree | Top Leader | Leader Score | Visualization |",
        "|---|---:|---:|:---:|:---:|---|---:|---|",
    ]
    for r in summary_rows:
        viz_name = Path(visualization_paths.get(r["grade"], "")).name if visualization_paths.get(r["grade"]) else "-"
        md_lines.append(
            f"| {r['grade']} | {r['nodes']} | {r['edges']} | {r['is_dag']} | {r['is_polytree']} | {r['top_leader']} | {r['top_leader_score']:.4f} | {viz_name} |"
        )

    (out_dir / "network_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return result
