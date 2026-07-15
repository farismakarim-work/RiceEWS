"""
MODUL 3 - Integrated Network Inference
======================================

Recover one integrated direct-cause graph over nodes ``(market_id, grade)``
using Algorithm 1 from Kinnear & Mazumdar (2023).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_granger_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"Invalid granger_causes value: {value!r}")


def _load_granger_json(granger_json_path: Path) -> Dict:
    if not granger_json_path.exists():
        raise FileNotFoundError(f"File granger_results.json tidak ditemukan: {granger_json_path}")
    with granger_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format granger_results.json tidak valid (harus object/dict).")
    return data


def _filter_granger_payload(data: Dict, markets: Optional[List[int]] = None, grades: Optional[List[str]] = None) -> Dict:
    """Filter integrated Module 2 payload by node attributes before Algorithm 1."""
    if not markets and not grades:
        return data

    filtered = dict(data)
    market_set = {int(market) for market in markets} if markets else None
    grade_set = {str(grade) for grade in grades} if grades else None

    nodes = data.get("nodes", [])
    kept_nodes = []
    kept_ids = set()
    for record in nodes:
        market_ok = market_set is None or int(record.get("market_id")) in market_set
        grade_ok = grade_set is None or str(record.get("grade")) in grade_set
        if market_ok and grade_ok:
            kept_nodes.append(record)
            kept_ids.add(str(record.get("node_id")))

    pairwise = data.get("pairwise_tests", {})
    kept_pairwise = {}
    for relation, result in pairwise.items():
        parsed = _parse_relation(relation)
        if parsed is None:
            continue
        source_node, target_node = parsed
        if source_node in kept_ids and target_node in kept_ids:
            kept_pairwise[relation] = result

    filtered["nodes"] = kept_nodes
    filtered["pairwise_tests"] = kept_pairwise
    return filtered


def _parse_relation(relation: str) -> Optional[Tuple[str, str]]:
    rel = str(relation).replace("->", "→")
    if "→" not in rel:
        return None
    source, target = rel.split("→", 1)
    return source.strip(), target.strip()


def _split_node_id(node_id: str) -> Tuple[int, str]:
    if not str(node_id).startswith("M") or "_" not in str(node_id):
        raise ValueError(f"Invalid node identifier: {node_id}")
    market_part, grade = str(node_id)[1:].split("_", 1)
    return int(market_part), grade


def _build_metadata_lookup(pairwise_tests: Dict) -> Dict[Tuple[str, str], Dict]:
    metadata: Dict[Tuple[str, str], Dict] = {}
    for relation, result in pairwise_tests.items():
        parsed = _parse_relation(relation)
        if parsed is None:
            continue
        source_node, target_node = parsed
        source_market, source_grade = _split_node_id(source_node)
        target_market, target_grade = _split_node_id(target_node)
        metadata[(source_node, target_node)] = {
            "source_node": source_node,
            "target_node": target_node,
            "source": source_market,
            "target": target_market,
            "grade_source": source_grade,
            "grade_target": target_grade,
            "lag": result.get("lag_order"),
            "p_value": result.get("p_value"),
            "adjusted_p_value": result.get("p_value_bh"),
            "test_statistic": result.get("f_statistic"),
            "f_statistic": result.get("f_statistic"),
            "relationship_type": (
                "within_grade" if source_grade == target_grade else "cross_grade"
            ),
            "direction": "source_to_target",
            "granger_causes": _parse_granger_flag(result.get("granger_causes", False)),
        }
    return metadata


def _build_ancestor_set_W(
    pairwise_tests: Dict,
) -> Tuple[set[Tuple[str, str]], set[frozenset], Dict[Tuple[str, str], Dict]]:
    metadata_lookup = _build_metadata_lookup(pairwise_tests)
    all_causes = {
        edge: metadata
        for edge, metadata in metadata_lookup.items()
        if metadata["granger_causes"]
    }

    excluded_bidirectional: set[frozenset] = set()
    to_exclude: set[Tuple[str, str]] = set()
    for source_node, target_node in list(all_causes.keys()):
        if (target_node, source_node) in all_causes:
            to_exclude.add((source_node, target_node))
            to_exclude.add((target_node, source_node))
            excluded_bidirectional.add(frozenset({source_node, target_node}))

    W = {edge for edge in all_causes if edge not in to_exclude}
    W_metadata = {edge: all_causes[edge] for edge in W}
    return W, excluded_bidirectional, W_metadata


def _compute_level_partition(W: set[Tuple[str, str]], nodes: List[str]) -> Dict[str, int]:
    if not nodes:
        return {}

    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    in_deg: Dict[str, int] = {node: 0 for node in nodes}
    for source_node, target_node in W:
        adj[source_node].append(target_node)
        in_deg[target_node] += 1

    queue = [node for node in nodes if in_deg[node] == 0]
    topo_order: List[str] = []
    in_deg_copy = in_deg.copy()
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for successor in adj[node]:
            in_deg_copy[successor] -= 1
            if in_deg_copy[successor] == 0:
                queue.append(successor)

    if len(topo_order) != len(nodes):
        raise ValueError("Integrated ancestor set W contains a directed cycle.")

    levels = {node: 0 for node in nodes}
    for node in topo_order:
        for successor in adj[node]:
            if levels[successor] < levels[node] + 1:
                levels[successor] = levels[node] + 1
    return levels


def _can_reach_via_E(source_node: str, target_node: str, E: set[Tuple[str, str]]) -> bool:
    adjacency: Dict[str, List[str]] = {}
    for source, target in E:
        adjacency.setdefault(source, []).append(target)

    visited: set[str] = set()
    stack: List[str] = [source_node]
    while stack:
        node = stack.pop()
        if node == target_node:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, []))
    return False


def _pairwise_recovery_algorithm1(
    W: set[Tuple[str, str]],
    nodes: List[str],
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    if not W:
        return [], {node: 0 for node in nodes}

    levels = _compute_level_partition(W, nodes)
    max_level = max(levels.values()) if levels else 0
    partitions: Dict[int, List[str]] = {}
    for node, level in levels.items():
        partitions.setdefault(level, []).append(node)

    E: set[Tuple[str, str]] = set()
    for k in range(1, max_level + 1):
        for target_node in partitions.get(k, []):
            for r in range(1, k + 1):
                source_level = k - r
                for source_node in partitions.get(source_level, []):
                    if (source_node, target_node) not in W:
                        continue
                    if _can_reach_via_E(source_node, target_node, E):
                        continue
                    E.add((source_node, target_node))

    return sorted(E), levels


def _prune_cyclic_ancestor_relations(
    W: set[Tuple[str, str]],
) -> Tuple[set[Tuple[str, str]], List[Tuple[str, str]]]:
    """Drop ancestor relations that remain inside directed cycles."""

    if not W:
        return W, []

    graph = nx.DiGraph()
    graph.add_edges_from(W)
    removed_edges: List[Tuple[str, str]] = []

    for component in nx.strongly_connected_components(graph):
        if len(component) <= 1:
            continue
        component_nodes = set(component)
        for edge in list(W):
            if edge[0] in component_nodes and edge[1] in component_nodes:
                removed_edges.append(edge)

    if not removed_edges:
        return W, []

    pruned_W = {edge for edge in W if edge not in set(removed_edges)}
    logger.warning(
        "Dropped %d cyclic ancestor relation(s) before Algorithm 1 because they violate the ancestor partial-order assumption.",
        len(removed_edges),
    )
    return pruned_W, sorted(removed_edges)


def _edges_to_dataframe(
    edges: List[Tuple[str, str]],
    metadata_lookup: Dict[Tuple[str, str], Dict],
) -> pd.DataFrame:
    rows = []
    for edge in edges:
        metadata = metadata_lookup[edge].copy()
        rows.append(metadata)

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "grade_source",
                "grade_target",
                "source_node",
                "target_node",
                "lag",
                "p_value",
                "adjusted_p_value",
                "test_statistic",
                "f_statistic",
                "relationship_type",
                "direction",
            ]
        )

    return pd.DataFrame(rows)


def _detect_cycle(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return not nx.is_directed_acyclic_graph(graph)


def _is_polytree(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(graph) and nx.is_forest(graph.to_undirected())


def _build_node_metrics(nodes: List[str], edge_df: pd.DataFrame) -> pd.DataFrame:
    out_degree = edge_df.groupby("source_node").size().to_dict() if not edge_df.empty else {}
    in_degree = edge_df.groupby("target_node").size().to_dict() if not edge_df.empty else {}
    weighted_out = edge_df.groupby("source_node")["test_statistic"].sum().to_dict() if not edge_df.empty else {}

    rows = []
    for node in nodes:
        market_id, grade = _split_node_id(node)
        od = int(out_degree.get(node, 0))
        idg = int(in_degree.get(node, 0))
        wod = float(weighted_out.get(node, 0.0))
        rows.append(
            {
                "node": node,
                "market_id": market_id,
                "grade": grade,
                "out_degree": od,
                "in_degree": idg,
                "weighted_out_degree": wod,
                "leader_score": od,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["leader_score", "weighted_out_degree", "in_degree", "node"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def _save_html_visualization(name: str, nodes: List[str], edge_df: pd.DataFrame, out_dir: Path) -> Path:
    html_path = out_dir / f"{name}.html"

    try:
        import plotly.graph_objects as go

        if not nodes:
            fig = go.Figure()
            fig.update_layout(title=name.replace("_", " ").title(), template="plotly_white")
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            return html_path

        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        positions = {
            node: (float(np.cos(angles[i])), float(np.sin(angles[i])))
            for i, node in enumerate(nodes)
        }

        edge_x: List[float] = []
        edge_y: List[float] = []
        for _, row in edge_df.iterrows():
            x0, y0 = positions[row["source_node"]]
            x1, y1 = positions[row["target_node"]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"))
        node_trace = go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=18, color="#1f77b4"),
            hovertext=nodes,
            hoverinfo="text",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=name.replace("_", " ").title(),
            showlegend=False,
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return html_path
    except Exception:
        edge_lines = "".join(
            f"<li>{row['source_node']} → {row['target_node']}</li>"
            for _, row in edge_df.iterrows()
        ) or "<li>(no edges)</li>"
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{name}</title></head>
<body><h1>{name}</h1><p>Nodes: {len(nodes)}</p><ul>{edge_lines}</ul></body></html>"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


def run_module3_network_inference(
    granger_json_path: str,
    output_dir: str,
    enforce_dag: bool = True,
    markets: Optional[List[int]] = None,
    grades: Optional[List[str]] = None,
) -> Dict:
    """
    Recover one integrated graph using Algorithm 1 only.

    ``enforce_dag`` is retained for API compatibility but no longer applies any
    extra heuristic cycle-breaking step.
    """

    granger_path = Path(granger_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_granger_json(granger_path)
    data = _filter_granger_payload(data, markets=markets, grades=grades)
    if data.get("analysis_type") != "integrated":
        raise ValueError("Module 3 expects integrated Module 2 output.")

    node_records = data.get("nodes", [])
    if not node_records:
        raise ValueError("No nodes remain for Module 3 after applying filters.")
    node_labels = [record["node_id"] for record in node_records]
    grades = sorted({record["grade"] for record in node_records})
    pairwise_tests = data.get("pairwise_tests", {})

    W, excluded_bidirectional, metadata_lookup = _build_ancestor_set_W(pairwise_tests)
    W, excluded_cycle_relations = _prune_cyclic_ancestor_relations(W)
    recovered_edges, levels = _pairwise_recovery_algorithm1(W, node_labels)
    edge_df = _edges_to_dataframe(recovered_edges, metadata_lookup)

    edge_pairs = [(row["source_node"], row["target_node"]) for _, row in edge_df.iterrows()]
    if _detect_cycle(node_labels, edge_pairs):
        raise ValueError("Algorithm 1 recovered a cyclic integrated graph; no extra heuristic applied.")

    metrics = _build_node_metrics(node_labels, edge_df)
    leaders_per_grade: Dict[str, List[Dict]] = {}
    visualization_paths: Dict[str, str] = {}
    summary_rows: List[Dict] = []

    integrated_viz = _save_html_visualization("network_graph_integrated", node_labels, edge_df, out_dir)
    visualization_paths["integrated"] = str(integrated_viz)

    for grade in grades:
        grade_nodes = [record["node_id"] for record in node_records if record["grade"] == grade]
        grade_metrics = metrics[metrics["grade"] == grade].reset_index(drop=True)
        leaders_per_grade[grade] = grade_metrics.to_dict(orient="records")
        grade_metrics.to_csv(out_dir / f"market_leaders_{grade}.csv", index=False)

        grade_edge_df = edge_df[
            (edge_df["grade_source"] == grade) | (edge_df["grade_target"] == grade)
        ].copy()
        grade_viz = _save_html_visualization(f"network_graph_{grade}", grade_nodes, grade_edge_df, out_dir)
        visualization_paths[grade] = str(grade_viz)

        summary_rows.append(
            {
                "grade": grade,
                "nodes": len(grade_nodes),
                "edges": int(len(grade_edge_df)),
                "out_edges": int((edge_df["grade_source"] == grade).sum()) if not edge_df.empty else 0,
                "in_edges": int((edge_df["grade_target"] == grade).sum()) if not edge_df.empty else 0,
                "cross_grade_edges": int(
                    (
                        ((edge_df["grade_source"] == grade) | (edge_df["grade_target"] == grade))
                        & (edge_df["grade_source"] != edge_df["grade_target"])
                    ).sum()
                )
                if not edge_df.empty
                else 0,
                "is_dag": True,
                "is_polytree": _is_polytree(node_labels, edge_pairs),
                "top_leader": grade_metrics.iloc[0]["node"] if not grade_metrics.empty else None,
                "top_leader_score": float(grade_metrics.iloc[0]["leader_score"]) if not grade_metrics.empty else 0.0,
            }
        )

    edge_df.to_csv(out_dir / "network_edges.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "network_summary.csv", index=False)
    metrics.to_csv(out_dir / "market_leaders_integrated.csv", index=False)

    result = {
        "module": "module_03_network_inference",
        "algorithm": "Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery",
        "input": str(granger_path),
        "output_dir": str(out_dir),
        "analysis_type": "integrated",
        "nodes": node_records,
        "levels": levels,
        "ancestor_relations": len(W),
        "recovered_edges": len(recovered_edges),
        "excluded_bidirectional_pairs": [sorted(list(pair)) for pair in excluded_bidirectional],
        "excluded_cycle_relations": [list(edge) for edge in excluded_cycle_relations],
        "summary": summary_rows,
        "leaders_per_grade": leaders_per_grade,
        "integrated_leaders": metrics.to_dict(orient="records"),
        "visualizations": visualization_paths,
        "enforce_dag": enforce_dag,
        "dag_enforcement_applied": False,
    }
    with (out_dir / "network_inference_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md_lines = [
        "# MODUL 3 - Network Inference Summary",
        "",
        "**Algorithm**: Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery",
        "",
        "> One integrated graph is recovered over nodes `(market_id, grade)`.",
        "> No direct edge thresholding or post-Algorithm-1 cycle-breaking heuristic is applied.",
        "> Ancestor relations that still form directed cycles are dropped before Algorithm 1",
        "> because they violate the ancestor partial-order assumption required by the paper.",
        "",
        f"Input: `{granger_path}`",
        f"Output dir: `{out_dir}`",
        "",
        "## Per Grade Summary",
        "",
        "| Grade | Nodes | Incident Edges | Out Edges | In Edges | Cross-Grade Edges | DAG | Polytree | Top Leader |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['grade']} | {row['nodes']} | {row['edges']} | {row['out_edges']} | "
            f"{row['in_edges']} | {row['cross_grade_edges']} | {row['is_dag']} | "
            f"{row['is_polytree']} | {row['top_leader']} |"
        )
    (out_dir / "network_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n[MODUL 3] Output tersimpan di: {out_dir}")
    print(
        f"  Integrated graph: nodes={len(node_labels)}, edges={len(recovered_edges)}, "
        f"ancestor_relations={len(W)}"
    )

    return result
