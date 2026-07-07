"""
MODUL 3 - Network Inference
===========================

Membangun struktur jaringan dari output Granger (MODUL 2), mengecek properti
DAG/polytree, dan menghasilkan ringkasan kandidat market leader berbasis
metrik graph-level sederhana.

Proses utama:
1. Baca granger_results.json dari MODUL 2.
2. Ekstrak significant edges (Granger-causes=True) per grade.
3. Terapkan cycle-breaking greedy (hapus edge ber-F-statistic terkecil dalam
   setiap siklus) hingga graf menjadi DAG.
4. Validasi properti DAG dan polytree.
5. Hitung metrik market leader (out-degree, in-degree, weighted out-degree).
6. Simpan output: network_edges.csv, market_leaders_{grade}.csv,
   network_inference_results.json, network_summary.{csv,md},
   network_graph_{grade}.html.

Catatan penggunaan:
- DAG enforcement (cycle-breaking) diaktifkan secara default
  (enforce_dag=True). Ini penting agar metrik centrality tidak bias akibat
  hubungan sirkular (A→B dan B→A secara bersamaan).
- Edges yang dihapus dilaporkan di kolom ``removed_edges`` pada output JSON
  per grade untuk transparansi/audit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_granger_json(granger_json_path: Path) -> Dict:
    """Load dan validasi granger_results.json dari MODUL 2."""
    if not granger_json_path.exists():
        raise FileNotFoundError(f"File granger_results.json tidak ditemukan: {granger_json_path}")
    with granger_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format granger_results.json tidak valid (harus object/dict).")
    return data


def _extract_edges_for_grade(grade_data: Dict, grade: str) -> pd.DataFrame:
    """
    Ekstrak significant causal edges dari pairwise Granger results untuk satu grade.

    Parameters
    ----------
    grade_data : dict
        Entry per grade dari granger_results.json.
    grade : str
        Nama grade (misal 'low1').

    Returns
    -------
    pd.DataFrame
        Kolom: grade, source, target, f_statistic, p_value, weight.
        Hanya berisi pasangan yang ``granger_causes=True``.
    """
    pairwise = grade_data.get("pairwise_tests", {})
    rows: List[Dict] = []

    for relation, result in pairwise.items():
        # Expected format: "M102→M101" atau "M102->M101"
        rel = str(relation).replace("->", "→")
        if "→" in rel:
            source, target = rel.split("→", 1)
        else:
            # Fallback: pisahkan berdasarkan underscore
            parts = rel.replace(" ", "").split("_")
            if len(parts) >= 2:
                source, target = parts[0], parts[1]
            else:
                logger.warning("Tidak dapat mem-parsing relasi: %s", relation)
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
    """Periksa apakah terdapat siklus pada directed graph menggunakan DFS."""
    graph: Dict[str, List[str]] = {n: [] for n in nodes}
    for s, t in edges:
        graph.setdefault(s, []).append(t)

    visited: set = set()
    visiting: set = set()

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
        if n not in visited and dfs(n):
            return True
    return False


def _find_one_cycle(nodes: List[str], edges: List[Tuple[str, str]]) -> Optional[List[str]]:
    """
    Temukan satu siklus dalam directed graph menggunakan DFS.

    Returns
    -------
    list of str atau None
        Urutan node yang membentuk siklus, atau None jika tidak ada siklus.
    """
    graph: Dict[str, List[str]] = {n: [] for n in nodes}
    for s, t in edges:
        graph.setdefault(s, []).append(t)

    visited: set = set()
    visiting: List[str] = []
    visiting_set: set = set()

    def dfs(node: str) -> Optional[List[str]]:
        if node in visiting_set:
            # Siklus ditemukan: ambil bagian yang membentuk siklus
            idx = visiting.index(node)
            return visiting[idx:]
        if node in visited:
            return None
        visiting.append(node)
        visiting_set.add(node)
        for nei in graph.get(node, []):
            result = dfs(nei)
            if result is not None:
                return result
        visiting.pop()
        visiting_set.discard(node)
        visited.add(node)
        return None

    for n in nodes:
        if n not in visited:
            result = dfs(n)
            if result is not None:
                return result
    return None


def _break_cycles_greedy(edge_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Hapus edge ber-F-statistic (weight) terkecil dari setiap siklus secara
    iteratif hingga graf menjadi DAG (Directed Acyclic Graph).

    Strategi greedy ini meminimalkan kehilangan informasi kausalitas dengan
    mempertahankan hubungan yang paling kuat (F-statistic terbesar) dan
    membuang yang paling lemah jika terjadi kontradiksi arah.

    Parameters
    ----------
    edge_df : pd.DataFrame
        DataFrame dengan kolom: source, target, weight (dan lainnya).

    Returns
    -------
    (dag_df, removed_edges)
        dag_df : pd.DataFrame — edges yang tersisa setelah cycle-breaking.
        removed_edges : list of (source, target) — edges yang dihapus.
    """
    if edge_df.empty:
        return edge_df.copy(), []

    df = edge_df.copy().reset_index(drop=True)
    removed_edges: List[Tuple[str, str]] = []
    max_iterations = len(df) + 1  # batas aman untuk mencegah infinite loop

    for _ in range(max_iterations):
        nodes = sorted(set(df["source"].tolist() + df["target"].tolist()))
        edge_pairs = list(zip(df["source"].astype(str), df["target"].astype(str)))

        if not _detect_cycle(nodes, edge_pairs):
            break  # Sudah DAG

        cycle = _find_one_cycle(nodes, edge_pairs)
        if cycle is None:
            break  # Tidak ada siklus yang dapat ditemukan

        # Identifikasi edges dalam siklus ini
        cycle_edge_set = set(
            (cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))
        )

        # Cari edge dalam siklus dengan weight (F-statistic) terkecil
        min_weight = float("inf")
        min_idx: Optional[int] = None

        for idx, row in df.iterrows():
            src = str(row["source"])
            tgt = str(row["target"])
            if (src, tgt) in cycle_edge_set:
                w = float(row.get("weight", row.get("f_statistic", 0.0)) or 0.0)
                if w < min_weight:
                    min_weight = w
                    min_idx = idx

        if min_idx is not None:
            removed_row = df.loc[min_idx]
            removed_edges.append((str(removed_row["source"]), str(removed_row["target"])))
            df = df.drop(index=min_idx).reset_index(drop=True)
        else:
            # Tidak ada edge yang dapat dihapus (seharusnya tidak terjadi)
            logger.warning("Cycle-breaking: tidak dapat menemukan edge untuk dihapus.")
            break

    return df, removed_edges


def _is_polytree(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """Cek apakah graf memenuhi polytree assumption: DAG + in-degree tiap node ≤ 1."""
    if _detect_cycle(nodes, edges):
        return False

    indeg: Dict[str, int] = {n: 0 for n in nodes}
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

    edge_count = len(edge_df)
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
        nodes_html = "".join([f"<li>{node}</li>" for node in nodes]) or "<li>(no nodes)</li>"
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
    enforce_dag: bool = True,
) -> Dict:
    """
    Jalankan MODUL 3 dari output MODUL 2.

    Proses:
    1. Baca granger_results.json.
    2. Ekstrak significant edges per grade.
    3. (Opsional) Terapkan cycle-breaking greedy agar output menjadi DAG.
    4. Hitung metrik market leader.
    5. Simpan semua output.

    Parameters
    ----------
    granger_json_path : str
        Path ke ``granger_results.json`` dari MODUL 2.
    output_dir : str
        Folder output MODUL 3, contoh: ``data/processed/module_03``.
    enforce_dag : bool, default True
        Jika True, terapkan cycle-breaking greedy sehingga graf output
        dijamin menjadi DAG. Edges yang dihapus dicatat di hasil JSON
        untuk keperluan audit. Ini **sangat direkomendasikan** agar
        metrik centrality (market leader identification) tidak bias.

    Returns
    -------
    dict
        Hasil inference lengkap per grade, termasuk path output.
    """
    granger_path = Path(granger_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_granger_json(granger_path)

    # Grades adalah top-level keys pada granger_results.json
    grades = list(data.keys())

    all_dag_edges: List[pd.DataFrame] = []
    leaders_per_grade: Dict[str, List[Dict]] = {}
    summary_rows: List[Dict] = []
    visualization_paths: Dict[str, str] = {}
    removed_edges_per_grade: Dict[str, List[Tuple[str, str]]] = {}

    for grade in grades:
        grade_data = data.get(grade, {})
        raw_edge_df = _extract_edges_for_grade(grade_data, grade)

        # --- Cycle-breaking: pastikan output adalah DAG ---
        if enforce_dag and not raw_edge_df.empty:
            raw_nodes = sorted(
                set(raw_edge_df["source"].tolist() + raw_edge_df["target"].tolist())
            )
            raw_pairs = list(zip(raw_edge_df["source"], raw_edge_df["target"]))
            has_cycle = _detect_cycle(raw_nodes, raw_pairs)

            if has_cycle:
                dag_edge_df, removed = _break_cycles_greedy(raw_edge_df)
                removed_edges_per_grade[grade] = removed
                if removed:
                    logger.info(
                        "Grade=%s: %d edge(s) dihapus untuk membentuk DAG: %s",
                        grade,
                        len(removed),
                        removed,
                    )
                    print(
                        f"  [DAG enforcement] Grade={grade}: {len(removed)} edge(s) "
                        f"dihapus untuk menghilangkan siklus."
                    )
                edge_df = dag_edge_df
            else:
                removed_edges_per_grade[grade] = []
                edge_df = raw_edge_df
        else:
            removed_edges_per_grade[grade] = []
            edge_df = raw_edge_df

        all_dag_edges.append(edge_df)

        nodes = sorted(
            set(edge_df["source"].tolist() + edge_df["target"].tolist())
        ) if not edge_df.empty else []
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
                "edges_removed_for_dag": len(removed_edges_per_grade.get(grade, [])),
                "is_dag": is_dag,
                "is_polytree": is_poly,
                "top_leader": top_leaders[0]["node"] if top_leaders else None,
                "top_leader_score": float(top_leaders[0]["leader_score"]) if top_leaders else 0.0,
            }
        )

        metrics.to_csv(out_dir / f"market_leaders_{grade}.csv", index=False)
        viz_path = _save_html_visualization(
            grade=grade, nodes=nodes, edge_df=edge_df, out_dir=out_dir
        )
        visualization_paths[grade] = str(viz_path)

    edges_df = pd.concat(all_dag_edges, ignore_index=True) if all_dag_edges else pd.DataFrame(
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
        "enforce_dag": enforce_dag,
        "summary": summary_rows,
        "leaders_per_grade": leaders_per_grade,
        "removed_edges_per_grade": {
            g: [list(e) for e in removed]
            for g, removed in removed_edges_per_grade.items()
        },
        "visualizations": visualization_paths,
    }

    with (out_dir / "network_inference_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md_lines = [
        "# MODUL 3 - Network Inference Summary",
        "",
        f"Input: `{granger_path}`",
        f"Output dir: `{out_dir}`",
        f"DAG enforcement: `{enforce_dag}`",
        "",
        "## Per Grade Summary",
        "",
        "| Grade | Nodes | Edges | Edges Removed | DAG | Polytree | Top Leader | Leader Score | Visualization |",
        "|---|---:|---:|---:|:---:|:---:|---|---:|---|",
    ]
    for r in summary_rows:
        viz_name = (
            Path(visualization_paths.get(r["grade"], "")).name
            if visualization_paths.get(r["grade"])
            else "-"
        )
        md_lines.append(
            f"| {r['grade']} | {r['nodes']} | {r['edges']} "
            f"| {r['edges_removed_for_dag']} "
            f"| {r['is_dag']} | {r['is_polytree']} | {r['top_leader']} "
            f"| {r['top_leader_score']:.4f} | {viz_name} |"
        )

    # Removed edges audit section
    if any(removed_edges_per_grade.values()):
        md_lines += [
            "",
            "## Edges Removed for DAG Enforcement",
            "",
            "| Grade | Source | Target |",
            "|---|---|---|",
        ]
        for grade, removed in removed_edges_per_grade.items():
            for src, tgt in removed:
                md_lines.append(f"| {grade} | {src} | {tgt} |")

    (out_dir / "network_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Console summary
    print(f"\n[MODUL 3] Output tersimpan di: {out_dir}")
    for r in summary_rows:
        print(
            f"  Grade={r['grade']}: nodes={r['nodes']}, edges={r['edges']}, "
            f"removed={r['edges_removed_for_dag']}, "
            f"DAG={r['is_dag']}, Polytree={r['is_polytree']}, "
            f"TopLeader={r['top_leader']}"
        )

    return result
