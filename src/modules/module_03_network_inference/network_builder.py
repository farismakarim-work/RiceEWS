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
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
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


# ---------------------------------------------------------------------------
# Algorithm 1 — Pairwise Recovery (Kinnear & Mazumdar 2023)
# ---------------------------------------------------------------------------

def _parse_relation(relation: str) -> Optional[Tuple[str, str]]:
    """Parse 'M102→M101' or 'M102->M101' into (source, target)."""
    rel = str(relation).replace("->", "→")
    if "→" in rel:
        source, target = rel.split("→", 1)
        return source.strip(), target.strip()
    parts = rel.replace(" ", "").split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _build_ancestor_set_W(
    pairwise_tests: Dict,
) -> Tuple[Set[Tuple[str, str]], Set[FrozenSet[str]], Dict[Tuple[str, str], float]]:
    """
    Build the ancestor set W and collect F-statistics from pairwise test results.

    **Step 1 of Algorithm 1** (Kinnear & Mazumdar 2023):

    - W = set of directed pairs (i, j) for which i pairwise Granger-causes j
      and j does *not* simultaneously pairwise Granger-cause i.
    - Bidirectional pairs (i→j AND j→i both significant) are excluded from W
      as *both* directions must be discarded (Corollary 2.3 / Remark 2.5):
      in a strongly causal graph, bidirectional pairwise causality is always
      caused by a confounder, making both edges invalid direct-cause edges.

    Parameters
    ----------
    pairwise_tests : dict
        Output of ``test_all_pairwise_relationships`` (keys: "M{x}→M{y}").

    Returns
    -------
    W : set of (source, target)
        Unambiguous ancestor relations (bidirectional pairs excluded).
    excluded_bidirectional : set of frozenset({src, tgt})
        Unordered pairs excluded because both directions were significant.
    f_stats : dict mapping (source, target) → F-statistic
        For edge weight in output DataFrame.
    """
    all_causes: Dict[Tuple[str, str], float] = {}

    for relation, result in pairwise_tests.items():
        if not result.get("granger_causes", False):
            continue
        parsed = _parse_relation(relation)
        if parsed is None:
            logger.warning("Tidak dapat mem-parsing relasi: %s", relation)
            continue
        source, target = parsed
        f_stat = float(result.get("f_statistic", 0.0) or 0.0)
        all_causes[(source, target)] = f_stat

    # Identify bidirectional pairs (both i→j and j→i significant) → exclude both
    to_exclude: Set[Tuple[str, str]] = set()
    excluded_bidirectional: Set[FrozenSet[str]] = set()
    for (src, tgt) in list(all_causes.keys()):
        if (tgt, src) in all_causes:
            to_exclude.add((src, tgt))
            to_exclude.add((tgt, src))
            excluded_bidirectional.add(frozenset({src, tgt}))

    W: Set[Tuple[str, str]] = {pair for pair in all_causes if pair not in to_exclude}
    f_stats: Dict[Tuple[str, str], float] = {k: v for k, v in all_causes.items() if k in W}

    return W, excluded_bidirectional, f_stats


def _compute_level_partition(
    W: Set[Tuple[str, str]],
) -> Dict[str, int]:
    """
    Assign each node in W a level equal to the length of its longest path from
    any root in W.

    Partition: P_k = {nodes at level k}.
    - P_0 = nodes with no predecessors in W (in-degree 0).
    - P_k = nodes whose longest-path distance from any root is k.

    This stratification is used by Algorithm 1's outer loop.

    Returns
    -------
    dict mapping node → level index
    """
    nodes: Set[str] = set()
    for src, tgt in W:
        nodes.add(src)
        nodes.add(tgt)

    if not nodes:
        return {}

    # Build adjacency and in-degree
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    for src, tgt in W:
        adj[src].append(tgt)
        in_deg[tgt] += 1

    # Topological sort (Kahn's algorithm)
    queue: List[str] = [n for n in nodes if in_deg[n] == 0]
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
        # W contains a cycle (shouldn't happen after bidirectional removal, but
        # be defensive).  Fall back: assign level 0 to all.
        logger.warning(
            "W contains a cycle after bidirectional-pair removal.  "
            "Level partition falls back to level 0 for all nodes."
        )
        return {n: 0 for n in nodes}

    # Longest-path DP (forward pass over topological order)
    levels: Dict[str, int] = {n: 0 for n in nodes}
    for node in topo_order:
        for successor in adj[node]:
            if levels[successor] < levels[node] + 1:
                levels[successor] = levels[node] + 1

    return levels


def _can_reach_via_E(source: str, target: str, E: Set[Tuple[str, str]]) -> bool:
    """
    Check whether ``target`` is reachable from ``source`` through the edges
    already in E (the partial edge set being built by Algorithm 1).

    Used to decide whether to add edge (source, target): if target is already
    reachable from source via E, then (source, target) is a *transitive* edge
    and should NOT be added to E.
    """
    adj: Dict[str, List[str]] = {}
    for s, t in E:
        adj.setdefault(s, []).append(t)

    visited: Set[str] = set()
    stack: List[str] = [source]

    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, []))

    return False


def _pairwise_recovery_algorithm1(
    W: Set[Tuple[str, str]],
    f_stats: Dict[Tuple[str, str], float],
    grade: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    **Algorithm 1** from Kinnear & Mazumdar (2023) — Pairwise Granger
    Causality Recovery via transitive reduction.

    Given the ancestor set W (bidirectional pairs already excluded), recovers
    the *direct* edge set E by iterating over the level partition P_0, P_1, …
    and adding edge (i, j) only when j is not already reachable from i through
    previously added edges (i.e., no transitive shortcut exists).

    This eliminates false positive edges in W that arise from transitivity.
    For example, if 1→3 and 3→4 are direct edges, then 1→4 appears in W as an
    ancestor relation but is *not* a direct edge and is excluded from E
    (see Example 2.4 in the paper).

    Parameters
    ----------
    W : set of (source, target)
        Ancestor relations (output of ``_build_ancestor_set_W``).
    f_stats : dict mapping (source, target) → F-statistic
        Used to preserve weight information in output.
    grade : str
        Grade label for logging.

    Returns
    -------
    E_list : list of (source, target)
        Direct edges recovered by Algorithm 1.
    levels : dict mapping node → level
        Level partition for logging / diagnostics.
    """
    if not W:
        return [], {}

    levels = _compute_level_partition(W)
    if not levels:
        return [], {}

    max_level = max(levels.values())

    # Group nodes by level
    P: Dict[int, List[str]] = {}
    for node, lev in levels.items():
        P.setdefault(lev, []).append(node)

    W_set: Set[Tuple[str, str]] = set(W)
    E: Set[Tuple[str, str]] = set()

    # Outer loop: for each level k ≥ 1 (target nodes j ∈ P_k)
    for k in range(1, max_level + 1):
        if k not in P:
            continue
        targets_at_k = P[k]

        for j in targets_at_k:
            # Inner loop: r = 1, 2, …, k  (source levels k−r, from closest to farthest)
            for r in range(1, k + 1):
                source_level = k - r
                if source_level not in P:
                    continue
                for i in P[source_level]:
                    if (i, j) not in W_set:
                        continue
                    # Add (i, j) only if j is NOT already reachable from i via E.
                    # If it is reachable, (i, j) is a transitive consequence of
                    # already-discovered shorter paths and should be skipped.
                    if _can_reach_via_E(i, j, E):
                        logger.debug(
                            "Grade=%s: transitive edge (%s→%s) excluded from E.",
                            grade, i, j,
                        )
                        continue
                    E.add((i, j))

    E_list = list(E)
    logger.info(
        "Grade=%s: Algorithm 1 recovered %d direct edges from %d ancestor relations in W.",
        grade, len(E_list), len(W),
    )
    return E_list, levels


def _edges_to_dataframe(
    edges: List[Tuple[str, str]],
    f_stats: Dict[Tuple[str, str], float],
    grade: str,
) -> pd.DataFrame:
    """Convert edge list to DataFrame compatible with downstream functions."""
    rows = []
    for src, tgt in edges:
        f_stat = f_stats.get((src, tgt), 0.0)
        rows.append({
            "grade": grade,
            "source": src,
            "target": tgt,
            "f_statistic": f_stat,
            "p_value": None,  # p-value is available via pairwise_tests if needed
            "weight": f_stat,
        })
    if not rows:
        return pd.DataFrame(columns=["grade", "source", "target", "f_statistic", "p_value", "weight"])
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

    .. note:: **RiceEWS heuristic — not part of Kinnear & Mazumdar (2023).**
        The Pairwise Recovery Algorithm 1 in the paper assumes the causal graph
        is already a strongly causal DAG structurally.  It does not include any
        cycle-breaking step.  This function is an *additional* engineering
        safeguard applied when the raw pairwise test results still contain
        directed cycles after bidirectional-pair removal and transitive
        reduction.  It is retained for robustness but should be understood as
        outside the theoretical guarantees of the paper.

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
            # Cycle found but no edge available to remove — this is a logic error.
            raise RuntimeError(
                "Cycle-breaking: cycle detected but no edge found to remove. "
                "This indicates a bug in the cycle-detection or edge-lookup logic."
            )

    return df, removed_edges


def _is_polytree(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """
    Check whether the graph is a **strongly causal graph (polytree)** as defined in
    Kinnear & Mazumdar (2023), Definition 2.9:

        A DAG G is *strongly causal* if its underlying undirected skeleton (the
        same graph ignoring edge direction) is a **forest** — i.e., there is at
        most one undirected path between any pair of nodes.

    This is *not* equivalent to "every node has at most one parent".  A diamond
    / v-structure (e.g. i→j←k) is a valid strongly causal graph because its
    undirected skeleton i–j–k has no cycle, even though j has two parents.
    See Kinnear & Mazumdar (2023) Section 2.5 and Figure 1(a) for explicit
    examples of multi-parent nodes in valid strongly causal graphs.
    """
    if _detect_cycle(nodes, edges):
        return False
    if not edges:
        return True
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return nx.is_forest(G.to_undirected())


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
    Jalankan MODUL 3 dari output MODUL 2 menggunakan **Algorithm 1**
    (Pairwise Recovery) dari Kinnear & Mazumdar (2023).

    Proses (sesuai paper):
    1. Baca granger_results.json dari MODUL 2.
    2. **Bangun ancestor set W** dari hasil pairwise Granger.
       - Pasangan bidireksional (i→j DAN j→i) dibuang **keduanya** sebagai
         konfounding (Corollary 2.3 / Remark 2.5). [Fix #8]
    3. **Terapkan Algorithm 1** (transitive reduction via level partition).
       Ini mengeliminasi edge transitif palsu yang muncul di W (misal
       1→4 yang muncul karena jalur 1→3→4, bukan karena 1 langsung
       mempengaruhi 4).  [Fix #6]
    4. Jika (enforce_dag=True) dan output Algorithm 1 masih mengandung siklus
       (tidak seharusnya untuk SCG), terapkan cycle-breaking greedy sebagai
       safeguard tambahan. **Ini adalah RiceEWS heuristic, bukan bagian paper.**
       [Fix #7]
    5. Hitung metrik market leader dari graf E yang sudah diproses.
    6. Simpan semua output.

    Parameters
    ----------
    granger_json_path : str
        Path ke ``granger_results.json`` dari MODUL 2.
    output_dir : str
        Folder output MODUL 3, contoh: ``data/processed/module_03``.
    enforce_dag : bool, default True
        Jika True, terapkan cycle-breaking greedy sebagai safeguard jika
        Algorithm 1 menghasilkan siklus (seharusnya tidak terjadi untuk SCG
        yang valid, tapi berguna untuk data nyata yang tidak memenuhi asumsi
        sepenuhnya).  Edges yang dihapus dicatat untuk audit.

    Returns
    -------
    dict
        Hasil inference lengkap per grade, termasuk path output.

    Notes
    -----
    **Methodology assumptions (Kinnear & Mazumdar 2023, Section 2):**

    - The exact recovery guarantee (Theorem 2.2) requires the true causal
      graph to be a *strongly causal* DAG (Definition 2.9) — i.e., a DAG
      whose undirected skeleton is a forest.
    - Assumption 2.1 must hold: the noise covariance matrix Σ_v is diagonal
      (no instantaneous causality).
    - The Assumption 2.1 (no-cancellation) and Definition 2.11 (persistence)
      are also required for the exact guarantee.

    For real-world rice market data these assumptions may hold approximately
    rather than exactly.  Results should be interpreted as *heuristic*
    estimates of the causal structure, not as guaranteed exact recovery.
    """
    granger_path = Path(granger_json_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_granger_json(granger_path)

    # Grades are top-level keys in granger_results.json
    grades = list(data.keys())

    all_dag_edges: List[pd.DataFrame] = []
    leaders_per_grade: Dict[str, List[Dict]] = {}
    summary_rows: List[Dict] = []
    visualization_paths: Dict[str, str] = {}
    removed_edges_per_grade: Dict[str, List[Tuple[str, str]]] = {}
    excluded_bidirectional_per_grade: Dict[str, List[List[str]]] = {}

    for grade in grades:
        grade_data = data.get(grade, {})
        pairwise_tests = grade_data.get("pairwise_tests", {})

        # --- Step 1: Build ancestor set W, exclude bidirectional pairs [Fix #8] ---
        W, excluded_bidir, f_stats = _build_ancestor_set_W(pairwise_tests)

        if excluded_bidir:
            excluded_list = [sorted(list(p)) for p in excluded_bidir]
            excluded_bidirectional_per_grade[grade] = excluded_list
            logger.info(
                "Grade=%s: %d bidirectional pair(s) excluded (Corollary 2.3): %s",
                grade, len(excluded_bidir), excluded_list,
            )
            print(
                f"  [Algorithm 1] Grade={grade}: {len(excluded_bidir)} "
                f"bidirectional pair(s) excluded as confounders."
            )
        else:
            excluded_bidirectional_per_grade[grade] = []

        # --- Step 2: Algorithm 1 — transitive reduction [Fix #6] ---
        e_edges, levels = _pairwise_recovery_algorithm1(W, f_stats, grade)
        edge_df = _edges_to_dataframe(e_edges, f_stats, grade)

        # --- Step 3 (optional): cycle-breaking safeguard [Fix #7] ---
        # This is a RiceEWS heuristic outside the paper, applied only when
        # Algorithm 1 still produces cycles (shouldn't happen for true SCG).
        removed_edges_per_grade[grade] = []
        if enforce_dag and not edge_df.empty:
            edge_nodes = sorted(set(edge_df["source"].tolist() + edge_df["target"].tolist()))
            edge_pairs = list(zip(edge_df["source"], edge_df["target"]))
            if _detect_cycle(edge_nodes, edge_pairs):
                logger.warning(
                    "Grade=%s: Algorithm 1 output still contains cycles. "
                    "Applying greedy cycle-breaking (RiceEWS heuristic, not from paper).",
                    grade,
                )
                dag_edge_df, removed = _break_cycles_greedy(edge_df)
                removed_edges_per_grade[grade] = removed
                if removed:
                    print(
                        f"  [Cycle-breaking heuristic] Grade={grade}: {len(removed)} "
                        f"edge(s) removed to enforce DAG (outside paper methodology)."
                    )
                edge_df = dag_edge_df

        all_dag_edges.append(edge_df)

        nodes = sorted(
            set(edge_df["source"].tolist() + edge_df["target"].tolist())
        ) if not edge_df.empty else []
        edge_pairs_final = list(zip(edge_df["source"], edge_df["target"])) if not edge_df.empty else []

        is_dag = not _detect_cycle(nodes, edge_pairs_final) if nodes else True
        is_poly = _is_polytree(nodes, edge_pairs_final) if nodes else True

        metrics = _build_node_metrics(nodes, edge_df)
        top_leaders = metrics.head(5).to_dict(orient="records")
        leaders_per_grade[grade] = top_leaders

        summary_rows.append(
            {
                "grade": grade,
                "nodes": len(nodes),
                "edges": int(len(edge_df)),
                "W_size": len(W),
                "bidirectional_excluded": len(excluded_bidirectional_per_grade[grade]),
                "transitive_removed": max(0, len(W) - len(e_edges)),
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
        "algorithm": "Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery",
        "input": str(granger_path),
        "output_dir": str(out_dir),
        "grades": grades,
        "enforce_dag": enforce_dag,
        "summary": summary_rows,
        "leaders_per_grade": leaders_per_grade,
        "excluded_bidirectional_per_grade": excluded_bidirectional_per_grade,
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
        "**Algorithm**: Kinnear & Mazumdar (2023) Algorithm 1 — Pairwise Recovery",
        "",
        "> **Methodology Notes:**",
        "> - Edges are recovered via *transitive reduction* (Algorithm 1), not direct",
        ">   thresholding of pairwise test results.",
        "> - Bidirectional pairs are excluded as confounders (Corollary 2.3).",
        "> - `is_polytree` checks whether the DAG's undirected skeleton is a forest",
        ">   (correct Definition 2.9), not whether each node has ≤1 parent.",
        "> - The exact-recovery guarantee requires the Strongly Causal Graph (SCG)",
        ">   assumption.  Real-world results are heuristic estimates.",
        "",
        f"Input: `{granger_path}`",
        f"Output dir: `{out_dir}`",
        f"DAG enforcement: `{enforce_dag}`",
        "",
        "## Per Grade Summary",
        "",
        "| Grade | Nodes | E (edges) | W (ancestors) | Bidir excluded | Transitive removed | DAG cycles removed | DAG | Polytree | Top Leader | Leader Score |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---|---:|",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r['grade']} | {r['nodes']} | {r['edges']} "
            f"| {r['W_size']} "
            f"| {r['bidirectional_excluded']} "
            f"| {r['transitive_removed']} "
            f"| {r['edges_removed_for_dag']} "
            f"| {r['is_dag']} | {r['is_polytree']} | {r['top_leader']} "
            f"| {r['top_leader_score']:.4f} |"
        )

    # Excluded bidirectional pairs section
    if any(excluded_bidirectional_per_grade.values()):
        md_lines += [
            "",
            "## Excluded Bidirectional Pairs (Confounders, Corollary 2.3)",
            "",
            "| Grade | Market A | Market B |",
            "|---|---|---|",
        ]
        for grade, pairs in excluded_bidirectional_per_grade.items():
            for pair in pairs:
                if len(pair) >= 2:
                    md_lines.append(f"| {grade} | {pair[0]} | {pair[1]} |")

    # Removed edges audit section
    if any(removed_edges_per_grade.values()):
        md_lines += [
            "",
            "## Edges Removed by Cycle-Breaking Heuristic (Outside Paper)",
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
            f"  Grade={r['grade']}: W={r['W_size']}, E={r['edges']}, "
            f"bidir_excl={r['bidirectional_excluded']}, "
            f"transitive_removed={r['transitive_removed']}, "
            f"DAG={r['is_dag']}, Polytree={r['is_polytree']}, "
            f"TopLeader={r['top_leader']}"
        )

    return result
