"""
Polytree/Strong Causality Validator Module
==========================================

Menguji apakah jaringan pasar beras memenuhi strong causality (polytree) assumption
dari paper Kinnear & Mazumdar (2023).

Reference:
- Exact Recovery of Granger Causality Graphs with Unconditional Pairwise Tests
- Network Science, Vol 11, Issue 3, pp. 431-457, 2023

Polytree adalah DAG (Directed Acyclic Graph) di mana:
1. Setiap node memiliki paling banyak satu parent
2. Tidak ada cycles
3. Structure ini memungkinkan unconditional pairwise tests untuk exact recovery
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, List, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolytreeTestResult:
    """Hasil pengujian polytree assumption"""
    is_polytree: bool
    is_dag: bool
    max_parents: int
    nodes_with_multiple_parents: List[str]
    cycles_found: List[List[str]]
    cycle_count: int
    strong_causality_score: float  # 0-1, 1 = perfect polytree
    recommendations: List[str]


class PolytreeValidator:
    """
    Validator untuk mengecek apakah causal graph memenuhi polytree assumption.
    
    Polytree properties:
    - DAG (no cycles)
    - Each node has at most 1 parent
    - Allows unconditional pairwise testing for exact recovery
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def validate_polytree(self, 
                         causal_graph: nx.DiGraph,
                         market_names: Dict[int, str] = None) -> PolytreeTestResult:
        """
        Validate apakah causal graph memenuhi polytree assumption.
        
        Parameters:
        -----------
        causal_graph : nx.DiGraph
            Directed graph of causal relationships
            Nodes: market indices
            Edges: causal relationships (from -> to)
        market_names : Dict[int, str], optional
            Mapping dari node index ke market name
            
        Returns:
        --------
        PolytreeTestResult
            Hasil validasi polytree
            
        Example:
        --------
        >>> G = nx.DiGraph()
        >>> G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        >>> validator = PolytreeValidator()
        >>> result = validator.validate_polytree(G)
        >>> print(f"Is Polytree: {result.is_polytree}")
        >>> print(f"Strong Causality Score: {result.strong_causality_score:.3f}")
        """
        
        if not isinstance(causal_graph, nx.DiGraph):
            raise ValueError("Input harus nx.DiGraph (directed graph)")
        
        # 1. Check if DAG (Directed Acyclic Graph)
        is_dag = nx.is_directed_acyclic_graph(causal_graph)
        cycles = list(nx.simple_cycles(causal_graph)) if not is_dag else []
        
        # 2. Check max in-degree (parents per node)
        in_degrees = dict(causal_graph.in_degree())
        max_parents = max(in_degrees.values()) if in_degrees else 0
        nodes_with_multiple_parents = [node for node, degree in in_degrees.items() 
                                       if degree > 1]
        
        # 3. Determine if it's a polytree
        is_polytree = is_dag and max_parents <= 1
        
        # 4. Calculate strong causality score
        strong_causality_score = self._calculate_strong_causality_score(
            is_dag, max_parents, len(causal_graph.nodes())
        )
        
        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            is_polytree, is_dag, max_parents, cycles, 
            len(nodes_with_multiple_parents), len(causal_graph.nodes())
        )
        
        # Format output dengan market names jika tersedia
        formatted_cycles = cycles
        if market_names:
            formatted_cycles = [[market_names.get(n, str(n)) for n in cycle] 
                               for cycle in cycles]
            nodes_with_multiple_parents = [market_names.get(n, str(n)) 
                                          for n in nodes_with_multiple_parents]
        
        result = PolytreeTestResult(
            is_polytree=is_polytree,
            is_dag=is_dag,
            max_parents=max_parents,
            nodes_with_multiple_parents=nodes_with_multiple_parents,
            cycles_found=formatted_cycles,
            cycle_count=len(cycles),
            strong_causality_score=strong_causality_score,
            recommendations=recommendations
        )
        
        if self.verbose:
            self._print_results(result)
        
        return result

    def validate_causal_matrix(self, 
                              causal_matrix: np.ndarray,
                              market_names: List[str] = None,
                              threshold: float = 0.1) -> PolytreeTestResult:
        """
        Validate polytree dari causal strength matrix.
        
        Parameters:
        -----------
        causal_matrix : np.ndarray, shape (n_markets, n_markets)
            Causal strength matrix
            causal_matrix[i,j] = strength dari market i -> market j
        market_names : List[str], optional
            Names dari setiap market (index)
        threshold : float
            Threshold untuk consider edge ada (default: 0.1)
            
        Returns:
        --------
        PolytreeTestResult
            Hasil validasi
        """
        
        # Convert matrix ke graph dengan threshold
        graph = self._matrix_to_graph(causal_matrix, threshold)
        
        # Create name mapping
        name_dict = None
        if market_names is not None:
            name_dict = {i: name for i, name in enumerate(market_names)}
        
        return self.validate_polytree(graph, name_dict)

    def _matrix_to_graph(self, matrix: np.ndarray, threshold: float) -> nx.DiGraph:
        """Convert causal matrix to directed graph"""
        G = nx.DiGraph()
        n = matrix.shape[0]
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(n):
                if i != j and abs(matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=matrix[i, j])
        
        return G

    def _calculate_strong_causality_score(self, 
                                         is_dag: bool,
                                         max_parents: int,
                                         num_nodes: int) -> float:
        """
        Calculate strong causality score (0-1).
        
        Score components:
        - DAG: 50%
        - Polytree (max_parents=1): 50%
        """
        
        if num_nodes == 0:
            return 1.0
        
        score = 0.0
        
        # DAG contribution (50%)
        if is_dag:
            score += 0.5
        
        # Polytree contribution (50%)
        # Perfect: max_parents = 1
        # Degraded: max_parents > 1
        if max_parents <= 1:
            score += 0.5
        else:
            # Penalty for multiple parents
            # max_parents=2 -> 0.25 (50% * 0.5)
            # max_parents=3 -> 0.167 (50% * 0.33)
            penalty_factor = 1.0 / max_parents
            score += 0.5 * penalty_factor
        
        return min(1.0, score)

    def _generate_recommendations(self,
                                 is_polytree: bool,
                                 is_dag: bool,
                                 max_parents: int,
                                 cycles: List,
                                 num_multiple_parent_nodes: int,
                                 num_nodes: int) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if is_polytree:
            recommendations.append(
                "✓ Graph memenuhi polytree assumption. "
                "Dapat menggunakan unconditional pairwise Granger tests untuk exact recovery."
            )
            return recommendations
        
        if not is_dag:
            recommendations.append(
                f"⚠ CRITICAL: Graph mengandung {len(cycles)} cycle(s). "
                "Polytree assumption violated. Perlu menghilangkan cycles:"
            )
            for i, cycle in enumerate(cycles, 1):
                recommendations.append(f"  Cycle {i}: {' → '.join(map(str, cycle))} → {cycle[0]}")
            recommendations.append(
                "  Saran: Review data atau gunakan conditional Granger causality tests."
            )
        
        if max_parents > 1:
            percentage = (num_multiple_parent_nodes / num_nodes * 100) if num_nodes > 0 else 0
            recommendations.append(
                f"⚠ WARNING: {num_multiple_parent_nodes}/{num_nodes} nodes ({percentage:.1f}%) "
                f"memiliki multiple parents (max={max_parents}). "
                f"Polytree assumption partially violated."
            )
            recommendations.append(
                "  Saran: Pertimbangkan pengaruh indirect causality. "
                "Gunakan conditional tests jika perlu lebih strict."
            )
        
        if not is_dag and max_parents > 1:
            recommendations.append(
                "\n📊 OVERALL ASSESSMENT: Graph violates polytree assumption severely. "
                "Rekomendasi:"
            )
            recommendations.append(
                "  1. Review data quality dan Granger test parameters")
            recommendations.append(
                "  2. Gunakan conditional Granger causality (Kinnear & Mazumdar 2023)")
            recommendations.append(
                "  3. Pertimbangkan preprocessing atau filtering pada causal relationships")
        
        return recommendations

    def _print_results(self, result: PolytreeTestResult) -> None:
        """Pretty print hasil validasi"""
        
        print("\n" + "="*70)
        print("POLYTREE/STRONG CAUSALITY VALIDATION RESULT")
        print("="*70)
        
        status_icon = "✓" if result.is_polytree else "✗"
        print(f"\n{status_icon} Is Polytree: {result.is_polytree}")
        print(f"✓ Is DAG: {result.is_dag}" if result.is_dag else f"✗ Is DAG: {result.is_dag}")
        print(f"\n📊 Max Parents per Node: {result.max_parents}")
        print(f"🔄 Cycles Found: {result.cycle_count}")
        print(f"📈 Strong Causality Score: {result.strong_causality_score:.3f}/1.000")
        
        if result.nodes_with_multiple_parents:
            print(f"\n⚠ Nodes with Multiple Parents ({len(result.nodes_with_multiple_parents)}):")
            for node in result.nodes_with_multiple_parents[:10]:  # Show first 10
                print(f"  - {node}")
            if len(result.nodes_with_multiple_parents) > 10:
                print(f"  ... and {len(result.nodes_with_multiple_parents) - 10} more")
        
        if result.cycles_found:
            print(f"\n🔁 Cycles Detected ({len(result.cycles_found)}):")
            for i, cycle in enumerate(result.cycles_found[:5], 1):  # Show first 5
                print(f"  Cycle {i}: {' → '.join(map(str, cycle))} → {cycle[0]}")
            if len(result.cycles_found) > 5:
                print(f"  ... and {len(result.cycles_found) - 5} more cycles")
        
        print("\n📋 RECOMMENDATIONS:")
        for rec in result.recommendations:
            print(f"  {rec}")
        
        print("\n" + "="*70)

    def analyze_graph_structure(self, 
                               causal_graph: nx.DiGraph) -> Dict:
        """
        Analyze struktur graph secara detail.
        
        Returns:
        --------
        Dict with keys:
            - num_nodes
            - num_edges
            - density
            - avg_in_degree
            - avg_out_degree
            - topological_sort (if DAG)
            - strongly_connected_components
        """
        
        analysis = {
            "num_nodes": causal_graph.number_of_nodes(),
            "num_edges": causal_graph.number_of_edges(),
            "density": nx.density(causal_graph),
            "avg_in_degree": np.mean([d for _, d in causal_graph.in_degree()]) 
                           if causal_graph.number_of_nodes() > 0 else 0,
            "avg_out_degree": np.mean([d for _, d in causal_graph.out_degree()]) 
                            if causal_graph.number_of_nodes() > 0 else 0,
        }
        
        # Topological sort jika DAG
        if nx.is_directed_acyclic_graph(causal_graph):
            analysis["topological_sort"] = list(nx.topological_sort(causal_graph))
        else:
            analysis["topological_sort"] = None
        
        # Strongly connected components
        analysis["num_scc"] = nx.number_strongly_connected_components(causal_graph)
        
        return analysis


class PolytreeOptimizer:
    """
    Optimizer untuk transform non-polytree graph menjadi polytree
    dengan minimal edge removals atau modifications.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def break_cycles(self, 
                    causal_graph: nx.DiGraph,
                    strategy: str = "min_weight_edges") -> Tuple[nx.DiGraph, List[Tuple]]:
        """
        Remove edges untuk break cycles.
        
        Parameters:
        -----------
        causal_graph : nx.DiGraph
            Original causal graph
        strategy : str
            - "min_weight_edges": Remove lowest weight edges
            - "min_count": Remove minimum edges
            - "feedback_arc_set": Use minimum feedback arc set
            
        Returns:
        --------
        Tuple[nx.DiGraph, List]
            (Modified DAG, removed edges)
        """
        
        G = causal_graph.copy()
        cycles = list(nx.simple_cycles(G))
        removed_edges = []
        
        if not cycles:
            if self.verbose:
                print("✓ No cycles found. Graph is already a DAG.")
            return G, removed_edges
        
        if strategy == "min_weight_edges":
            return self._break_cycles_min_weight(G, cycles)
        elif strategy == "min_count":
            return self._break_cycles_min_count(G, cycles)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def reduce_multiple_parents(self,
                               causal_graph: nx.DiGraph,
                               strategy: str = "max_weight") -> Tuple[nx.DiGraph, List[Tuple]]:
        """
        Reduce multiple parents per node ke single parent (polytree).
        
        Parameters:
        -----------
        causal_graph : nx.DiGraph
        strategy : str
            - "max_weight": Keep edge with max weight
            - "min_weight": Keep edge with min weight
            
        Returns:
        --------
        Tuple[nx.DiGraph, List]
            (Modified polytree, removed edges)
        """
        
        G = causal_graph.copy()
        removed_edges = []
        
        for node in G.nodes():
            in_edges = list(G.in_edges(node, data=True))
            
            if len(in_edges) > 1:
                if strategy == "max_weight":
                    # Keep edge with max weight
                    max_edge = max(in_edges, key=lambda e: e[2].get('weight', 1))
                    edges_to_remove = [e for e in in_edges if e != max_edge]
                else:
                    # Keep edge with min weight
                    min_edge = min(in_edges, key=lambda e: e[2].get('weight', 1))
                    edges_to_remove = [e for e in in_edges if e != min_edge]
                
                for edge in edges_to_remove:
                    G.remove_edge(edge[0], edge[1])
                    removed_edges.append((edge[0], edge[1]))
        
        if self.verbose:
            print(f"✓ Reduced to polytree: {len(removed_edges)} edges removed")
        
        return G, removed_edges
    
    def _break_cycles_min_weight(self, G: nx.DiGraph, cycles: List) -> Tuple[nx.DiGraph, List]:
        """Break cycles by removing lowest weight edges"""
        removed_edges = []
        
        for cycle in cycles:
            # Find edge dengan minimum weight dalam cycle
            cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
            min_edge = min(cycle_edges, 
                         key=lambda e: G[e[0]][e[1]].get('weight', 1))
            
            if G.has_edge(*min_edge):
                G.remove_edge(*min_edge)
                removed_edges.append(min_edge)
        
        return G, removed_edges
    
    def _break_cycles_min_count(self, G: nx.DiGraph, cycles: List) -> Tuple[nx.DiGraph, List]:
        """Break cycles by removing minimum number of edges"""
        removed_edges = []
        
        # Simple greedy: remove one edge per cycle (lowest weight)
        for cycle in cycles:
            cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
            min_edge = min(cycle_edges,
                         key=lambda e: G[e[0]][e[1]].get('weight', 1))
            
            if G.has_edge(*min_edge):
                G.remove_edge(*min_edge)
                removed_edges.append(min_edge)
        
        return G, removed_edges


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_causal_graph(num_nodes: int = 10, 
                              edge_probability: float = 0.2,
                              force_polytree: bool = False) -> nx.DiGraph:
    """
    Create sample causal graph for testing.
    
    Parameters:
    -----------
    num_nodes : int
    edge_probability : float
        Probability of edge between any pair
    force_polytree : bool
        If True, generate graph that satisfies polytree assumption
        
    Returns:
    --------
    nx.DiGraph
    """
    
    if force_polytree:
        # Create DAG dengan max 1 parent per node
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Topological ordering
        nodes = list(range(num_nodes))
        
        # Add edges with max 1 parent per node
        for i in range(1, num_nodes):
            if np.random.random() < 0.7:  # 70% chance to have a parent
                parent = np.random.randint(0, i)
                weight = np.random.uniform(0.5, 2.0)
                G.add_edge(parent, i, weight=weight)
    else:
        # Random DAG (could have multiple parents)
        G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
        
        # Add weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 2.0)
    
    return G


if __name__ == "__main__":
    # Example usage
    print("Testing Polytree Validator...\n")
    
    # Test 1: Perfect polytree
    print("TEST 1: Perfect Polytree")
    print("-" * 50)
    polytree = create_sample_causal_graph(num_nodes=8, force_polytree=True)
    validator = PolytreeValidator(verbose=True)
    result1 = validator.validate_polytree(polytree)
    
    # Test 2: Non-polytree (random graph)
    print("\n\nTEST 2: Random Graph (likely non-polytree)")
    print("-" * 50)
    random_graph = create_sample_causal_graph(num_nodes=8, force_polytree=False, 
                                             edge_probability=0.3)
    result2 = validator.validate_polytree(random_graph)
    
    # Test 3: Graph analysis
    print("\n\nTEST 3: Detailed Graph Structure Analysis")
    print("-" * 50)
    analysis = validator.analyze_graph_structure(random_graph)
    print(f"Graph Analysis: {analysis}")
