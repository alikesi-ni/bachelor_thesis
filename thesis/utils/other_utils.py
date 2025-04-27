from typing import Dict

import networkx as nx

from scipy.sparse import csr_array, vstack


def has_distinct_edge_labels(graph: nx.Graph) -> bool:
    edge_to_label_map = nx.get_edge_attributes(graph, "label")
    return len(edge_to_label_map) != 0 and (len(set(edge_to_label_map.values())) > 1)


def has_distinct_node_labels(graph: nx.Graph) -> bool:
    node_to_label_map = nx.get_node_attributes(graph, "label")
    return (len(node_to_label_map) != 0) and (len(set(node_to_label_map.values())) > 1)

def remove_node_labels(graph: nx.Graph) -> None:
    for node in graph.nodes:
        graph.nodes[node].pop("label", None)

def convert_to_feature_matrix(gid_to_feature_vector_map: Dict[int, Dict[int, int]]) -> csr_array:
    """
    Converts a dictionary of graph-level feature vectors into a sparse matrix.

    Each row corresponds to a graph (identified by gid), and each column
    corresponds to a color ID. The values represent the frequency of that
    color in the corresponding graph.

    Color IDs are used directly as column indices. GIDs are assumed to be
    non-negative integers and used as row indices.

    Assumptions
    -----------
    - Color IDs are consecutive, non-negative integers (e.g., from WL refinement)
    - GIDs are unique non-negative integers (e.g., from node metadata)
    - The matrix is built up to max_gid and max_color_id inclusively

    Parameters
    ----------
    gid_to_feature_vector_map : dict[int, dict[int, int]]
        A mapping from graph ID (gid) to its color histogram feature vector,
        where each inner dictionary maps color IDs to their counts.

    Returns
    -------
    csr_array
        A sparse matrix of shape (max_gid + 1, max_color_id + 1), where
        row i corresponds to gid = i, and column j corresponds to color ID = j.
    """

    all_gids = sorted(gid_to_feature_vector_map)
    max_gid = max(all_gids)

    max_color_id = max(color for vec in gid_to_feature_vector_map.values() for color in vec)

    row_count = max_gid + 1 # sparse matrix starts with row 0
    col_count = max_color_id + 1 # sparse matrix starts with col 0

    rows = []

    for gid in range(0, row_count):
        vec = gid_to_feature_vector_map.get(gid, {})
        indices = list(vec.keys())  # color IDs as column indices
        values = list(vec.values())
        row = csr_array((values, ([0] * len(indices), indices)), shape=(1, col_count))
        rows.append(row)

    return vstack(rows, format="csr")

def analyze(graphs):
    """
    Analyze a list of NetworkX graphs by printing basic statistics.
    """
    num_graphs = len(graphs)
    num_nodes = sum(len(g.nodes) for g in graphs)
    num_edges = sum(len(g.edges) for g in graphs)

    avg_nodes = num_nodes / num_graphs if num_graphs > 0 else 0
    avg_edges = num_edges / num_graphs if num_graphs > 0 else 0

    # Aggregate node, edge, and graph labels
    node_labels = set()
    edge_labels = set()
    graph_labels = set()

    for g in graphs:
        graph_labels.add(g.graph.get("graph_label"))

        for _, attrs in g.nodes(data=True):
            if "label" in attrs:
                node_labels.add(attrs["label"])

        for _, _, attrs in g.edges(data=True):
            if "label" in attrs:
                edge_labels.add(attrs["label"])

    print(f"Number of graphs: {num_graphs}")
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {num_edges}")
    print(f"Average nodes per graph: {avg_nodes:.2f}")
    print(f"Average edges per graph: {avg_edges:.2f}")
    print(f"Number of unique graph labels: {len(graph_labels)}")
    print(f"Number of unique node labels: {len(node_labels)}")
    print(f"Number of unique edge labels: {len(edge_labels)}")
    print("-" * 50)
