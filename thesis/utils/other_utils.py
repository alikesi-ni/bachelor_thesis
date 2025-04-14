import networkx as nx


def has_distinct_edge_labels(graph: nx.Graph) -> bool:
    edge_labels = nx.get_edge_attributes(graph, "label")
    return len(edge_labels) != 0 and (len(set(edge_labels.values())) > 1)


def has_distinct_node_labels(graph: nx.Graph) -> bool:
    node_labels = nx.get_node_attributes(graph, "label")
    return (len(node_labels) != 0) and (len(set(node_labels.values())) > 1)

def remove_node_labels(graph: nx.Graph) -> None:
    for node in graph.nodes:
        graph.nodes[node].pop("label", None)