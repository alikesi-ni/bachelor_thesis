import itertools

import networkx as nx


def generate_feature_vector(refined_graph: nx.Graph) -> dict:
    """
    Generates the feature vector for a single refined graph.

    Parameters
    ----------
    refined_graph : nx.Graph
        A nx.Graph instance that has already been refined using GWL

    Returns
    -------
    out : dict
        A dictionary, where a key-value pair represent a color, and its count.

    """
    fv = dict()

    for color in list(itertools.chain(*nx.get_node_attributes(refined_graph, "color-stack").values())):

        if color in fv.keys():
            fv[color] += 1
        else:
            fv[color] = 1

    return fv


def generate_feature_vectors(refined_disjoint_graph: nx.Graph) -> dict:
    """
    Generates the feature vectors for all the graphs.

    Parameters
    ----------
    refined_disjoint_graph : nx.Graph
        A nx.Graph instance (which is disjoint union of all the graphs) that has already been refined using GWL

    Returns
    -------
    out : dict
        A dictionary, where a key is the graph id and value is a dictionary containing the corresponding
        feature vector.

    """

    fvs = dict()

    nodes = refined_disjoint_graph.nodes

    for node in nodes:

        gid = nodes[node]["gid"]
        colors = nodes[node]["color-stack"]

        if gid not in fvs.keys():
            fvs[gid] = dict()

        for color in colors:
            if color in fvs[gid].keys():
                fvs[gid][color] += 1
            else:
                fvs[gid][color] = 1

    return fvs


def has_distinct_edge_labels(graph: nx.Graph) -> bool:
    edge_labels = nx.get_edge_attributes(graph, "label")
    return len(edge_labels) != 0 and (len(set(edge_labels.values())) > 1)


def has_distinct_node_labels(graph: nx.Graph) -> bool:
    node_labels = nx.get_node_attributes(graph, "label")
    return (len(node_labels) != 0) and (len(set(node_labels.values())) > 1)

def remove_node_labels(graph: nx.Graph) -> None:
    for node in graph.nodes:
        graph.nodes[node].pop("label", None)