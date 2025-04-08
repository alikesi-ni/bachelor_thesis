import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

from GWL_python.color_hierarchy.color_hierarchy_tree import ColorHierarchyTree


def update_colors(graph: nx.Graph, color_hierarchy_tree: ColorHierarchyTree) -> None:
    """
    Updates the color of the associated nodes for all leaves of the ColorHierarchyTree.

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph instance that is being refined

    color_hierarchy_tree : ColorHierarchyTree
        Corresponding instance of ColorHierarchyTree

    """

    for leaf in color_hierarchy_tree.get_leaves():
        for node in leaf.associated_vertices:
            graph.nodes[node]["color-stack"].extend([leaf.color])


def generate_neighbor_color_count(graph: nx.Graph, vertex: int, edge_labels: dict) -> dict:
    """
    Generates the neighbor color count for a given vertex.

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph instance that is being refined

    vertex : int
        ID of the node/ vertex for which the neighbor color count need to be generated

    edge_labels : dict
        A dictionary, where the key is the edge, and the corresponding value is its assigned label

    Returns
    -------
    out : dict
        A dictionary, where a key-value pair represent a node, and the count of node colors of its neighbors

    """

    color_neighbor_count = dict()

    for neighbor in nx.neighbors(graph, vertex):

        if len(edge_labels) != 0:
            edge = (vertex, neighbor) if (vertex, neighbor) in edge_labels.keys() else (neighbor, vertex)
            neighbor_color = (graph.nodes[neighbor]["color-stack"][-1], edge_labels[edge])

        else:
            neighbor_color = graph.nodes[neighbor]["color-stack"][-1]

        if neighbor_color in color_neighbor_count.keys():
            color_neighbor_count[neighbor_color] += 1
        else:
            color_neighbor_count[neighbor_color] = 1

    return color_neighbor_count


def draw_gwl_refined_graph(refined_graph: nx.Graph, pos: dict = None) -> None:
    """
    Visualizes the NetworkX Graph that has been refined using Gradual Weisfeiler-Leman color refinement.

    Parameters
    ----------
    refined_graph : nx.Graph
        A nx.Graph instance that has been refined

    pos : dict
        A dictionary of positions keyed by node (default: None)

    """

    refined_colors = {node: colors[-1] for node, colors in
                      nx.get_node_attributes(refined_graph, "color-stack").items()}

    unique_color_set = set(refined_colors.values())

    if len(unique_color_set) > 48:
        print("Warning: Currently only 48 distinct colors are available for coloring nodes in the "
              "refined graph. \nThis refined graph requires more than 48 distinct colors, "
              "consequently only a uniform color is used.")

        color_map = {node: "skyblue" for node, _ in refined_colors.items()}

    else:

        available_colors = list()
        available_colors.extend([matplotlib.colormaps["Paired"](i) for i in range(12)])
        available_colors.extend([matplotlib.colormaps["Dark2"](i) for i in range(8)])
        available_colors.extend([matplotlib.colormaps["tab20"](i) for i in range(20)])
        available_colors.extend([matplotlib.colormaps["Accent"](i) for i in range(8)])

        color_map = {node: available_colors[color] for node, color in refined_colors.items()}

    if pos is None:
        pos = nx.spring_layout(refined_graph)

    nx.draw(refined_graph, pos=pos, with_labels=True, node_color=list(color_map.values()))
    plt.show()
