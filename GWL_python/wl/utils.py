import networkx as nx

import matplotlib
import matplotlib.pyplot as plt


def draw_wl_refined_graph(refined_graph: nx.Graph, pos: dict = None) -> None:
    """
    Visualizes the NetworkX Graph that has been refined using Weisfeiler-Leman color refinement.

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

        unique_colors = {color: available_colors[idx] for idx, color in enumerate(sorted(unique_color_set))}

        color_map = {node: unique_colors[color] for node, color in refined_colors.items()}

    if pos is None:
        pos = nx.spring_layout(refined_graph)

    nx.draw(refined_graph, pos=pos, with_labels=True, node_color=list(color_map.values()))
    plt.show()
