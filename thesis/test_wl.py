import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from thesis.weisfeiler_leman import refine_color
from utils import convert_to_disjoint_union_graph

from GWL_python.wl.wl import WeisfeilerLeman


def count_unique_colors(graph):
    """Count the number of unique colors in the graph."""
    color_stack = nx.get_node_attributes(graph, "color-stack")
    unique_colors = set()

    for colors in color_stack.values():
        unique_colors.update(colors)

    return len(unique_colors)


def wl_one_iteration(graph):
    """Use the WeisfeilerLeman class to perform one iteration and count colors before and after."""
    # Instantiate the WL class with 1 iteration
    wl = WeisfeilerLeman(refinement_steps=1)

    # Count unique colors before refinement
    initial_colors = count_unique_colors(graph)
    print(f"Number of unique colors before WL iteration: {initial_colors}")

    # Perform one iteration of Weisfeiler-Lehman refinement
    wl.refine_color(graph)

    # Count unique colors after one iteration
    final_colors = count_unique_colors(graph)
    print(f"Number of unique colors after one WL iteration: {final_colors}")

dataset_names = [
    "KKI", "PTC_FM", "COLLAB", "DD", "IMDB-BINARY", "MSRC_9",
    "NCI1", "REDDIT-BINARY", "MUTAG", "ENZYMES", "PROTEINS"
]

# dataset_name = 'MUTAG'
dataset_name = 'PTC_FM'
# dataset_name = 'KKI'
# dataset_name = 'IMDB-BINARY'
# dataset_name = 'MSRC_9'
# dataset_name = 'NCI1'

root = './data/'
dataset = TUDataset(root, dataset_name)

print(f"Loaded {len(dataset)} graphs from {dataset_name}.")

# def convert_to_disjoint_union_graph(dataset):
#     """
#     Convert all graphs in the dataset to a single disjoint union graph while preserving node attributes.
#     """
#     networkx_graphs = []
#
#     for data in dataset:
#         # Convert the graph to NetworkX format with the node attributes from 'x'
#         G = to_networkx(data, node_attrs=["x"], to_undirected=True)
#
#         # Undo the one-hot encoding and set the 'label' attribute
#         for node, attrs in G.nodes(data=True):
#             if "x" in attrs:
#                 # Convert one-hot encoded feature list to an integer label
#                 try:
#                     label = attrs["x"].index(1.0) # Get the index of the max value (1.0 in one-hot)
#                 except Exception:
#                     label = -1  # In case of unexpected data format
#                 G.nodes[node]["label"] = label
#                 # Remove the original 'x' attribute since it is now redundant
#                 del G.nodes[node]["x"]
#             else:
#                 # Default label if no attribute present
#                 G.nodes[node]["label"] = -1
#
#         # Add the graph to the list
#         networkx_graphs.append(G)
#
#     # Create a single disjoint union graph from the list of NetworkX graphs
#     disjoint_union_graph = nx.disjoint_union_all(networkx_graphs)
#     return disjoint_union_graph

# disjoint_graph = convert_to_disjoint_union_graph(dataset[:20])


disjoint_graph = convert_to_disjoint_union_graph(dataset)

# wl = WeisfeilerLeman(refinement_steps=2)
# wl.refine_color(disjoint_graph)

refine_color(disjoint_graph, 20)

print("Done)")
# wl_one_iteration(disjoint_graph)
