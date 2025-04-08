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

disjoint_graph = convert_to_disjoint_union_graph(dataset)

wl = WeisfeilerLeman(refinement_steps=13)
wl.refine_color(disjoint_graph)

# refine_color(disjoint_graph, 3)

print("Done)")
# wl_one_iteration(disjoint_graph)
