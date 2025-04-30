import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.other_utils import analyze
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv


def analyze_nodes(graph: nx.Graph, dataset_name):
    nodes = list(graph.nodes)

    if not nodes:
        print(f"[{dataset_name}] Problem: Graph has no nodes!")
        return

    if nodes[0] != 0:
        print(f"[{dataset_name}] Problem: Node indices must start at 0 (starts at {nodes[0]})")
    elif nodes != list(range(len(nodes))):
        print(f"[{dataset_name}] Problem: Node indices must be consecutive integers starting from 0")
    else:
        print(f"[{dataset_name}] OK: Nodes are sorted and start at 0")


dataset_names = [
    "PTC_FM",
    "KKI",
    "EGO-1",
    "EGO-2",
    "EGO-3",
    "EGO-4",
    "DD",
    "IMDB-BINARY",
    "MSRC_9",
    "REDDIT-BINARY",
    "ENZYMES"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../data", dataset_name)
    # analyze(graphs)
    disjoint_graph = nx.disjoint_union_all(graphs)

    analyze_nodes(disjoint_graph, dataset_name)



