import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.other_utils import remove_node_labels
from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.other_utils import analyze
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv

dataset_names = [
"ENZYMES"
    # "MSRC_9_DIFF_SRC"
    #"IMDB-BINARY",
    # "REDDIT-BINARY",
    # "EGO-2",
    # "EGO-3",
    # "DD",
    # "EGO-4"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../data", dataset_name)
    analyze(graphs)
    disjoint_graph = nx.disjoint_union_all(graphs)

    # colored_graph = ColoredGraph(disjoint_graph)

    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    q_grid = [2**i for i in range(3, 0, -1)] + [1]
    n_max = 256
    c_grid = [10**i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

    evaluate_quasistable_cv(
        disjoint_graph, graph_id_label_map, q_grid, n_max, c_grid, folds=10, dataset_name=dataset_name, repeats=1
    )