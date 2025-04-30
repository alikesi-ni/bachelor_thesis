import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.other_utils import analyze
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv, evaluate_gwl_cv

dataset_names = [
    # "PTC_FM",
    # "KKI",
    # "EGO-1",
    # "IMDB-BINARY",
    # "MSRC_9"
    "ENZYMES"
    # "NCI1"
    #"REDDIT-BINARY",
    #"EGO-2",
    #"EGO-3",
    #"DD",
    #"EGO-4"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../data", dataset_name)

    analyze(graphs)

    disjoint_graph = nx.disjoint_union_all(graphs)

    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    h_grid = list(range(1, 11))  # h ∈ {1, ..., 10}
    k_grid = [2, 4, 8, 16]  # k ∈ {2, 4, 8, 16}
    c_grid = [10 ** i for i in range(-3, 4)]  # C ∈ {1e-3, ..., 1e3}

    evaluate_gwl_cv(
        disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid,
        dataset_name=dataset_name, folds=10, repeats=1
    )