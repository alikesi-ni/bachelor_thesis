import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv, evaluate_gwl_cv

dataset_names = [
    "PTC_FM"
    # "KKI",
    # "ENZYMES",
    # "MSRC_9",
    # "IMDB-BINARY",
    # "REDDIT-BINARY",
    # "EGO-2",
    # "EGO-3",
    # "DD",
    # "EGO-4"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)

    colored_graph = ColoredGraph(disjoint_graph)

    # wl = WeisfeilerLemanColoringGraph(colored_graph, refinement_steps=25)
    # wl.refine(verbose=True)

    # qsc = QuasiStableColoringGraph(colored_graph, q=1, n_colors=512)
    # qsc.refine()

    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    h_grid = list(range(1, 11))
    k_grid = [2, 4, 8, 16]
    q_grid = [2**i for i in range(3, -1, -1)] + [0]
    n_max = 512

    c_grid = [10**i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

    evaluate_quasistable_cv(
        disjoint_graph, graph_id_label_map, q_grid, n_max, c_grid, folds=10, dataset_name=dataset_name, repeats=1, start_repeat=1
    )

    # evaluate_wl_cv(
    #     disjoint_graph, graph_id_label_map, h_grid, c_grid, folds=10, dataset_name=dataset_name, repeats=5, start_repeat=1
    # )
    #
    # evaluate_gwl_cv(disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid, dataset_name=dataset_name, folds=10, repeats=5, start_repeat=1)