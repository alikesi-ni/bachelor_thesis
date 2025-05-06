import os

import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv, evaluate_gwl_cv, \
    get_stats_from_test_results_csv

dataset_names = [
    # "PTC_FM"
    # "KKI",
    # "ENZYMES",
    # "MSRC_9",
    # "IMDB-BINARY",
    # "REDDIT-BINARY",
    "EGO-1"
    # "EGO-2"
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

    # h_grid = list(range(1, 11))
    h_grid = [1000]
    k_grid = [2, 4, 8, 16]
    q_grid = [2**i for i in range(3, -1, -1)] + [0]
    n_max = 1024

    refinement_steps_grid = h_grid

    c_grid = [10**i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

    evaluate_quasistable_cv(
        disjoint_graph, graph_id_label_map, refinement_steps_grid=refinement_steps_grid, c_grid=c_grid, folds=10, dataset_name=dataset_name, repeats=1, start_repeat=1
    )

    # res_dir = "../tests/MSRC_9-Evaluation-QSC-20250506_111034"

    # get_stats_from_test_results_csv(os.path.join(res_dir, "test_results.csv"))

    # evaluate_wl_cv(
    #     disjoint_graph, graph_id_label_map, h_grid, c_grid, folds=10, dataset_name=dataset_name, repeats=1, start_repeat=1
    # )
    #
    # evaluate_gwl_cv(disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid, dataset_name=dataset_name, folds=10, repeats=5, start_repeat=1)