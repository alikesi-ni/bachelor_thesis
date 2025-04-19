import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_quasistable_cv, evaluate_wl_cv

dataset_names = [
    #"IMDB-BINARY",
    "REDDIT-BINARY",
    "EGO-2",
    "EGO-3",
    "DD",
    "EGO-4"
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

    q_grid = [2**i for i in range(6, 2, -1)]
    n_grid = [np.inf]
    # q_grid = [4, 3, 2, 1]
    # n_grid = [2**i for i in range(8, 10)]

    c_grid = [10**i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

    evaluate_wl_cv(
        disjoint_graph, graph_id_label_map, h_grid, c_grid, folds=10, dataset_name=dataset_name, repeats=10
    )

    # evaluate_quasistable_cv(
    #     disjoint_graph, graph_id_label_map, q_grid, n_grid, c_grid, folds=10, dataset_name=dataset_name, repeats=10
    # )