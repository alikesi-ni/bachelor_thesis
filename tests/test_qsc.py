import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.other_utils import remove_node_labels
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_wl_cv, evaluate_quasistable_cv

dataset_name = "PTC_FM"

graphs = dataset_to_graphs("../data", dataset_name)
disjoint_graph = nx.disjoint_union_all(graphs)
# remove_node_labels(disjoint_graph)
#
# colored_graph = ColoredGraph(disjoint_graph)
# qsc = QuasiStableColoringGraph(colored_graph, q=32)
# qsc.refine()

graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

q_grid = list(range(1, 6))  # q ∈ {1, 2, 3, 4, 5}
n_grid = [2**i for i in range(9, 12)] + [np.inf]  # n ∈ {8, 16, 32, 64, 128, np.inf}
c_grid = [10**i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

mean_acc, std_acc = evaluate_quasistable_cv(
    disjoint_graph, graph_id_label_map, q_grid, n_grid, c_grid, folds=4, dataset_name="PTC_FM"
)