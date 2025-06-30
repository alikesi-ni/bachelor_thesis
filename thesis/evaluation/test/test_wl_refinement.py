import networkx as nx
import numpy as np

from thesis.evaluation.qsc_refinement import QscRefinement
from thesis.evaluation.wl_refinement import WlRefinement
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_names = [
    # # small datasets
    # "KKI",
    "PTC_FM",
    # "MSRC_9",
    # "MUTAG",

    # # large datasets
    # "COLLAB",
    # "DD",
    # "REDDIT-BINARY",

    # # medium datsets
    # "IMDB-BINARY",
    # "NCI1",

    # # social network datasets
    # "EGO-1",
    # "EGO-2",
    # "EGO-3",
    # "EGO-4",

    # # new datasets
    # "ENZYMES",
    # "PROTEINS"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    max_step = np.inf

    wl_refinement = WlRefinement(dataset_name, disjoint_graph, graph_id_label_map, refinement_steps=39, base_dir="../../evaluation-results")
    wl_refinement.run()