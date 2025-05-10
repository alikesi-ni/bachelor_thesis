import networkx as nx
import numpy as np

from thesis.evaluation.qsc_evaluation import QscEvaluation
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_names = [
    # # small datasets
    # "KKI",
    # "PTC_FM",

    # # large datasets
    # "COLLAB",
    # "DD",

    # # medium datsets
    # "IMDB-BINARY",
    # "NCI1",
    # "REDDIT-BINARY",

    # # social network datasets
    # "EGO-1",
    # "EGO-2"
    # "EGO-3",
    # "EGO-4",

    # # new dataset(s)
    # "MSRC_9",
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    max_step = np.inf

    qsc_evaluation = QscEvaluation(dataset_name, disjoint_graph, graph_id_label_map)
    qsc_evaluation.refine_and_create_feature_vector_matrices(q=0)