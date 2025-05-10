import networkx as nx
import numpy as np

from thesis.evaluation.evaluation_parameters import EvaluationParameters
from thesis.evaluation.qsc_evaluation import QscEvaluation
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_names = [
    # # small datasets
    # "KKI",
    # "PTC_FM",
    # "MSRC_9",

    # # large datasets
    # "COLLAB",
    # "DD",
    # "REDDIT-BINARY",

    # # medium datsets
    # "IMDB-BINARY",
    # "NCI1",

    # # social network datasets
    # "EGO-1",
    "EGO-2",
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

    evaluation_parameters = EvaluationParameters(method="h_grid", h_grid=list(range(0,11)), q_strictly_descending=False, include_inbetween_steps=True)

    qsc_evaluation = QscEvaluation(dataset_name, disjoint_graph, graph_id_label_map, evaluation_parameters,
                                   base_dir="../../evaluation-results")
    print(evaluation_parameters.to_dirname())