import networkx as nx
import numpy as np

from thesis.evaluation.evaluation_parameters import EvaluationParameters
from thesis.evaluation.qsc_evaluation import QscEvaluation
from thesis.evaluation.utils import generate_report
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_names = [
    # # small datasets
    # "KKI",
    # "PTC_FM",
    # "MSRC_9",

    # # large datasets
    # "COLLAB",
    # "DD",
    "REDDIT-BINARY",

    # # medium datsets
    "IMDB-BINARY",
    # "NCI1",

    # # social network datasets
    # "EGO-1",
    # "EGO-2",
    "EGO-3",
    # "EGO-4",

    # # new datasets
    "ENZYMES",
    # "PROTEINS"
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    evaluation_parameters_list = [
        # EvaluationParameters(method="h_grid", h_grid=list(range(0, 33)), q_strictly_descending=False, include_inbetween_steps=True),
        # EvaluationParameters(method="h_grid", h_grid=list(range(0, 33)), q_strictly_descending=True, include_inbetween_steps=True),
        EvaluationParameters(method="q_half", q_strictly_descending=True, include_inbetween_steps=False),
        EvaluationParameters(method="h_grid", h_grid=list(range(0, 17)) + [32, 64, 128, 256], q_strictly_descending=True, include_inbetween_steps=True),
        EvaluationParameters(method="h_grid", h_grid=list(range(0, 17)) + [32, 64, 128, 256], q_strictly_descending=False, include_inbetween_steps=True),
    ]

    best_accuracy = 0
    best_std = np.inf
    best_parameters = ""
    for parameters in evaluation_parameters_list:
        qsc_evaluation = QscEvaluation(dataset_name, disjoint_graph, graph_id_label_map, parameters,
                                       base_dir="../../evaluation-results")
        accuracy, std = qsc_evaluation.evaluate()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_std = std
            best_parameters = parameters.to_dirname()

    print(f"BEST PARAMETERS: {best_parameters}")
    print(f"BEST ACCURACY: {best_accuracy:.2f} +- {best_std:.2f}")
    # eval_output_dir = "../../evaluation-results/QSC-EGO-2/h_grid__0-1-2-3-4-5-6-7-8-9-10__with_inbetween__without_desc"
    # generate_report(eval_output_dir, eval_output_dir)