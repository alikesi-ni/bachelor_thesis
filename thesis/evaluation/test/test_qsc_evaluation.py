import networkx as nx
import numpy as np

from thesis.evaluation.qsc_evaluation import QscEvaluation
from thesis.evaluation.step_settings import StepSettings
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_names = [
    # # small datasets
    # "KKI",
    # "PTC_FM",
    "MSRC_9",

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
    "ENZYMES",
    # "PROTEINS",

    # "IMDB-BINARY",
    # "DD",
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../../data", dataset_name)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    step_settings = [
        # StepSettings(method="q_ratio", method_params={"q_ratio": 0.3, "allow_duplicate_steps": True}),
        # StepSettings(method="q_ratio", method_params={"q_ratio": 0.5, "allow_duplicate_steps": True}),
        # StepSettings(method="q_ratio", method_params={"q_ratio": 0.7, "allow_duplicate_steps": True}),
        StepSettings(method="h_grid", method_params={"h_grid": list(range(0, 17)) + [32, 64, 128, 256], "q_strictly_descending": True}),
        StepSettings(method="h_grid", method_params={"h_grid": list(range(0, 17)) + [32, 64, 128, 256], "q_strictly_descending": False}),
    ]

    best_accuracy = 0
    best_std = np.inf
    best_parameters = ""
    for step_setting in step_settings:
        qsc_evaluation = QscEvaluation(dataset_name, graph_id_label_map, step_setting,
                                       base_dir="../../evaluation-results")
        accuracy, std = qsc_evaluation.evaluate()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_std = std
            best_parameters = step_setting.to_dirname()

    print(f"BEST PARAMETERS: {best_parameters}")
    print(f"BEST ACCURACY: {best_accuracy:.2f} +- {best_std:.2f}")