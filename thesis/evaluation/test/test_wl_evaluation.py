from thesis.evaluation.wl_evaluation import WlEvaluation
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
    # "EGO-2",
    # "EGO-3",
    # "EGO-4",

    # # new datasets
    # "ENZYMES",
    # "PROTEINS",

    # "IMDB-BINARY",
    # "DD",
]

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../../data", dataset_name)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    h_grid = list(range(0, 11))

    wl_evaluation = WlEvaluation(dataset_name, graph_id_label_map, h_grid,
                                   base_dir="../../evaluation-results")
    wl_evaluation.run()