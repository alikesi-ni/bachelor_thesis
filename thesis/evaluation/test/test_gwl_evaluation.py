from thesis.evaluation.gwl_evaluation import GwlEvaluation
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

    # social network datasets
    # "EGO-1",
    # "EGO-2",
    # "EGO-3",
    # "EGO-4",

    # new datasets
    # "ENZYMES",
    # "PROTEINS",
]

init_methods = ["forgy", "kmeans++"]
k_grid = [2**i for i in range(1, 5)]  # 2, 4, 8, 16
h_grid = list(range(11))  # steps 0 to 10

for dataset_name in dataset_names:
    graphs = dataset_to_graphs("../../../data", dataset_name)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    for init_method in init_methods:
        for k in k_grid:
            gwl_eval = GwlEvaluation(
                dataset_name=dataset_name,
                graph_id_label_map=graph_id_label_map,
                cluster_init=init_method,
                n_clusters=k,
                h_grid=h_grid,
                base_dir="../../evaluation-results"
            )

            gwl_eval.evaluate()