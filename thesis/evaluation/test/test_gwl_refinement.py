import networkx as nx

from thesis.evaluation.gwl_refinement import GwlRefinement
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

init_methods = ["forgy", "kmeans++"]
k_grid = [2**i for i in range(1, 5)]  # 2, 4, 8, 16
h_grid = list(range(11))  # 0 through 10
max_h = max(h_grid)

for dataset_name in dataset_names:

    graphs = dataset_to_graphs("../../../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    for init_method in init_methods:
        for k in k_grid:
            gwl_refinement = GwlRefinement(
                dataset_name=dataset_name,
                disjoint_graph=disjoint_graph,
                graph_id_label_map=graph_id_label_map,
                n_clusters=k,
                cluster_init=init_method,
                refinement_steps=max_h,
                base_dir="../../evaluation-results",
                logging=True,
            )
            gwl_refinement.run()