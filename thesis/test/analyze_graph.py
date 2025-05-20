from thesis.utils.other_utils import analyze
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
    # "PROTEINS"
]

for dataset_name in dataset_names:
    graphs = dataset_to_graphs("../../data", dataset_name)
    analyze(graphs)