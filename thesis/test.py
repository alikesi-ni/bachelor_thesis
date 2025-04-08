from torch_geometric.datasets import TUDataset
from utils import *

dataset_names = [
    "KKI", "PTC_FM", "COLLAB", "DD", "IMDB-BINARY", "MSRC_9",
    "NCI1", "REDDIT-BINARY", "MUTAG", "ENZYMES", "PROTEINS"
]

# for dataset_name in dataset_names:
#     try:
#         print(f"Loading dataset: {dataset_name}")
#         root = './data/'
#         dataset = TUDataset(root, dataset_name)
#         analyze(dataset)
#     except Exception as e:
#         print(f"Error loading dataset {dataset_name}: {e}")

# dataset_name = 'MUTAG'
# dataset_name = 'KKI'
dataset_name = 'PTC_FM'
# dataset_name = 'COLLAB'
# dataset_name = 'DD'
# dataset_name = 'IMDB-BINARY'

root = './data/'
dataset = TUDataset(root, dataset_name)

# G = nx.complete_graph(5)

# graphs = convert_dataset_to_networkx_graphs(dataset)
disjoint_union_graph = convert_to_disjoint_union_graph(dataset)
print(count_labeled_edges(disjoint_union_graph))
# print(graphs)

# root = './'
# dataset = TUDataset(root, 'PROTEINS', cleaned=True)
# print(dataset)