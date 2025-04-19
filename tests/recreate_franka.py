import networkx as nx

from thesis.utils.other_utils import has_distinct_edge_labels
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_wl_cv

dataset_name = "PTC_FM"

graphs = dataset_to_graphs("../data", dataset_name)
disjoint_graph = nx.disjoint_union_all(graphs)
has_distinct_edge_labels(disjoint_graph)

graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

c_grid = [10**i for i in range(-3, 4)]  # 10^-3 to 10^3
h_grid = list(range(0, 11))             # h = 0 to 10

mean_acc, std_acc = evaluate_wl_cv(disjoint_graph, graph_id_label_map, h_grid, c_grid, dataset_name=dataset_name)
print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

# mean_acc, std_acc = evaluate_nested_cv(disjoint_graph, graph_labels, h_grid, c_grid, use_cosine=True, dataset_name=dataset_name)
# print(f"[Cosine] Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")