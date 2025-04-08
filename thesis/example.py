import pickle
import time

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset

from thesis import refinements
from utils import *
from read_data_utils import graph_data_to_graph_list, dataset_to_graphs

from GWL_python.graph_dataset import GraphDataset
from GWL_python.gwl import GradualWeisfeilerLeman
from GWL_python.kernels import GWLSubtreeKernel
from GWL_python.wl import WeisfeilerLeman

# with open("../GWL_python/data-pickled/PTC_FM", mode="rb") as pickled_data:
#     dataset: GraphDataset = pickle.load(pickled_data)

# dataset = GraphDataset("../GWL_python/data", "PTC_FM")
#
# # retrieve graph ids and labels
# graph_id_label_mapping = dataset.get_graphs_labels()
# #
# graph_ids = np.fromiter(graph_id_label_mapping.keys(), int)
# graph_labels = np.fromiter(graph_id_label_mapping.values(), int)

# disjoint union of all graphs of PTC_FM dataset
# graphA = dataset.get_graphs_as_disjoint_union()
# n = 4
# graphA = dataset.get_graphs()[n+1]
# draw_graph_with_labels(graphA)
#
# tu_dataset = TUDataset("./data", "PTC_FM")
# # graphB = convert_to_disjoint_union_graph(tu_dataset)
# graphB = convert_dataset_to_networkx_graphs(tu_dataset)[n]
# draw_graph_with_labels(graphB)
#
# graphC = graph_data_to_graph_list("./data/PTC_FM")[n]
# draw_graph_with_labels(graphC)

graphA = GraphDataset("./data", "PTC_FM").get_graphs_as_disjoint_union()
graphB = nx.disjoint_union_all(dataset_to_graphs("./data", "PTC_FM"))


graphA_wl = graphA.copy()
graphB_wl = graphB.copy()

graphA_gwl = graphA.copy()
graphB_gwl = graphB.copy()

wl = WeisfeilerLeman(refinement_steps=13)
wl.refine_color(graphA_wl)
refinements.wl_refine(graphB_wl, 13)

#
feature_vectors = wl.generate_feature_vectors(graphA_wl)
print(len(feature_vectors[1]))
feature_vectors_2 = refinements.generate_feature_vectors(graphB_wl)
print(len(feature_vectors_2[1]))

# color refinement using GWL
gwl = GradualWeisfeilerLeman(refinement_steps=4, n_cluster=3)
gwl.refine_color(graph=graphA_gwl)
#
# generate feature vectors for each graph
feature_vectors = gwl.generate_feature_vectors(refined_disjoint_graph=graphA_gwl)
print(len(feature_vectors[1]))
feature_vectors_2 = refinements.generate_feature_vectors(graphA_gwl)
print(len(feature_vectors_2[1]))

if feature_vectors == feature_vectors_2:
    print("The feature vectors are identical.")
else:
    print("The feature vectors are different.")

#
# # generate train and test mask
train_mask, test_mask, y_train, y_test = train_test_split(
    graph_ids, graph_labels, test_size=0.2, shuffle=True, stratify=graph_labels)
#
# # precompute kernels
kernel = GWLSubtreeKernel(normalize=True)
#
K_train = kernel.fit_transform(feature_vectors, train_mask)
K_test = kernel.transform(feature_vectors, train_mask, test_mask)
#
# # Uses the SVC classifier to perform classification
# model = SVC(kernel="precomputed", C=0.001)
# model.fit(K_train, y_train)
#
# predictions = model.predict(K_test)
#
# # Computes and prints the classification accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {round(accuracy * 100, 3)}%")
