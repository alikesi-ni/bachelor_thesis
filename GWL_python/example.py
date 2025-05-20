import pickle

import networkx as nx
import numpy as np
from scipy.sparse import save_npz
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from GWL_python.graph_dataset.graph_dataset import GraphDataset
from GWL_python.gwl.gwl import GradualWeisfeilerLeman
from GWL_python.kernels.gwl_subtree import GWLSubtreeKernel
from thesis.utils.other_utils import convert_to_feature_matrix
from thesis.utils.read_data_utils import dataset_to_graphs

dataset_name = "MSRC_9"
dataset = GraphDataset("../data", dataset_name)

###
graphs = dataset_to_graphs("../data", dataset_name)
graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}
graph = nx.disjoint_union_all(graphs)

# retrieve graph ids and labels
graph_id_label_mapping = dataset.get_graphs_labels()


graph_ids = np.fromiter(graph_id_label_mapping.keys(), int)
graph_labels = np.fromiter(graph_id_label_mapping.values(), int)

# disjoint union of all graphs of PTC_FM dataset
graph = dataset.get_graphs_as_disjoint_union()

# color refinement using GWL
gwl = GradualWeisfeilerLeman(refinement_steps=4, n_cluster=3)
gwl.refine_color(graph=graph)

# generate feature vectors for each graph
feature_vectors = gwl.generate_feature_vectors(refined_disjoint_graph=graph)
feature_matrix = convert_to_feature_matrix(feature_vectors)
save_npz("../tests/GWL_feature_matrix.npz", feature_matrix)

np.savez("../tests/split_parameter_example.npz",
         graph_ids=graph_ids,
         graph_labels=graph_labels)

# generate train and test mask
train_mask, test_mask, y_train, y_test = train_test_split(
    graph_ids, graph_labels, test_size=0.2, random_state=42, stratify=graph_labels)


np.savez("../tests/splits_example.npz",
         train_ids=train_mask,
         test_ids=test_mask,
         y_train=y_train,
         y_test=y_test)

# precompute kernels
kernel = GWLSubtreeKernel(normalize=True)

K_train = kernel.fit_transform(feature_vectors, train_mask)
K_test = kernel.transform(feature_vectors, train_mask, test_mask)

# Uses the SVC classifier to perform classification
model = SVC(kernel="precomputed", C=0.001)
model.fit(K_train, y_train)

predictions = model.predict(K_test)

# Computes and prints the classification accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {round(accuracy * 100, 3)}%")
