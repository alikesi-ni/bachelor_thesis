import pickle

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from graph_dataset import GraphDataset
from gwl import GradualWeisfeilerLeman
from kernels import GWLSubtreeKernel

with open("./data-pickled/PTC_FM", mode="rb") as pickled_data:
    dataset: GraphDataset = pickle.load(pickled_data)

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

# generate train and test mask
train_mask, test_mask, y_train, y_test = train_test_split(
    graph_ids, graph_labels, test_size=0.2, shuffle=True, stratify=graph_labels)

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
