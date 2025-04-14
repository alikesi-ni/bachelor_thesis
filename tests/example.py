import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from GWL_python.graph_dataset.graph_dataset import GraphDataset
from GWL_python.gwl.gwl import GradualWeisfeilerLeman
from GWL_python.kernels.wl_subtree import WLSubtreeKernel


dataset_name = "KKI"
dataset = GraphDataset("../data", dataset_name)

# retrieve graph ids and labels
graph_id_label_mapping = dataset.get_graphs_labels()

graph_ids = np.fromiter(graph_id_label_mapping.keys(), int)
graph_labels = np.fromiter(graph_id_label_mapping.values(), int)

# disjoint union of all graphs of PTC_FM dataset
graph = dataset.get_graphs_as_disjoint_union()

# color refinement using GWL
gwl = GradualWeisfeilerLeman(refinement_steps=4, n_cluster=3)
gwl.refine_color(graph=graph)

# Feature vectors (sparse)
feature_vectors_dict = gwl.generate_feature_vectors(refined_disjoint_graph=graph)

# Turn dict into index-aligned list of sparse vectors

# First, determine the dimensionality (highest feature index + 1)
all_indices = [max(vec.keys()) if len(vec) > 0 else 0 for vec in feature_vectors_dict.values()]
vector_dim = max(all_indices) + 1

# Convert each dict to a csr_matrix row
feature_vectors_list = [
    csr_matrix((list(vec.values()), ([0]*len(vec), list(vec.keys()))), shape=(1, vector_dim))
    for i in graph_ids
    for vec in [feature_vectors_dict[i]]
]

# Stack into sparse matrix
from scipy.sparse import vstack
X_sparse = vstack(feature_vectors_list)

# Train/test split (fixed, reproducible, no shuffle)
train_mask, test_mask, y_train, y_test = train_test_split(
    np.arange(len(graph_ids)),
    graph_labels,
    test_size=0.2,
    stratify=graph_labels,
    shuffle=True,
    random_state=1
)

X_train_sparse = X_sparse[train_mask]
X_test_sparse = X_sparse[test_mask]

# ============ Method 1: Linear Kernel ============
K_train_linear = cosine_similarity(X_train_sparse, X_train_sparse)
K_test_linear = cosine_similarity(X_test_sparse, X_train_sparse)

model_linear = SVC(kernel="precomputed", C=0.001)
model_linear.fit(K_train_linear, y_train)
pred_linear = model_linear.predict(K_test_linear)
acc_linear = accuracy_score(y_test, pred_linear)
print(f"[Linear Kernel] Accuracy: {round(acc_linear * 100, 3)}%")

# ============ Method 2: Custom WLSubtreeKernel ============
custom_kernel = WLSubtreeKernel(normalize=True)

# Reuse graph_ids directly as keys into feature_vectors_dict
id_train = [graph_ids[i] for i in train_mask]
id_test = [graph_ids[i] for i in test_mask]

K_train_custom = custom_kernel.fit_transform(feature_vectors_dict, id_train)
K_test_custom = custom_kernel.transform(feature_vectors_dict, id_train, id_test)

model_custom = SVC(kernel="precomputed", C=0.001)
model_custom.fit(K_train_custom, y_train)
pred_custom = model_custom.predict(K_test_custom)
acc_custom = accuracy_score(y_test, pred_custom)
print(f"[WLSubtree Kernel] Accuracy: {round(acc_custom * 100, 3)}%")