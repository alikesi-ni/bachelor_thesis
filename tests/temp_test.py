import numpy as np
from scipy.sparse import load_npz

X1 = load_npz("GWL_feature_matrix.npz")
X2 = load_npz("my_gwl_simple.npz")

def sparse_equal(A, B):
    if A.shape != B.shape:
        return False
    diff = (A != B)
    return diff.nnz == 0 if hasattr(diff, 'nnz') else np.all(diff == 0)

def arrays_equal(a, b):
    """Check whether two arrays are exactly equal (works for sparse or dense)."""
    if hasattr(a, "toarray"):
        a = a.toarray()
    if hasattr(b, "toarray"):
        b = b.toarray()
    return np.array_equal(np.asarray(a), np.asarray(b))

def compare_arrays(label, a, b):
    same = np.array_equal(a, b)
    print(f"{label}: {'MATCH' if same else 'DIFFER'}")
    if not same:
        print("a:", a)
        print("b:", b)

def compare_splits(name1, ids1, y1, name2, ids2, y2):
    """
    Compare train/test splits from two sources.
    name1, name2 -> names for identification (e.g., "example.py" and "cv")
    ids1, ids2   -> indices (train or test)
    y1, y2       -> corresponding labels
    """
    ids1 = np.asarray(ids1)
    ids2 = np.asarray(ids2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    if np.array_equal(ids1, ids2):
        print(f"[OK] Indices match between {name1} and {name2}")
    else:
        print(f"[DIFF] Indices DO NOT match between {name1} and {name2}")
        print(f"{name1} indices:", ids1)
        print(f"{name2} indices:", ids2)

    if np.array_equal(y1, y2):
        print(f"[OK] Labels match between {name1} and {name2}")
    else:
        print(f"[DIFF] Labels DO NOT match between {name1} and {name2}")
        print(f"{name1} labels:", y1)
        print(f"{name2} labels:", y2)



print("Exactly equal?", sparse_equal(X1, X2))

if arrays_equal(X1, X2):
    print("The arrays are exactly equal.")
else:
    print("The arrays differ.")

data_1 = np.load("split_parameter_example.npz")
data_2 = np.load("split_parameter_simple.npz")

compare_arrays("Graph IDs", data_1["graph_ids"], data_2["graph_ids"])
compare_arrays("Labels y", data_1["graph_labels"], data_2["graph_labels"])


data_1_b = np.load("splits_example.npz")
data_2_b = np.load("splits_simple.npz")

compare_arrays("train_ids", data_1_b["train_ids"], data_2_b["train_ids"])
compare_arrays("test_ids", data_1_b["test_ids"], data_2_b["test_ids"])
compare_arrays("y_train", data_1_b["y_train"], data_2_b["y_train"])
compare_arrays("y_test", data_1_b["y_test"], data_2_b["y_test"])

