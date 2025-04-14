import csv

import numpy as np
from scipy.sparse import vstack, csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.other_utils import generate_feature_vectors
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph


def sparse_vector_from_dict(vec_dict: dict, dim=None) -> csr_matrix:
    if not vec_dict:
        return csr_matrix((1, 1))
    indices = list(vec_dict.keys())
    values = list(vec_dict.values())
    if dim is None:
        dim = max(indices) + 1
    return csr_matrix((values, ([0]*len(indices), indices)), shape=(1, dim))

def evaluate_nested_cv(disjoint_graph, graph_labels: dict, h_grid, c_grid, use_cosine=False, dataset_name="DATASET"):
    gids = np.array(list(graph_labels.keys()))
    y = np.array(list(graph_labels.values()))
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []

    suffix = "wl_cosine" if use_cosine else "wl_linear"
    filename = f"{dataset_name}_{suffix}.csv"

    # Create/open file for appending
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "h", "accuracy"])  # header

        for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(gids, y), 1):
            outer_train_gids = gids[outer_train_idx]
            outer_test_gids = gids[outer_test_idx]

            best_score = -1
            best_params = None

            for h in h_grid:
                cg = ColoredGraph(disjoint_graph.copy())
                wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=h)
                wl.refine()

                fv_dict = generate_feature_vectors(cg.graph)
                dim = max(max(d.keys(), default=0) for d in fv_dict.values()) + 1

                X_train = vstack([sparse_vector_from_dict(fv_dict[gid], dim=dim) for gid in outer_train_gids])
                y_train = y[outer_train_idx]

                inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
                for C in c_grid:
                    inner_scores = []
                    for train_idx, val_idx in inner_cv.split(X_train, y_train):
                        K_train = cosine_similarity(X_train[train_idx], X_train[train_idx]) if use_cosine else linear_kernel(X_train[train_idx])
                        K_val = cosine_similarity(X_train[val_idx], X_train[train_idx]) if use_cosine else linear_kernel(X_train[val_idx], X_train[train_idx])
                        clf = SVC(kernel="precomputed", C=C)
                        clf.fit(K_train, y_train[train_idx])
                        preds = clf.predict(K_val)
                        inner_scores.append(accuracy_score(y_train[val_idx], preds))

                    avg_score = np.mean(inner_scores)
                    writer.writerow([C, h, avg_score])        # Write immediately
                    f.flush()                                # Ensure write is not buffered
                    print(f"[Outer Fold {outer_fold_idx}] C={C}, h={h}, Accuracy={avg_score:.4f}")

                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = (h, C)

            # Final outer test eval
            h_best, C_best = best_params
            cg = ColoredGraph(disjoint_graph.copy())
            wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=h_best)
            wl.refine()
            fv_dict = generate_feature_vectors(cg.graph)
            dim = max(max(d.keys(), default=0) for d in fv_dict.values()) + 1
            X = vstack([sparse_vector_from_dict(fv_dict[gid], dim=dim) for gid in gids])

            K_train = cosine_similarity(X[outer_train_idx], X[outer_train_idx]) if use_cosine else linear_kernel(X[outer_train_idx])
            K_test = cosine_similarity(X[outer_test_idx], X[outer_train_idx]) if use_cosine else linear_kernel(X[outer_test_idx], X[outer_train_idx])

            clf = SVC(kernel="precomputed", C=C_best)
            clf.fit(K_train, y[outer_train_idx])
            preds = clf.predict(K_test)
            acc = accuracy_score(y[outer_test_idx], preds)
            accuracies.append(acc)

            print(f"[Outer Fold {outer_fold_idx}] BEST C={C_best}, h={h_best} → Test Accuracy: {acc:.4f}")
            print("-" * 60)

    return np.mean(accuracies), np.std(accuracies)