import csv
from datetime import datetime
import time

from typing import Dict

import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph


def evaluate_wl_cv(disjoint_graph: nx.Graph, graph_id_label_map: Dict[int, int], h_grid, c_grid, dataset_name="DATASET"):
    sorted_graph_id_label_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_graph_id_label_map.keys()))
    y = np.array(list(sorted_graph_id_label_map.values()))
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    accuracies = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{dataset_name}_normalized.csv"

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
                colored_graph = ColoredGraph(disjoint_graph)
                wl = WeisfeilerLemanColoringGraph(colored_graph, refinement_steps=h)
                wl.refine()

                # Time the dictionary-based method
                start = time.time()
                test = colored_graph.generate_gid_to_feature_vector_map()
                end = time.time()
                print(f"generate_gid_to_feature_vector_map: {end - start:.6f} seconds")

                # Time the matrix-based method
                start = time.time()
                X = colored_graph.generate_feature_matrix()
                end = time.time()
                print(f"generate_feature_matrix: {end - start:.6f} seconds")

                X_train = X[outer_train_gids]
                y_train = y[outer_train_idx]

                inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
                for C in c_grid:
                    inner_scores = []
                    for train_idx, val_idx in inner_cv.split(X_train, y_train):
                        K_train = cosine_similarity(X_train[train_idx], X_train[train_idx])
                        K_val = cosine_similarity(X_train[val_idx], X_train[train_idx])
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
            colored_graph = ColoredGraph(disjoint_graph)
            wl = WeisfeilerLemanColoringGraph(colored_graph, refinement_steps=h_best)
            wl.refine()

            X = colored_graph.generate_feature_matrix()

            K_train = cosine_similarity(X[outer_train_idx], X[outer_train_idx])
            K_test = cosine_similarity(X[outer_test_idx], X[outer_train_idx])

            clf = SVC(kernel="precomputed", C=C_best)
            clf.fit(K_train, y[outer_train_idx])
            preds = clf.predict(K_test)
            acc = accuracy_score(y[outer_test_idx], preds)
            accuracies.append(acc)

            print(f"[Outer Fold {outer_fold_idx}] BEST C={C_best}, h={h_best} → Test Accuracy: {acc:.4f}")
            print("-" * 60)

    return np.mean(accuracies), np.std(accuracies)


def evaluate_quasistable_cv(disjoint_graph: nx.Graph, graph_id_label_map: dict[int, int],
                             q_grid, n_grid, c_grid, dataset_name="DATASET"):
    sorted_graph_id_label_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_graph_id_label_map.keys()))
    y = np.array(list(sorted_graph_id_label_map.values()))
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    accuracies = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{dataset_name}_normalized.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "q", "n", "accuracy"])

        for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(gids, y), 1):
            outer_train_gids = gids[outer_train_idx]
            outer_test_gids = gids[outer_test_idx]

            best_score = -1
            best_params = None

            for q in q_grid:
                for n in n_grid:
                    colored_graph = ColoredGraph(disjoint_graph.copy())
                    qsc = QuasiStableColoringGraph(colored_graph, q=q, n_colors=n)
                    qsc.refine()

                    # Time the matrix-based method
                    start = time.time()
                    X = colored_graph.generate_feature_matrix()
                    end = time.time()
                    print(f"[q={q}, n={n}] generate_feature_matrix: {end - start:.6f} seconds")

                    X_train = X[outer_train_idx]
                    y_train = y[outer_train_idx]

                    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
                    for C in c_grid:
                        inner_scores = []
                        for train_idx, val_idx in inner_cv.split(X_train, y_train):
                            K_train = cosine_similarity(X_train[train_idx], X_train[train_idx])
                            K_val = cosine_similarity(X_train[val_idx], X_train[train_idx])
                            clf = SVC(kernel="precomputed", C=C)
                            clf.fit(K_train, y_train[train_idx])
                            preds = clf.predict(K_val)
                            inner_scores.append(accuracy_score(y_train[val_idx], preds))

                        avg_score = np.mean(inner_scores)
                        writer.writerow([C, q, n, avg_score])
                        f.flush()
                        print(f"[Outer Fold {outer_fold_idx}] C={C}, q={q}, n={n}, Accuracy={avg_score:.4f}")

                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = (q, n, C)

            # Final outer test eval
            q_best, n_best, C_best = best_params
            colored_graph = ColoredGraph(disjoint_graph.copy())
            qsc = QuasiStableColoringGraph(colored_graph, q=q_best, n_colors=n_best)
            qsc.refine()
            X = colored_graph.generate_feature_matrix()

            K_train = cosine_similarity(X[outer_train_idx], X[outer_train_idx])
            K_test = cosine_similarity(X[outer_test_idx], X[outer_train_idx])

            clf = SVC(kernel="precomputed", C=C_best)
            clf.fit(K_train, y[outer_train_idx])
            preds = clf.predict(K_test)
            acc = accuracy_score(y[outer_test_idx], preds)
            accuracies.append(acc)

            print(f"[Outer Fold {outer_fold_idx}] BEST C={C_best}, q={q_best}, n={n_best} → Test Accuracy: {acc:.4f}")
            print("-" * 60)

    return np.mean(accuracies), np.std(accuracies)