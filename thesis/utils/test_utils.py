import csv
import logging
import pickle
import sys
from datetime import datetime
import time

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.gwl_coloring import GWLColoringGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.logger_config import setup_logger
from thesis.utils.other_utils import has_distinct_edge_labels, convert_to_feature_matrix, has_distinct_node_labels
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph


def evaluate_wl_cv(disjoint_graph, graph_id_label_map, h_grid, c_grid,
                   dataset_name="DATASET", folds=10, logging=True, repeats=1, start_repeat=1):
    from datetime import datetime

    sorted_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_map.keys()))
    y = np.array(list(sorted_map.values()))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "WL"

    train_filename = f"{timestamp}_{dataset_name}_{refinement_method}_train.csv"
    test_filename = f"{timestamp}_{dataset_name}_{refinement_method}_test.csv"
    log_filename = f"{timestamp}_{dataset_name}_{refinement_method}_log.txt"

    logger = setup_logger(__name__, log_filename) if logging else None

    def log(msg):
        if logging:
            logger.info(msg)
        else:
            print(msg)

    log(f"Dataset: {dataset_name}: NO EDGE LABELS")
    log("Algorithm: WLST")
    log(f"Parameters: h_grid={h_grid}, c_grid={c_grid}, folds={folds}, repeats={repeats}, start_repeat={start_repeat}")

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)
        writer_train.writerow(["i", "fold", "C", "h", "accuracy"])
        writer_test.writerow(["i", "fold", "C", "h", "accuracy"])

        for i in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=i)
            log(f"[i={i}] Dataset: {dataset_name}")
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(gids, y), 1):
                best_score = -1
                best_params = None

                for h in h_grid:
                    cg = ColoredGraph(disjoint_graph.copy())
                    start = time.time()
                    wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=h)
                    wl.refine()
                    end = time.time()
                    log(f"[i={i} fold={fold}] h={h} WL time: {end - start:.4f}s")

                    start = time.time()
                    X = cg.generate_feature_matrix()
                    end = time.time()
                    log(f"[i={i} fold={fold}] X.shape={X.shape} generated in {end - start:.4f}s")

                    X_train = X[train_idx]
                    y_train = y[train_idx]

                    inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                    for C in c_grid:
                        inner_scores = []
                        for ti, vi in inner_cv.split(X_train, y_train):
                            K_train = cosine_similarity(X_train[ti], X_train[ti])
                            K_val = cosine_similarity(X_train[vi], X_train[ti])
                            clf = SVC(kernel="precomputed", C=C)
                            clf.fit(K_train, y_train[ti])
                            preds = clf.predict(K_val)
                            inner_scores.append(accuracy_score(y_train[vi], preds))

                        avg_score = np.mean(inner_scores)
                        writer_train.writerow([i, fold, C, h, avg_score])
                        f_train.flush()
                        log(f"[i={i} fold={fold}] C={C} h={h} Accuracy={avg_score:.4f}")
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = (h, C)

                h_best, C_best = best_params
                cg = ColoredGraph(disjoint_graph.copy())
                wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=h_best)
                wl.refine()
                X = cg.generate_feature_matrix()

                K_train = cosine_similarity(X[train_idx], X[train_idx])
                K_test = cosine_similarity(X[test_idx], X[train_idx])
                clf = SVC(kernel="precomputed", C=C_best)
                clf.fit(K_train, y[train_idx])
                preds = clf.predict(K_test)
                acc = accuracy_score(y[test_idx], preds)
                writer_test.writerow([i, fold, C_best, h_best, acc])
                f_test.flush()
                log(f"[i={i} fold={fold}] BEST C={C_best}, h={h_best} # Test Acc: {acc:.4f}")

def evaluate_gwl_cv(disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid,
                    dataset_name="DATASET", folds=10, logging=True, repeats=1, start_repeat=1):
    from datetime import datetime

    sorted_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_map.keys()))
    y = np.array(list(sorted_map.values()))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "GWL"

    train_filename = f"{timestamp}_{dataset_name}_{refinement_method}_train.csv"
    test_filename = f"{timestamp}_{dataset_name}_{refinement_method}_test.csv"
    log_filename = f"{timestamp}_{dataset_name}_{refinement_method}_log.txt"

    logger = setup_logger(log_filename) if logging else None

    def log(msg):
        if logging:
            logger.info(msg)
        else:
            print(msg)

    log(f"Dataset: {dataset_name}")
    log("Algorithm: GWL")
    log(f"Parameters: h_grid={h_grid}, k_grid={k_grid}, c_grid={c_grid}, folds={folds}, repeats={repeats}, start_repeat={start_repeat}")

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)
        writer_train.writerow(["i", "fold", "C", "h", "k", "accuracy"])
        writer_test.writerow(["i", "fold", "C", "h", "k", "accuracy"])

        for i in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=i)
            log(f"[i={i}] Dataset: {dataset_name}")

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(gids, y), 1):
                best_score = -1
                best_params = None

                for h in h_grid:
                    for k in k_grid:
                        cg = ColoredGraph(disjoint_graph.copy())
                        start = time.time()
                        gwl = GWLColoringGraph(cg, refinement_steps=h, n_cluster=k)
                        gwl.refine()
                        end = time.time()
                        log(f"[i={i} fold={fold}] h={h}, k={k} # GWL time: {end - start:.4f}s")

                        start = time.time()
                        X = cg.generate_feature_matrix()
                        end = time.time()
                        log(f"[i={i} fold={fold}] X.shape={X.shape} generated in {end - start:.4f}s")

                        X_train = X[train_idx]
                        y_train = y[train_idx]

                        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                        for C in c_grid:
                            inner_scores = []
                            for ti, vi in inner_cv.split(X_train, y_train):
                                K_train = cosine_similarity(X_train[ti], X_train[ti])
                                K_val = cosine_similarity(X_train[vi], X_train[ti])
                                clf = SVC(kernel="precomputed", C=C)
                                clf.fit(K_train, y_train[ti])
                                preds = clf.predict(K_val)
                                inner_scores.append(accuracy_score(y_train[vi], preds))

                            avg_score = np.mean(inner_scores)
                            writer_train.writerow([i, fold, C, h, k, avg_score])
                            f_train.flush()
                            log(f"[i={i} fold={fold}] C={C}, h={h}, k={k} Accuracy={avg_score:.4f}")

                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = (h, k, C)

                # Final evaluation with best h, k, C
                h_best, k_best, C_best = best_params
                cg = ColoredGraph(disjoint_graph.copy())
                gwl = GWLColoringGraph(cg, refinement_steps=h_best, n_cluster=k_best)
                gwl.refine()
                X = cg.generate_feature_matrix()

                K_train = cosine_similarity(X[train_idx], X[train_idx])
                K_test = cosine_similarity(X[test_idx], X[train_idx])
                clf = SVC(kernel="precomputed", C=C_best)
                clf.fit(K_train, y[train_idx])
                preds = clf.predict(K_test)
                acc = accuracy_score(y[test_idx], preds)
                writer_test.writerow([i, fold, C_best, h_best, k_best, acc])
                f_test.flush()
                log(f"[i={i} fold={fold}] BEST C={C_best}, h={h_best}, k={k_best} # Test Acc: {acc:.4f}")


def evaluate_quasistable_cv(disjoint_graph, graph_id_label_map,
                             q_grid, n_max, c_grid, dataset_name="DATASET", folds=10, logging=True, repeats=10, start_repeat=1):
    sorted_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_map.keys()))
    y = np.array(list(sorted_map.values()))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "QSC"

    train_filename = f"{timestamp}_{dataset_name}_{refinement_method}_train.csv"
    test_filename = f"{timestamp}_{dataset_name}_{refinement_method}_test.csv"
    log_filename = f"{timestamp}_{dataset_name}_{refinement_method}_log.txt"

    logger = setup_logger(__name__, log_filename) if logging else None

    def log(msg):
        if logging:
            logger.info(msg)
        else:
            print(msg)

    log(f"Dataset: {dataset_name}")
    log("Algorithm: QSC")
    log(f"Parameters: q_grid={q_grid}, n_max={n_max}, c_grid={c_grid}, folds={folds}, repeats={repeats}, start_repeat={start_repeat}")

    has_edges = has_distinct_edge_labels(disjoint_graph)
    has_nodes = has_distinct_node_labels(disjoint_graph)

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)
        writer_train.writerow(["i", "fold", "C", "q", "n", "accuracy"])
        writer_test.writerow(["i", "fold", "C", "q", "n", "accuracy"])

        q_grid_sorted = sorted(q_grid, reverse=True)

        for i in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=i)
            log(f"[i={i}] Dataset: {dataset_name}")

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(gids, y), 1):
                best_score = -1
                best_params = None
                cg = ColoredGraph(disjoint_graph.copy())
                if has_edges:
                    wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=1)
                    wl.refine()

                q_n_features = {}

                for q_val in q_grid_sorted:
                    qsc = QuasiStableColoringGraph(cg, q=q_val, n_colors=n_max)
                    qsc.refine()
                    n_val = len(qsc.partitions)
                    X = cg.generate_feature_matrix()
                    q_n_features[(q_val, n_val)] = X
                    log(f"[i={i} fold={fold}] Refined with q={q_val}, n={n_val}")
                    if n_val == n_max:
                        break

                # Train and validate SVMs
                for (q_val, n_val), X in q_n_features.items():
                    X_train = X[train_idx]
                    y_train = y[train_idx]

                    inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                    for C in c_grid:
                        inner_scores = []
                        for ti, vi in inner_cv.split(X_train, y_train):
                            K_train = cosine_similarity(X_train[ti], X_train[ti])
                            K_val = cosine_similarity(X_train[vi], X_train[ti])
                            clf = SVC(kernel="precomputed", C=C)
                            clf.fit(K_train, y_train[ti])
                            preds = clf.predict(K_val)
                            inner_scores.append(accuracy_score(y_train[vi], preds))

                        avg_score = np.mean(inner_scores)
                        writer_train.writerow([i, fold, C, q_val, n_val, avg_score])
                        f_train.flush()
                        log(f"[i={i} fold={fold}] C={C} q={q_val} n={n_val} Accuracy={avg_score:.4f}")

                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = (q_val, n_val, C)

                # Final test
                q_best, n_best, C_best = best_params
                X = q_n_features[(q_best, n_best)]
                K_train = cosine_similarity(X[train_idx], X[train_idx])
                K_test = cosine_similarity(X[test_idx], X[train_idx])
                clf = SVC(kernel="precomputed", C=C_best)
                clf.fit(K_train, y[train_idx])
                preds = clf.predict(K_test)
                acc = accuracy_score(y[test_idx], preds)
                writer_test.writerow([i, fold, C_best, q_best, n_best, acc])
                f_test.flush()
                log(f"[i={i} fold={fold}] BEST C={C_best} q={q_best} n={n_best} # Test Acc: {acc:.4f}")

                # Save the last qsc for inspection
                # base_name = f"{timestamp}_{dataset_name}_i{i}_fold{fold}_q{q_best}_n{n_best}"
                # with open(f"{base_name}_qsc.pkl", "wb") as f_qsc:
                #     pickle.dump(qsc, f_qsc)
                # log(f"[i={i} fold={fold}] Saved QSC to {base_name}_qsc.pkl")

def summarize_repeat_results(test_filename: str, per_repeat: bool = True):
    """
    Summarizes repeated k-fold cross-validation results from a test CSV.

    Args:
        test_filename (str): Path to the CSV file generated by evaluate_wl_cv or evaluate_quasistable_cv.
        per_repeat (bool): Whether to print per-repeat means.

    Returns:
        tuple: (overall_mean, overall_std)
    """
    df = pd.read_csv(test_filename)

    if "i" not in df.columns or "accuracy" not in df.columns:
        raise ValueError("CSV must contain columns: 'i' and 'accuracy'.")

    if per_repeat:
        print("\nMean accuracy per repeat:")
        repeat_means = df.groupby("i")["accuracy"].mean()
        print(repeat_means.to_string())

    all_accuracies = df["accuracy"]
    overall_mean = all_accuracies.mean()
    overall_std = all_accuracies.std()

    print(f"\nOverall mean accuracy: {overall_mean:.4f}")
    print(f"Overall std deviation: {overall_std:.4f}")

    repeat_means = df.groupby("i")["accuracy"].mean()
    overall_mean = repeat_means.mean()
    overall_std = repeat_means.std()

    print(repeat_means)
    print(f"\nMean accuracy across repeats: {overall_mean:.4f}")
    print(f"Standard deviation across repeats: {overall_std:.4f}")

    return overall_mean, overall_std
