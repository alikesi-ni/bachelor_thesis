import itertools
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.gwl_coloring import GWLColoringGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.logger_config import LoggerFactory
from thesis.utils.other_utils import has_distinct_edge_labels, has_distinct_node_labels
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph
import csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_wl_cv(disjoint_graph, graph_id_label_map, h_grid, c_grid,
                   dataset_name="DATASET", folds=10, logging=True, repeats=1, start_repeat=1):

    sorted_map = dict(sorted(graph_id_label_map.items()))
    graph_ids = np.array(list(sorted_map.keys()))
    graph_labels = np.array(list(sorted_map.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "WL"
    main_dir = f"{dataset_name}-Evaluation-Manual-{timestamp}"
    os.makedirs(main_dir, exist_ok=True)

    # Output files
    train_filename = os.path.join(main_dir, "train_results.csv")
    test_filename = os.path.join(main_dir, "test_results.csv")
    log_filename = os.path.join(main_dir, "evaluation_log.txt")

    logger = LoggerFactory.get_full_logger(__name__, log_filename) if logging else LoggerFactory.get_console_logger(__name__, "error")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("Algorithm: WLST")

    #### 1️⃣ Precompute feature vectors ####

    fv_dir = os.path.join(main_dir, "feature_vectors")
    os.makedirs(fv_dir, exist_ok=True)

    for h in h_grid:
        fv_file = os.path.join(fv_dir, f"h-{h}")
        if not os.path.exists(fv_file):
            logger.info(f"Computing feature vector for h={h}...")
            cg = ColoredGraph(disjoint_graph.copy())
            wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=h)
            wl.refine()
            X = cg.generate_feature_matrix()
            logger.info(f"Dimension of feature vector for h={h} is {X.shape[1] - 1}")
            with open(fv_file, "wb") as f:
                pickle.dump(X, f)

    #### 2️⃣ Cross-validation ####

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train, delimiter=";")
        writer_test = csv.writer(f_test, delimiter=";")
        writer_train.writerow(["Trial", "Outer Fold", "C", "h", "Inner Accuracy"])
        writer_test.writerow(["Trial", "Outer Fold", "C", "h", "Outer Test Accuracy"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=trial)
            logger.info(f"[Trial {trial}] Starting outer cross-validation")

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels), 1):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                #### Inner CV — hyperparameter search ####
                best_score = -1
                best_params = None

                for h in h_grid:
                    fv_file = os.path.join(fv_dir, f"h-{h}")
                    with open(fv_file, "rb") as f:
                        fv_matrix = pickle.load(f)

                    for C in c_grid:

                        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                        inner_accuracies = []

                        for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                            x_inner_train = x_train[inner_train_idx]
                            y_inner_train = y_train[inner_train_idx]
                            x_val = x_train[inner_val_idx]
                            y_val = y_train[inner_val_idx]

                            K_train = cosine_similarity(fv_matrix[x_inner_train], fv_matrix[x_inner_train])
                            K_val = cosine_similarity(fv_matrix[x_val], fv_matrix[x_inner_train])

                            model = SVC(kernel="precomputed", C=C)
                            model.fit(K_train, y_inner_train)
                            y_pred = model.predict(K_val)
                            acc = accuracy_score(y_val, y_pred) * 100
                            inner_accuracies.append(acc)

                        avg_inner_acc = np.mean(inner_accuracies)
                        writer_train.writerow([trial, outer_fold, C, h, avg_inner_acc])
                        f_train.flush()

                        logger.info(f"[Trial {trial} Fold {outer_fold}] h={h} C={C} Avg Inner Acc={avg_inner_acc:.2f}")

                        if avg_inner_acc > best_score:
                            best_score = avg_inner_acc
                            best_params = (h, C)

                #### Outer fold test ####
                h_best, C_best = best_params

                fv_file = os.path.join(fv_dir, f"h-{h_best}")
                with open(fv_file, "rb") as f:
                    fv_matrix = pickle.load(f)

                K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                model = SVC(kernel="precomputed", C=C_best)
                model.fit(K_train, y_train)
                y_pred = model.predict(K_test)
                outer_acc = accuracy_score(y_test, y_pred) * 100

                writer_test.writerow([trial, outer_fold, C_best, h_best, outer_acc])
                f_test.flush()

                logger.info(f"[Trial {trial} Fold {outer_fold}] Outer Test Acc={outer_acc:.2f}")

    logger.info("Evaluation complete.")

def evaluate_gwl_cv(disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid,
                    dataset_name="DATASET", folds=10, logging=True, repeats=1, start_repeat=1):

    sorted_map = dict(sorted(graph_id_label_map.items()))
    graph_ids = np.array(list(sorted_map.keys()))
    graph_labels = np.array(list(sorted_map.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "GWL"
    main_dir = f"{dataset_name}-Evaluation-Manual-{timestamp}"
    os.makedirs(main_dir, exist_ok=True)

    # Output files
    train_filename = os.path.join(main_dir, "train_results.csv")
    test_filename = os.path.join(main_dir, "test_results.csv")
    trial_acc_filename = os.path.join(main_dir, "trial-accuracies.csv")
    log_filename = os.path.join(main_dir, "evaluation_log.txt")

    logger = LoggerFactory.get_full_logger(__name__, log_filename) if logging else LoggerFactory.get_console_logger(__name__, "error")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("Algorithm: GWL")

    #### 1️⃣ Prepare parameter grids ####

    gwl_search_space = {
        "refinement-steps": h_grid,
        "num-clusters": k_grid,
        "cluster-init-method": ["forgy"]  # or your default
    }
    gwl_param_combinations = [dict(zip(gwl_search_space.keys(), values))
                              for values in itertools.product(*gwl_search_space.values())]

    model_param_combinations = [{"C": C} for C in c_grid]

    #### 2️⃣ Precompute feature vectors ####
    fv_dir = os.path.join(main_dir, "feature_vectors")
    os.makedirs(fv_dir, exist_ok=True)

    for params in gwl_param_combinations:
        fv_file = os.path.join(fv_dir, "-".join(map(str, params.values())))
        if not os.path.exists(fv_file):
            logger.info(f"Computing feature vector for params {params}...")
            cg = ColoredGraph(disjoint_graph.copy())
            gwl = GWLColoringGraph(cg,
                                   refinement_steps=params["refinement-steps"],
                                   n_cluster=params["num-clusters"],
                                   cluster_initialization_method=params["cluster-init-method"])
            gwl.refine()
            X = cg.generate_feature_matrix()
            with open(fv_file, "wb") as f:
                pickle.dump(X, f)

    #### 3️⃣ Cross-validation ####

    with open(train_filename, "w", newline="") as f_train, \
         open(test_filename, "w", newline="") as f_test, \
         open(trial_acc_filename, "w", newline="") as f_trial_acc:

        writer_train = csv.writer(f_train, delimiter=";")
        writer_test = csv.writer(f_test, delimiter=";")
        writer_trial_acc = csv.writer(f_trial_acc, delimiter=";")

        writer_train.writerow(["Trial", "Outer Fold", "C", "h", "k", "method", "Inner Accuracy"])
        writer_test.writerow(["Trial", "Outer Fold", "C", "h", "k", "method", "Outer Test Accuracy"])
        writer_trial_acc.writerow(["Trial", "Average-Accuracy"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
            logger.info(f"[Trial {trial}] Starting outer cross-validation")

            outer_fold_accuracies = []

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels), 1):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                #### Inner CV — hyperparameter search ####
                best_score = -1
                best_params = None

                for gwl_params in gwl_param_combinations:
                    fv_file = os.path.join(fv_dir, "-".join(map(str, gwl_params.values())))
                    with open(fv_file, "rb") as f:
                        feature_vectors = pickle.load(f)

                    for model_params in model_param_combinations:

                        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
                        inner_accuracies = []

                        for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                            x_inner_train = x_train[inner_train_idx]
                            y_inner_train = y_train[inner_train_idx]
                            x_val = x_train[inner_val_idx]
                            y_val = y_train[inner_val_idx]

                            K_train = cosine_similarity(feature_vectors[x_inner_train], feature_vectors[x_inner_train])
                            K_val = cosine_similarity(feature_vectors[x_val], feature_vectors[x_inner_train])

                            model = SVC(kernel="precomputed", C=model_params["C"])
                            model.fit(K_train, y_inner_train)
                            y_pred = model.predict(K_val)
                            acc = accuracy_score(y_val, y_pred) * 100
                            inner_accuracies.append(acc)

                        avg_inner_acc = np.mean(inner_accuracies)
                        writer_train.writerow([trial, outer_fold, model_params["C"],
                                               gwl_params["refinement-steps"],
                                               gwl_params["num-clusters"],
                                               gwl_params["cluster-init-method"],
                                               avg_inner_acc])
                        f_train.flush()

                        logger.info(f"[Trial {trial} Fold {outer_fold}] Params {gwl_params} C={model_params['C']} Avg Inner Acc={avg_inner_acc:.2f}")

                        if avg_inner_acc > best_score:
                            best_score = avg_inner_acc
                            best_params = (gwl_params, model_params)

                #### Outer fold test ####
                gwl_best, model_best = best_params

                fv_file = os.path.join(fv_dir, "-".join(map(str, gwl_best.values())))
                with open(fv_file, "rb") as f:
                    feature_vectors = pickle.load(f)

                K_train = cosine_similarity(feature_vectors[x_train], feature_vectors[x_train])
                K_test = cosine_similarity(feature_vectors[x_test], feature_vectors[x_train])

                model = SVC(kernel="precomputed", C=model_best["C"])
                model.fit(K_train, y_train)
                y_pred = model.predict(K_test)
                outer_acc = accuracy_score(y_test, y_pred) * 100

                writer_test.writerow([trial, outer_fold, model_best["C"],
                                      gwl_best["refinement-steps"],
                                      gwl_best["num-clusters"],
                                      gwl_best["cluster-init-method"],
                                      outer_acc])
                f_test.flush()

                logger.info(f"[Trial {trial} Fold {outer_fold}] Outer Test Acc={outer_acc:.2f}")

                outer_fold_accuracies.append(outer_acc)

            #### Trial accuracy ####
            logger.info(f"Trial {trial}] Outer fold accuracies: {outer_fold_accuracies}")
            trial_avg = np.mean(outer_fold_accuracies)
            writer_trial_acc.writerow([trial, trial_avg])
            f_trial_acc.flush()
            logger.info(f"[Trial {trial}] Average outer fold accuracy: {trial_avg:.2f}")

    logger.info("Evaluation complete.")


def evaluate_quasistable_cv(disjoint_graph, graph_id_label_map,
                             refinement_steps_grid, c_grid,
                             dataset_name="DATASET", folds=10, logging=True,
                             repeats=10, start_repeat=1):

    sorted_map = dict(sorted(graph_id_label_map.items()))
    graph_ids = np.array(list(sorted_map.keys()))
    graph_labels = np.array(list(sorted_map.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "QSC"

    #### Create output folder ####
    main_dir = f"{dataset_name}-Evaluation-QSC-{timestamp}"
    os.makedirs(main_dir, exist_ok=True)

    train_filename = os.path.join(main_dir, "train_results.csv")
    test_filename = os.path.join(main_dir, "test_results.csv")
    log_filename = os.path.join(main_dir, "evaluation_log.txt")

    logger = LoggerFactory.get_full_logger(__name__, log_filename) if logging else LoggerFactory.get_console_logger(__name__, "error")

    logger.info(f"Dataset: {dataset_name}")
    logger.info("Algorithm: QSC")
    logger.info(f"Parameters: refinement_steps_grid={refinement_steps_grid}, c_grid={c_grid}, folds={folds}, repeats={repeats}, start_repeat={start_repeat}")

    #### 1️⃣ Precompute feature vectors ####
    fv_dir = os.path.join(main_dir, "feature_vectors")
    os.makedirs(fv_dir, exist_ok=True)

    refinement_steps_grid = sorted(refinement_steps_grid)
    cg = ColoredGraph(disjoint_graph.copy())
    qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=np.inf, q_tolerance=0.0, logger=logger)

    logger.info(f"Computing feature vector for refinement_step={0}...")
    fv_matrix = cg.generate_feature_matrix()
    logger.info(f"Dimension of feature vector for refinement_step={0} is {fv_matrix.shape[1] - 1}")
    params = {
        "n_colors": len(qsc.partitions),
        "refinement_steps": 0,
        "q_error": np.inf
    }


    fv_filename = f"step_{0}.pkl"
    fv_path = os.path.join(fv_dir, fv_filename)

    with open(fv_path, "wb") as f:
        pickle.dump((fv_matrix, params), f)

    saved_steps = set()

    max_requested_step = max(refinement_steps_grid)
    while qsc.refinement_step < max_requested_step:
        n_colors, refinement_step, q_error = qsc.refine_one_step()

        if refinement_step in refinement_steps_grid and refinement_step not in saved_steps:
            fv_filename = f"step_{refinement_step}.pkl"
            fv_path = os.path.join(fv_dir, fv_filename)

            logger.info(f"Computing feature vector for refinement_step={refinement_step}...")
            fv_matrix = cg.generate_feature_matrix()
            logger.info(f"Dimension of feature vector for refinement_step={refinement_step} is {fv_matrix.shape[1] - 1}")
            params = {
                "n_colors": len(qsc.partitions),
                "refinement_steps": refinement_step,
                "q_error": q_error
            }

            with open(fv_path, "wb") as f:
                pickle.dump((fv_matrix, params), f)

            saved_steps.add(refinement_step)

            logger.info(f"Saved feature vector for step={refinement_step} "
                        f"(n_colors={params['n_colors']}, q_error={q_error:.4f})")

        if qsc.q_error == 0.0:
            logger.info("Q-error reached 0.0 — stopping further refinement.")
            break

    #### 2️⃣ Cross-validation ####
    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        writer_train.writerow(["Trial", "Fold", "C", "step", "accuracy", "n_colors", "q_error"])
        writer_test.writerow(["Trial", "Fold", "C", "step", "accuracy", "n_colors", "q_error"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=trial)
            logger.info(f"[Trial {trial}] Starting outer cross-validation")

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels), 1):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                #### Inner CV — hyperparameter search ####
                best_score = -1
                best_params = None

                for step in refinement_steps_grid:
                    fv_filename = f"step_{step}.pkl"
                    fv_path = os.path.join(fv_dir, fv_filename)

                    if not os.path.exists(fv_path):
                        logger.warning(f"Feature vector for step={step} not found, skipping.")
                        continue

                    with open(fv_path, "rb") as f:
                        fv_matrix, params = pickle.load(f)

                    for C in c_grid:

                        inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                        inner_accuracies = []

                        for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                            x_inner_train = x_train[inner_train_idx]
                            y_inner_train = y_train[inner_train_idx]
                            x_val = x_train[inner_val_idx]
                            y_val = y_train[inner_val_idx]

                            K_train = cosine_similarity(fv_matrix[x_inner_train], fv_matrix[x_inner_train])
                            K_val = cosine_similarity(fv_matrix[x_val], fv_matrix[x_inner_train])

                            model = SVC(kernel="precomputed", C=C)
                            model.fit(K_train, y_inner_train)
                            y_pred = model.predict(K_val)
                            outer_acc = accuracy_score(y_val, y_pred) * 100
                            inner_accuracies.append(outer_acc)

                        avg_inner_acc = np.mean(inner_accuracies)
                        writer_train.writerow([trial, outer_fold, C, step, avg_inner_acc, params['n_colors'], params['q_error']])
                        f_train.flush()

                        logger.info(f"[Trial {trial} Fold {outer_fold}] step={step} C={C}  Avg Inner Acc={avg_inner_acc:.4f}")

                        if avg_inner_acc > best_score:
                            best_score = avg_inner_acc
                            best_params = (step, C, params)

                #### Outer fold test ####
                if best_params is None:
                    logger.warning(f"No valid parameter combination found for Trial {trial} Fold {outer_fold}. Skipping.")
                    continue

                step_best, C_best, params_best = best_params

                fv_filename = f"step_{step_best}.pkl"
                fv_path = os.path.join(fv_dir, fv_filename)

                with open(fv_path, "rb") as f:
                    fv_matrix, _ = pickle.load(f)

                K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                model = SVC(kernel="precomputed", C=C_best)
                model.fit(K_train, y_train)
                y_pred = model.predict(K_test)
                outer_acc = accuracy_score(y_test, y_pred) * 100

                writer_test.writerow([trial, outer_fold, C_best, step_best, outer_acc, params_best['n_colors'], params_best['q_error']])
                f_test.flush()

                logger.info(f"[Trial {trial} Fold {outer_fold}] BEST C={C_best} step={step_best} Outer Test Acc={outer_acc:.4f} "
                            f"(n_colors={params_best['n_colors']}, q_error={params_best['q_error']:.4f})")

    logger.info("Evaluation complete.")

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

def evaluate_gwl_simple(disjoint_graph, graph_id_label_map, h, k, C,
                        test_size=0.3, random_state=42, dataset_name="DATASET", logging=True):
    """
    Simple single train/test split evaluation for GWL.
    Mirrors example.py style evaluation.
    """
    from datetime import datetime

    sorted_map = dict(sorted(graph_id_label_map.items()))
    gids = np.array(list(sorted_map.keys()))
    y = np.array(list(sorted_map.values()))

    np.savez("../tests/split_parameter_simple.npz",
             graph_ids=gids,
             graph_labels=y)

    # Split data
    train_ids, test_ids, y_train, y_test = train_test_split(
        gids, y, test_size=0.2, random_state=42, stratify=y
    )

    np.savez("splits_simple.npz",
             train_ids=train_ids,
             test_ids=test_ids,
             y_train=y_train,
             y_test=y_test)

    # Color the graph
    cg = ColoredGraph(disjoint_graph.copy())
    gwl = GWLColoringGraph(cg, refinement_steps=h, n_cluster=k)
    gwl.refine()

    # Feature matrix
    X = cg.generate_feature_matrix()
    save_npz("my_gwl_simple.npz", X)

    # Precomputed kernel
    K_train = cosine_similarity(X[train_ids], X[train_ids])
    K_test = cosine_similarity(X[test_ids], X[train_ids])

    # Train and evaluate SVM
    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, y_train)
    preds = clf.predict(K_test)
    acc = accuracy_score(y_test, preds)

    if logging:
        print(f"Dataset: {dataset_name}")
        print(f"h={h}, k={k}, C={C}")
        print(f"Accuracy: {acc:.4f}")

    return acc

