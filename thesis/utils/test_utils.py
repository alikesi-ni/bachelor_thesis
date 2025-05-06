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
from scipy.sparse import hstack, csr_matrix

def load_and_accumulate_fvs(main_dir, q_grid):
    """
    For each q threshold, load the FV of the step BEFORE q was crossed,
    and accumulate only the last n_color columns from each step.
    """
    refinement_csv = os.path.join(main_dir, "refinement_results.csv")
    df = pd.read_csv(refinement_csv)
    df = df.sort_values("step").reset_index(drop=True)

    # 1️⃣ Find steps before each q threshold
    selected_steps = []
    for q in q_grid:
        passed = df[df["max_q_error"] <= q]
        if not passed.empty:
            first_step = passed.iloc[0]["step"]
            prev_step = df[df["step"] < first_step]["step"].max()
            if pd.isna(prev_step):
                prev_step = 0
        else:
            prev_step = df["step"].max()
        selected_steps.append(int(prev_step))

    print("Selected steps per q:", dict(zip(q_grid, selected_steps)))

    # 2️⃣ Accumulate FVs
    accumulated_fvs = {}
    previous_fv = None
    previous_n_colors = 0

    for q, step in zip(q_grid, selected_steps):
        fv_filename = os.path.join(main_dir, "feature_vectors", f"step_{step}.pkl")
        with open(fv_filename, "rb") as f:
            fv_matrix, params = pickle.load(f)
            current_n_colors = params["n_colors"]

        # Get only the last n_colors columns
        new_part = fv_matrix[:, -current_n_colors:]

        if previous_fv is None:
            accumulated = new_part
        else:
            accumulated = hstack([previous_fv, new_part])

        accumulated_fvs[q] = (accumulated.tocsr(), params)

        # Update for next round
        previous_fv = accumulated
        previous_n_colors = current_n_colors

    return accumulated_fvs


def evaluate_wl_cv(disjoint_graph, graph_id_label_map, h_grid, c_grid,
                   dataset_name="DATASET", folds=10, logging=True, repeats=1, start_repeat=1):

    sorted_map = dict(sorted(graph_id_label_map.items()))
    graph_ids = np.array(list(sorted_map.keys()))
    graph_labels = np.array(list(sorted_map.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refinement_method = "WL"
    main_dir = f"{dataset_name}-Evaluation-WL-{timestamp}"
    os.makedirs(main_dir, exist_ok=True)

    # Output files
    train_filename = os.path.join(main_dir, "train_results.csv")
    test_filename = os.path.join(main_dir, "test_results.csv")
    log_filename = os.path.join(main_dir, "evaluation_log.txt")
    refinement_filename = os.path.join(main_dir, "refinement_results.csv")

    logger = LoggerFactory.get_full_logger(__name__, log_filename) if logging else LoggerFactory.get_console_logger(__name__, "error")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("Algorithm: WLST")

    #### 1️⃣ Precompute feature vectors ####

    fv_dir = os.path.join(main_dir, "feature_vectors")
    os.makedirs(fv_dir, exist_ok=True)

    refinement_results = []

    # Prepare refinement_steps_grid
    h_grid = sorted(h_grid)
    max_step = max(h_grid)

    cg = ColoredGraph(disjoint_graph.copy())
    wl = WeisfeilerLemanColoringGraph(cg)

    for step in range(max_step + 1):
        if step == 0:
            fv_matrix = cg.generate_feature_matrix()
            n_colors = cg.get_num_colors()
        else:
            n_colors, current_step = wl.refine_one_step()
            fv_matrix = cg.generate_feature_matrix()

        feature_dim = fv_matrix.shape[1] - 1

        fv_filename = os.path.join(fv_dir, f"step_{step}.pkl")
        with open(fv_filename, "wb") as f:
            pickle.dump((fv_matrix, {"step": step, "feature_dim": feature_dim, "n_colors": n_colors}), f)
        logger.info(f"Saved feature vector for step={step} (feature_dim={feature_dim}, n_colors={n_colors})")

        refinement_results.append({"step": step, "feature_dim": feature_dim, "n_colors": n_colors})

        # 🔥 Stop if stable
        if wl.is_stable:
            logger.info(f"Refinement stabilized at step={step} with n_colors={n_colors}. No further refinement.")
            break

    # Write refinement_results.csv
    pd.DataFrame(refinement_results).to_csv(refinement_filename, index=False)

    #### 2️⃣ Cross-validation ####

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train, delimiter=";")
        writer_test = csv.writer(f_test, delimiter=";")
        writer_train.writerow(["trial", "Outer fold", "C", "step", "Inner Accuracy", "n_colors"])
        writer_test.writerow(["trial", "Outer fold", "C", "step", "Outer Test Accuracy", "n_colors"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=trial)
            logger.info(f"[trial {trial}] Starting outer cross-validation")

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels), 1):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                #### Inner CV — hyperparameter search ####
                best_score = -1
                best_params = None

                for step in h_grid:
                    fv_filename = os.path.join(fv_dir, f"step_{step}.pkl")
                    with open(fv_filename, "rb") as f:
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
                            acc = accuracy_score(y_val, y_pred) * 100
                            inner_accuracies.append(acc)

                        avg_inner_acc = np.mean(inner_accuracies)
                        writer_train.writerow([trial, outer_fold, C, step, avg_inner_acc, params["n_colors"]])
                        f_train.flush()

                        logger.info(f"[trial {trial} fold {outer_fold}] step={step} C={C} Avg Inner Acc={avg_inner_acc:.2f}")

                        if avg_inner_acc > best_score:
                            best_score = avg_inner_acc
                            best_params = (step, C, params["n_colors"])

                #### Outer fold test ####
                step_best, C_best, n_colors_best = best_params

                fv_filename = os.path.join(fv_dir, f"step_{step_best}.pkl")
                with open(fv_filename, "rb") as f:
                    fv_matrix, _ = pickle.load(f)

                K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                model = SVC(kernel="precomputed", C=C_best)
                model.fit(K_train, y_train)
                y_pred = model.predict(K_test)
                outer_acc = accuracy_score(y_test, y_pred) * 100

                writer_test.writerow([trial, outer_fold, C_best, step_best, outer_acc, n_colors_best])
                f_test.flush()

                logger.info(f"[trial {trial} fold {outer_fold}] BEST C={C_best} step={step_best} Outer Test Acc={outer_acc:.2f}")

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

        writer_train.writerow(["trial", "Outer fold", "C", "h", "k", "method", "Inner Accuracy"])
        writer_test.writerow(["trial", "Outer fold", "C", "h", "k", "method", "Outer Test Accuracy"])
        writer_trial_acc.writerow(["trial", "Average-Accuracy"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
            logger.info(f"[trial {trial}] Starting outer cross-validation")

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

                        logger.info(f"[trial {trial} fold {outer_fold}] Params {gwl_params} C={model_params['C']} Avg Inner Acc={avg_inner_acc:.2f}")

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

                logger.info(f"[trial {trial} fold {outer_fold}] Outer Test Acc={outer_acc:.2f}")

                outer_fold_accuracies.append(outer_acc)

            #### trial accuracy ####
            logger.info(f"trial {trial}] Outer fold accuracies: {outer_fold_accuracies}")
            trial_avg = np.mean(outer_fold_accuracies)
            writer_trial_acc.writerow([trial, trial_avg])
            f_trial_acc.flush()
            logger.info(f"[trial {trial}] Average outer fold accuracy: {trial_avg:.2f}")

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
    main_dir = f"{dataset_name}-Evaluation-QSC-20250505_231551"
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

    fv_dir = generate_quasistable_feature_vectors(
        disjoint_graph, refinement_steps_grid, fv_dir, logger
    )

    #### 2️⃣ Cross-validation ####
    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        writer_train.writerow(["trial", "fold", "C", "step", "accuracy", "n_colors", "q_error"])
        writer_test.writerow(["trial", "fold", "C", "step", "accuracy", "n_colors", "q_error"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=trial)
            logger.info(f"[trial {trial}] Starting outer cross-validation")

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
                        writer_train.writerow([trial, outer_fold, C, step, avg_inner_acc, params['n_colors'], params['max_q_error']])
                        f_train.flush()

                        logger.info(f"[trial {trial} fold {outer_fold}] step={step} C={C}  Avg Inner Acc={avg_inner_acc:.4f}")

                        if avg_inner_acc > best_score:
                            best_score = avg_inner_acc
                            best_params = (step, C, params)

                #### Outer fold test ####
                if best_params is None:
                    logger.warning(f"No valid parameter combination found for trial {trial} fold {outer_fold}. Skipping.")
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

                writer_test.writerow([trial, outer_fold, C_best, step_best, outer_acc, params_best['n_colors'], params_best['max_q_error']])
                f_test.flush()

                logger.info(f"[trial {trial} fold {outer_fold}] BEST C={C_best} step={step_best} Outer Test Acc={outer_acc:.4f} "
                            f"(n_colors={params_best['n_colors']}, q_error={params_best['max_q_error']:.4f})")

    logger.info("Evaluation complete.")
    return main_dir

def generate_quasistable_feature_vectors(disjoint_graph, refinement_steps_grid, fv_dir, logger):
    """
    Generates and saves feature vectors for a range of refinement steps for Quasi-Stable Coloring.
    Saves the feature vectors and writes a refinement_results.csv.
    """
    os.makedirs(fv_dir, exist_ok=True)

    refinement_steps_grid = sorted(refinement_steps_grid)
    cg = ColoredGraph(disjoint_graph.copy())
    qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=np.inf, q_tolerance=0.0, logger=logger)

    # Step 0 feature vector
    logger.info(f"Computing feature vector for step=0...")
    fv_matrix = cg.generate_feature_matrix()
    params = {
        "n_colors": len(qsc.partitions),
        "step": 0,
        "max_q_error": np.inf
    }

    fv_filename = f"step_0.pkl"
    fv_path = os.path.join(fv_dir, fv_filename)
    with open(fv_path, "wb") as f:
        pickle.dump((fv_matrix, params), f)

    logger.info(
        f"Saved feature vector for step=0 "
        f"(n_colors={params['n_colors']}, max_q_error={params['max_q_error']:.4f})"
    )

    # CSV to record refinement stats
    refinement_results_file = os.path.join(fv_dir, "refinement_results.csv")
    with open(refinement_results_file, "w", newline="") as f_refine:
        writer = csv.writer(f_refine)
        writer.writerow(["step", "feature_dim", "max_q_error", "n_colors"])
        writer.writerow([
            0,
            fv_matrix.shape[1] - 1,
            params["max_q_error"],
            params["n_colors"]
        ])

    saved_steps = set()
    max_requested_step = max(refinement_steps_grid)

    while qsc.refinement_step < max_requested_step:
        n_colors, step, max_q_error = qsc.refine_one_step()

        if step in refinement_steps_grid and step not in saved_steps:
            logger.info(f"Computing feature vector for step={step}...")
            fv_matrix = cg.generate_feature_matrix()
            params = {
                "n_colors": len(qsc.partitions),
                "step": step,
                "max_q_error": max_q_error
            }

            fv_filename = f"step_{step}.pkl"
            fv_path = os.path.join(fv_dir, fv_filename)
            with open(fv_path, "wb") as f:
                pickle.dump((fv_matrix, params), f)

            logger.info(
                f"Saved feature vector for step={step} "
                f"(n_colors={params['n_colors']}, max_q_error={params['max_q_error']:.4f})"
            )

            with open(refinement_results_file, "a", newline="") as f_refine:
                writer = csv.writer(f_refine)
                writer.writerow([
                    params["step"],
                    fv_matrix.shape[1] - 1,
                    params["max_q_error"],
                    params["n_colors"]
                ])

            saved_steps.add(step)

        if qsc.q_error == 0.0:
            logger.info("Q-error reached 0.0 — stopping further refinement.")
            break

    return fv_dir


def get_stats_from_test_results_csv(test_filename: str):
    """
    Summarizes repeated k-fold cross-validation results from a test CSV.

    Args:
        test_filename (str): Path to the CSV file generated by evaluate_wl_cv or evaluate_quasistable_cv.

    Returns:
        tuple: (overall_mean, overall_std)
    """
    df = pd.read_csv(test_filename)

    if "trial" not in df.columns or "accuracy" not in df.columns:
        raise ValueError("CSV must contain columns: 'trial' and 'accuracy'.")

    trial_means = df.groupby("trial")["accuracy"].mean()

    mean = trial_means.mean()
    std = trial_means.std()

    print(f"trial means: {trial_means}")
    print(f"Mean accuracy across trials: {mean:.4f}")
    print(f"Standard deviation across trials: {std:.4f}")

    return mean, std

def load_last_n_color_columns(fv_filepath):
    """Loads a pickled feature vector matrix and extracts only the last n_color columns."""
    with open(fv_filepath, "rb") as f:
        fv_matrix, params = pickle.load(f)
        n_colors = params["n_colors"]

    last_columns = fv_matrix[:, -n_colors:]
    return last_columns

def load_fv_and_params(fv_filepath):
    """Loads a pickled feature vector matrix and its parameters."""
    with open(fv_filepath, "rb") as f:
        fv_matrix, params = pickle.load(f)
    return fv_matrix, params


def load_and_accumulate_fvs(main_dir, q_grid):
    """
    For each q threshold, load the FV of the step BEFORE q was crossed,
    and accumulate only the last n_color columns from each step.
    Returns only the final accumulated feature vector (csr_matrix).
    """
    refinement_csv = os.path.join(main_dir, "refinement_results.csv")
    df = pd.read_csv(refinement_csv)
    df = df.sort_values("step").reset_index(drop=True)

    selected_steps = []
    for q in q_grid:
        passed = df[df["max_q_error"] <= q]
        if not passed.empty:
            first_step = passed.iloc[0]["step"]
            prev_step = df[df["step"] < first_step]["step"].max()
            if pd.isna(prev_step):
                prev_step = 0
        else:
            prev_step = df["step"].max()
        selected_steps.append(int(prev_step))

    print("Selected steps per q:", dict(zip(q_grid, selected_steps)))

    previous_fv = None

    for q, step in zip(q_grid, selected_steps):
        fv_filename = os.path.join(main_dir, "feature_vectors", f"step_{step}.pkl")
        new_part = load_last_n_color_columns(fv_filename)

        if previous_fv is None:
            accumulated = new_part
        else:
            accumulated = hstack([previous_fv, new_part])

        previous_fv = accumulated

    return accumulated

def evaluate_fixed_feature_vector(
    feature_matrix,
    graph_id_label_map,
    C_grid,
    dataset_name="DATASET",
    folds=10,
    repeats=10,
    start_repeat=1,
    output_dir=".",
    logger=None
):

    sorted_map = dict(sorted(graph_id_label_map.items()))
    graph_ids = np.array(list(sorted_map.keys()))
    graph_labels = np.array(list(sorted_map.values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "FixedFV"

    main_dir = os.path.join(output_dir, f"{dataset_name}-Evaluation-{method_name}-{timestamp}")
    os.makedirs(main_dir, exist_ok=True)

    train_filename = os.path.join(main_dir, "train_results.csv")
    test_filename = os.path.join(main_dir, "test_results.csv")

    if logger:
        logger.info(f"Dataset: {dataset_name}")
        logger.info("Algorithm: Fixed Feature Vector Evaluation")
        logger.info(f"C_grid={C_grid}, folds={folds}, repeats={repeats}, start_repeat={start_repeat}")

    with open(train_filename, "w", newline="") as f_train, open(test_filename, "w", newline="") as f_test:
        writer_train = csv.writer(f_train)
        writer_test = csv.writer(f_test)

        writer_train.writerow(["trial", "fold", "C", "Inner Accuracy"])
        writer_test.writerow(["trial", "fold", "C", "accuracy"])

        for trial in range(start_repeat, repeats + 1):
            outer_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=trial)

            if logger:
                logger.info(f"[trial {trial}] Starting outer cross-validation")

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels), 1):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                best_score = -1
                best_C = None

                for C in C_grid:
                    inner_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
                    inner_accuracies = []

                    for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                        x_inner_train = x_train[inner_train_idx]
                        y_inner_train = y_train[inner_train_idx]
                        x_val = x_train[inner_val_idx]
                        y_val = y_train[inner_val_idx]

                        K_train = cosine_similarity(feature_matrix[x_inner_train], feature_matrix[x_inner_train])
                        K_val = cosine_similarity(feature_matrix[x_val], feature_matrix[x_inner_train])

                        model = SVC(kernel="precomputed", C=C)
                        model.fit(K_train, y_inner_train)
                        y_pred = model.predict(K_val)
                        inner_acc = accuracy_score(y_val, y_pred) * 100
                        inner_accuracies.append(inner_acc)

                    avg_inner_acc = np.mean(inner_accuracies)
                    writer_train.writerow([trial, outer_fold, C, avg_inner_acc])
                    f_train.flush()

                    if logger:
                        logger.info(f"[trial {trial} fold {outer_fold}] C={C} Avg Inner Acc={avg_inner_acc:.2f}")

                    if avg_inner_acc > best_score:
                        best_score = avg_inner_acc
                        best_C = C

                # Outer test with best C
                K_train = cosine_similarity(feature_matrix[x_train], feature_matrix[x_train])
                K_test = cosine_similarity(feature_matrix[x_test], feature_matrix[x_train])

                model = SVC(kernel="precomputed", C=best_C)
                model.fit(K_train, y_train)
                y_pred = model.predict(K_test)
                outer_acc = accuracy_score(y_test, y_pred) * 100

                writer_test.writerow([trial, outer_fold, best_C, outer_acc])
                f_test.flush()

                if logger:
                    logger.info(
                        f"[trial {trial} fold {outer_fold}] BEST C={best_C} Outer Test Acc={outer_acc:.2f}"
                    )

    if logger:
        logger.info("Fixed feature vector evaluation complete.")

    return main_dir