import os
import csv
import pickle
import pandas as pd
import numpy as np
import networkx as nx

from typing import Optional, List
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from thesis.evaluation.utils import generate_report
from thesis.utils.logger_config import LoggerFactory
from tests.test_print_system_info import log_machine_spec


class WlEvaluation:
    def __init__(
        self,
        dataset_name: str,
        graph_id_label_map: dict[int, int],
        h_grid: Optional[List[int]] = None,
        c_grid: Optional[List[float]] = None,
        folds: int = 10,
        trials: int = 10,
        start_trial: int = 0,
        base_dir: str = "../evaluation-results",
        logging: bool = True
    ):
        if c_grid is None:
            c_grid = [10 ** i for i in range(-3, 4)]
        if h_grid is None:
            h_grid = list(range(0, 11))

        self.dataset_name = dataset_name
        self.graph_ids = np.array(sorted(graph_id_label_map.keys()))
        self.graph_labels = np.array([graph_id_label_map[i] for i in self.graph_ids])

        self.c_grid = c_grid
        self.h_grid = sorted(h_grid)
        self.folds = folds
        self.repeats = trials
        self.start_repeat = start_trial

        self.data_dir_path = os.path.join(base_dir, f"WL-{dataset_name}")
        h_grid_str = "-".join(str(h) for h in self.h_grid)
        self.eval_output_dir = os.path.join(self.data_dir_path, f"h_grid__{h_grid_str}")
        os.makedirs(self.eval_output_dir, exist_ok=True)

        self.fvm_dir_path = os.path.join(self.data_dir_path, "feature_vector_matrices")
        self.train_path = os.path.join(self.eval_output_dir, "train_results.csv")
        self.test_path = os.path.join(self.eval_output_dir, "test_results.csv")
        self.log_path = os.path.join(self.eval_output_dir, "evaluation_log.txt")

        log_file_exists = os.path.exists(self.log_path)
        self.logger = LoggerFactory.get_full_logger(__name__, self.log_path) if logging \
            else LoggerFactory.get_console_logger(__name__, "error")

        if log_file_exists:
            self.logger.info("====================")
            self.logger.info("CONTINUING EVALUATION")
            self.logger.info("====================")

        log_machine_spec(self.logger)
        self.logger.info("--------------------")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info("Algorithm: WL (Weisfeiler–Leman)")
        self.logger.info(f"h_grid: {self.h_grid}")
        self.logger.info("--------------------")

    def run(self):
        self.logger.info(
            f"Evaluation Parameters: C={self.c_grid}, folds={self.folds}, repeats={self.repeats}, start_repeat={self.start_repeat}"
        )

        existing_results = set()
        if os.path.exists(self.test_path):
            try:
                df_existing = pd.read_csv(self.test_path)
                for _, row in df_existing.iterrows():
                    existing_results.add((int(row["trial"]), int(row["fold"])))
                self.logger.info(f"Loaded {len(existing_results)} completed (trial, fold) pairs.")
            except Exception as e:
                self.logger.warning(f"Failed to load existing test results: {e}")

        with open(self.train_path, "a", newline="") as f_train, open(self.test_path, "a", newline="") as f_test:
            writer_train = csv.writer(f_train)
            writer_test = csv.writer(f_test)

            if os.path.getsize(self.train_path) == 0:
                writer_train.writerow(["trial", "fold", "C", "param", "accuracy"])
            if os.path.getsize(self.test_path) == 0:
                writer_test.writerow(["trial", "fold", "C", "param", "accuracy"])

            for trial in range(self.start_repeat, self.repeats):
                outer_cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=trial)
                self.logger.info(f"[trial {trial}] Starting outer cross-validation")

                for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.graph_ids, self.graph_labels)):
                    if (trial, outer_fold) in existing_results:
                        self.logger.info(f"[trial {trial} fold {outer_fold}] Already completed. Skipping.")
                        continue

                    # Remove outdated training rows for this trial/fold
                    if os.path.exists(self.train_path):
                        with open(self.train_path, "r", newline="") as f:
                            lines = f.readlines()
                        if len(lines) > 1:
                            df_train = pd.read_csv(self.train_path)
                            df_train_filtered = df_train[
                                ~((df_train["trial"] == trial) & (df_train["fold"] >= outer_fold))
                            ]
                            with open(self.train_path, "w", newline="") as f_train_filtered:
                                writer = csv.writer(f_train_filtered)
                                writer.writerow(["trial", "fold", "C", "param", "accuracy"])
                                for _, row in df_train_filtered.iterrows():
                                    writer.writerow([
                                        int(row["trial"]),
                                        int(row["fold"]),
                                        float(row["C"]),
                                        int(row["param"]),
                                        float(row["accuracy"])
                                    ])
                        else:
                            self.logger.info(f"Skipping train cleanup — only header present.")

                    x_train, y_train = self.graph_ids[train_idx], self.graph_labels[train_idx]
                    x_test, y_test = self.graph_ids[test_idx], self.graph_labels[test_idx]

                    best_score = -1
                    best_params = None

                    for step in self.h_grid:
                        fv_path = os.path.join(self.fvm_dir_path, f"step_{step}.pkl")
                        if not os.path.exists(fv_path):
                            self.logger.warning(f"Missing FV file for step {step}, skipping.")
                            continue
                        with open(fv_path, "rb") as f:
                            fv_matrix, _ = pickle.load(f)

                        for C in self.c_grid:
                            inner_cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=0)
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
                            writer_train.writerow([trial, outer_fold, C, step, avg_inner_acc])
                            f_train.flush()

                            self.logger.info(f"[trial {trial} fold {outer_fold}] step={step} C={C} "
                                             f"Avg Inner Acc={avg_inner_acc:.4f}")

                            if avg_inner_acc > best_score:
                                best_score = avg_inner_acc
                                best_params = (step, C)

                    if best_params is None:
                        self.logger.warning(f"No valid parameter combination for trial {trial} fold {outer_fold}. Skipping.")
                        continue

                    step_best, C_best = best_params
                    fv_path = os.path.join(self.fvm_dir_path, f"step_{step_best}.pkl")
                    with open(fv_path, "rb") as f:
                        fv_matrix, _ = pickle.load(f)

                    K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                    K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                    model = SVC(kernel="precomputed", C=C_best)
                    model.fit(K_train, y_train)
                    y_pred = model.predict(K_test)
                    test_acc = accuracy_score(y_test, y_pred) * 100

                    writer_test.writerow([trial, outer_fold, C_best, step_best, test_acc])
                    f_test.flush()

                    self.logger.info(f"[trial {trial} fold {outer_fold}] BEST step={step_best} C={C_best} "
                                     f"Outer Test Acc={test_acc:.4f}")

        # Generate report if not already there
        report_path = os.path.join(self.eval_output_dir, "report.txt")
        if not os.path.exists(report_path):
            accuracy, std = generate_report(self.eval_output_dir, self.eval_output_dir)
            self.logger.info("Evaluation complete.")
            self.logger.info(f"Mean accuracy: {accuracy:.2f} ± {std:.2f}")
            self.logger.info("--------------------")
            return accuracy, std
        else:
            self.logger.info("Evaluation already complete — report exists.")
            return None, None
