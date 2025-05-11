from datetime import datetime
import os
import csv
import pickle
from typing import Optional

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from thesis.evaluation.evaluation_parameters import EvaluationParameters
from thesis.evaluation.step_settings import StepSettings
from thesis.evaluation.utils import stitch_feature_vectors, generate_report
from thesis.utils.logger_config import LoggerFactory
from tests.test_print_system_info import log_machine_spec


class QscEvaluation:
    def __init__(
        self,
        dataset_name: str,
        disjoint_graph: nx.Graph,
        graph_id_label_map: dict[int, int],
        step_settings: StepSettings,
        c_grid: Optional[list[float]] = None,
        folds: int = 10,
        repeats: int = 10,
        start_repeat: int = 1,
        base_dir: str = "../evaluation-results",
        logging: bool = True
    ):
        if c_grid is None:
            c_grid = [10 ** i for i in range(-3, 4)]  # C ∈ {1e-3 to 1e3}

        self.c_grid = c_grid

        self.dataset_name = dataset_name
        self.disjoint_graph = disjoint_graph
        self.step_settings = step_settings
        self.c_grid = c_grid
        self.folds = folds
        self.repeats = repeats
        self.start_repeat = start_repeat

        sorted_map = dict(sorted(graph_id_label_map.items()))
        self.graph_ids = np.array(list(sorted_map.keys()))
        self.graph_labels = np.array(list(sorted_map.values()))

        self.base_dir = base_dir
        self.data_dir_path = os.path.join(self.base_dir, f"QSC-{self.dataset_name}")
        if not os.path.exists(self.data_dir_path):
            raise FileNotFoundError(f"Expected data directory does not exist: {self.data_dir_path}")

        self.fvm_dir_path = os.path.join(self.data_dir_path, "feature_vector_matrices")
        if not os.path.isdir(self.fvm_dir_path):
            raise FileNotFoundError("Missing feature_vector_matrices directory.")

        self.refinement_results_file_path = os.path.join(self.data_dir_path, "refinement_results.csv")
        if not os.path.isfile(self.refinement_results_file_path):
            raise FileNotFoundError("Missing refinement_results.csv.")

        self.eval_output_dir = os.path.join(self.data_dir_path, step_settings.to_dirname())
        os.makedirs(self.eval_output_dir, exist_ok=True)

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
        self.logger.info("Algorithm: QSC")
        self.logger.info(f"Evaluation: {step_settings.to_dirname()}")
        self.logger.info("--------------------")
        self.logger.info("Parameter and associated steps:")
        for param, steps in self.step_settings.get_list_param_steps(self.data_dir_path):
            self.logger.info(f"  param={param} -> steps={steps}")
        self.logger.info("--------------------")

    def evaluate(self):
        self.logger.info(
            f"Evaluation Parameters: C={self.c_grid}, folds={self.folds}, repeats={self.repeats}, start_repeat={self.start_repeat}"
        )

        existing_results = set()

        # load existing test results if available
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

            # write headers if empty
            if os.path.getsize(self.train_path) == 0:
                writer_train.writerow(["trial", "fold", "C", "param", "accuracy"])
            if os.path.getsize(self.test_path) == 0:
                writer_test.writerow(["trial", "fold", "C", "param", "accuracy"])

            for trial in range(self.start_repeat, self.repeats + 1):
                outer_cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=trial)
                self.logger.info(f"[trial {trial}] Starting outer cross-validation")

                for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.graph_ids, self.graph_labels),
                                                                   1):
                    if (trial, outer_fold) in existing_results:
                        self.logger.info(f"[trial {trial} fold {outer_fold}] Already completed. Skipping.")
                        continue

                    # Remove outdated training rows for this trial/fold and beyond
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
                            self.logger.info(
                                f"Skipping train cleanup — {self.train_path} contains only headers or is empty.")

                    x_train, y_train = self.graph_ids[train_idx], self.graph_labels[train_idx]
                    x_test, y_test = self.graph_ids[test_idx], self.graph_labels[test_idx]

                    best_score = -1
                    best_params = None

                    for param, steps in self.step_settings.get_list_param_steps(self.data_dir_path):
                        try:
                            fv_matrix = stitch_feature_vectors(self.data_dir_path, steps)
                        except FileNotFoundError:
                            self.logger.warning(
                                f"Feature vectors for steps={steps} (param={param}) not found. Skipping.")
                            continue

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
                            writer_train.writerow([trial, outer_fold, C, param, avg_inner_acc])
                            f_train.flush()

                            self.logger.info(f"[trial {trial} fold {outer_fold}] param={param} C={C} "
                                             f"Avg Inner Acc={avg_inner_acc:.4f}")

                            if avg_inner_acc > best_score:
                                best_score = avg_inner_acc
                                best_params = (param, C)

                    if best_params is None:
                        self.logger.warning(
                            f"No valid parameter combination found for trial {trial} fold {outer_fold}. Skipping.")
                        continue

                    param_best, C_best = best_params
                    steps_best = dict(self.step_settings.get_list_param_steps(self.data_dir_path))[param_best]
                    fv_matrix = stitch_feature_vectors(self.data_dir_path, steps_best)

                    K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                    K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                    model = SVC(kernel="precomputed", C=C_best)
                    model.fit(K_train, y_train)
                    y_pred = model.predict(K_test)
                    test_acc = accuracy_score(y_test, y_pred) * 100

                    writer_test.writerow([trial, outer_fold, C_best, param_best, test_acc])
                    f_test.flush()

                    self.logger.info(f"[trial {trial} fold {outer_fold}] BEST param={param_best} C={C_best} "
                                     f"Outer Test Acc={test_acc:.4f}")

        # Only generate report if it doesn't already exist
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


