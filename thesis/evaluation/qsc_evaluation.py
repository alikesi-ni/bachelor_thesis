from datetime import datetime
import os
import csv
import pickle
from typing import Optional

import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from thesis.evaluation.evaluation_parameters import EvaluationParameters
from thesis.utils.logger_config import LoggerFactory
from tests.test_print_system_info import log_machine_spec


class QscEvaluation:
    def __init__(
        self,
        dataset_name: str,
        disjoint_graph: nx.Graph,
        graph_id_label_map: dict[int, int],
        parameters: EvaluationParameters,
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
        self.parameters = parameters
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

        parameters.set_data_dir_path(self.data_dir_path)

        self.eval_output_dir = os.path.join(self.data_dir_path, parameters.to_dirname())
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
        self.logger.info(f"Evaluation: {parameters.to_dirname()}")
        self.logger.info("--------------------")

    def evaluate(self):
        self.logger.info(f"Evaluation Parameters: C={self.c_grid}, folds={self.folds}, repeats={self.repeats}, start_repeat={self.start_repeat}")

        with open(self.train_path, "w", newline="") as f_train, open(self.test_path, "w", newline="") as f_test:
            writer_train = csv.writer(f_train)
            writer_test = csv.writer(f_test)

            writer_train.writerow(["trial", "fold", "C", "step_set", "accuracy", "n_colors", "q_error"])
            writer_test.writerow(["trial", "fold", "C", "step_set", "accuracy", "n_colors", "q_error"])

            for trial in range(self.start_repeat, self.repeats + 1):
                outer_cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=trial)
                self.logger.info(f"[trial {trial}] Starting outer cross-validation")

                for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(self.graph_ids, self.graph_labels), 1):
                    x_train, y_train = self.graph_ids[train_idx], self.graph_labels[train_idx]
                    x_test, y_test = self.graph_ids[test_idx], self.graph_labels[test_idx]

                    best_score = -1
                    best_params = None

                    for step_set in self.parameters.step_sets:
                        try:
                            fv_matrix, params = self.stitch_feature_vectors(step_set)
                        except FileNotFoundError:
                            self.logger.warning(f"Feature vectors for step set {step_set} not found. Skipping.")
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
                            writer_train.writerow([trial, outer_fold, C, step_set, avg_inner_acc,
                                                   params['partition_count'], params['max_q_error']])
                            f_train.flush()

                            self.logger.info(f"[trial {trial} fold {outer_fold}] step_set={step_set} C={C} "
                                             f"Avg Inner Acc={avg_inner_acc:.4f}")

                            if avg_inner_acc > best_score:
                                best_score = avg_inner_acc
                                best_params = (step_set, C, params)

                    if best_params is None:
                        self.logger.warning(f"No valid parameter combination found for trial {trial} fold {outer_fold}. Skipping.")
                        continue

                    step_best, C_best, params_best = best_params
                    fv_matrix, _ = self.stitch_feature_vectors(step_best)

                    K_train = cosine_similarity(fv_matrix[x_train], fv_matrix[x_train])
                    K_test = cosine_similarity(fv_matrix[x_test], fv_matrix[x_train])

                    model = SVC(kernel="precomputed", C=C_best)
                    model.fit(K_train, y_train)
                    y_pred = model.predict(K_test)
                    test_acc = accuracy_score(y_test, y_pred) * 100

                    writer_test.writerow([trial, outer_fold, C_best, step_best, test_acc,
                                          params_best['partition_count'], params_best['max_q_error']])
                    f_test.flush()

                    self.logger.info(f"[trial {trial} fold {outer_fold}] BEST C={C_best} step_set={step_best} "
                                     f"Outer Test Acc={test_acc:.4f} "
                                     f"(n_colors={params_best['partition_count']}, q_error={params_best['max_q_error']:.4f})")

        self.logger.info("Evaluation complete.")
