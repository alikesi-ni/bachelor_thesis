import csv
import os
import pickle
import time

import networkx as nx
import numpy as np

from tests.test_print_system_info import log_machine_spec
from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.logger_config import LoggerFactory


class QscEvaluation:
    def __init__(self, dataset_name, disjoint_graph: nx.Graph, graph_id_label_map: dict[int, int],
                 base_dir="../evaluation-results", logging=True):
        self.dataset_name = dataset_name
        self.disjoint_graph = disjoint_graph
        sorted_map = dict(sorted(graph_id_label_map.items()))
        self.graph_ids = np.array(list(sorted_map.keys()))
        self.graph_labels = np.array(list(sorted_map.values()))
        self.base_dir = base_dir
        self.last_completed_step = -1
        self.data_dir = f"{self.base_dir}/QSC-{self.dataset_name}"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        train_filename = os.path.join(self.data_dir, "train_results.csv")
        test_filename = os.path.join(self.data_dir, "test_results.csv")
        log_filename = os.path.join(self.data_dir, "evaluation_log.txt")
        self.logger = LoggerFactory.get_full_logger(__name__, log_filename) if logging \
            else LoggerFactory.get_console_logger(__name__, "error")
        log_machine_spec(self.logger)
        self.logger.info("--------------------")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info("Algorithm: QSC")
        self.logger.info("--------------------")
        self.path_to_qsc_graph = None

    def refine_and_create_feature_vector_matrices(self, step=np.inf, q: int = 1):
        fvm_dir = os.path.join(self.data_dir, "feature_vector_matrices")
        os.makedirs(fvm_dir, exist_ok=True)

        cg = ColoredGraph(self.disjoint_graph.copy())
        qsc = QuasiStableColoringGraph(
            cg,
            q=0.0,
            n_colors=np.inf,
            q_tolerance=0.0,
            max_steps=np.inf,
            logger=self.logger
        )

        self.logger.info(f"Computing feature vector for step=0...")

        start_time = time.time()
        fv_matrix = cg.generate_feature_matrix()
        elapsed_time = time.time() - start_time

        params = {
            "step": 0,
            "feature_dim": fv_matrix.shape[1] - 1,
            "max_q_error": qsc.current_max_q_error,
            "partition_count": len(qsc.partitions),
            "witness_pair_count": 0,  # No witnesses at initial state
            "calculation_time_in_seconds": round(elapsed_time, 6)
        }

        #### Save the feature vector matrix ####
        fvm_filename = f"step_0.pkl"
        fvm_path = os.path.join(fvm_dir, fvm_filename)

        with open(fvm_path, "wb") as f:
            pickle.dump((fv_matrix, params), f)

        self.logger.info(
            f"Saved feature vector for step=0 "
            f"(max_q_error={params['max_q_error']:.1f}, "
            f"feature_dim={params['feature_dim']}, "
            f"partition_count={params['partition_count']}, "
            f"witness_pair_count={params['witness_pair_count']}, "
            f"calculation_time={params['calculation_time_in_seconds']:.6f} s)"
        )

        refinement_results_file = os.path.join(self.data_dir, "refinement_results.csv")

        with open(refinement_results_file, "w", newline="") as f_refine:
            writer = csv.writer(f_refine)

            # Write header only if new
            writer.writerow([
                "step", "feature_dim", "max_q_error",
                "partition_count", "witness_pair_count",
                "calculation_time_in_seconds"
            ])
            writer.writerow([
                params["step"],
                params["feature_dim"],
                params["max_q_error"],
                params["partition_count"],
                params["witness_pair_count"],
                params["calculation_time_in_seconds"]
            ])

        with open(os.path.join(self.data_dir, "qsc.pkl"), "wb") as f:
            pickle.dump(qsc, f)

        self.last_completed_step += 1
        self.last_max_q_error = params["max_q_error"]

        while self.last_completed_step < step and self.last_max_q_error > q:

            start_time = time.time()
            self.logger.info(f"Refining for step={self.last_completed_step + 1}...")
            n_colors, current_step, max_q_error, witness_pair_count = qsc.refine_one_step()
            self.logger.info(f"Computing feature vector for step={current_step}...")

            fv_matrix = cg.generate_feature_matrix()
            elapsed_time = round(time.time() - start_time, 6)

            params = {
                "step": current_step,
                "feature_dim": fv_matrix.shape[1] - 1,
                "max_q_error": max_q_error,
                "partition_count": len(qsc.partitions),
                "witness_pair_count": 0,  # Optional: if you want to count witnesses, can add here
                "calculation_time_in_seconds": elapsed_time
            }

            # --- Save FV ---
            fvm_filename = f"step_{current_step}.pkl"
            fvm_path = os.path.join(fvm_dir, fvm_filename)
            fvm_difference = fv_matrix[:, -len(qsc.partitions):]
            with open(fvm_path, "wb") as f:
                pickle.dump((fvm_difference, params), f)

            self.logger.info(
                f"Saved feature vector for step={current_step} "
                f"(max_q_error={params['max_q_error']:.1f}, "
                f"feature_dim={params['feature_dim']}, "
                f"partition_count={params['partition_count']}, "
                f"witness_pair_count={params['witness_pair_count']}, "
                f"time={params['calculation_time_in_seconds']:.6f} s)"
            )

            # --- Append to CSV ---
            with open(refinement_results_file, "a", newline="") as f_refine:
                writer = csv.writer(f_refine)
                writer.writerow([
                    params["step"],
                    params["feature_dim"],
                    params["max_q_error"],
                    params["partition_count"],
                    params["witness_pair_count"],
                    params["calculation_time_in_seconds"]
                ])

            # --- Save updated QSC object ---
            with open(os.path.join(self.data_dir, "qsc.pkl"), "wb") as f:
                pickle.dump(qsc, f)

            # --- Update for the next loop iteration ---
            self.last_completed_step = current_step
            self.last_max_q_error = max_q_error

            if max_q_error == 0.0:
                self.logger.info("Reached stable coloring: q-error is 0. Ending refinement.")
                break