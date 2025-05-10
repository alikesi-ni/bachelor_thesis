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


class QscRefinement:
    def __init__(self, dataset_name, disjoint_graph: nx.Graph, graph_id_label_map: dict[int, int],
                 base_dir="../evaluation-results", logging=True):
        self.dataset_name = dataset_name
        self.disjoint_graph = disjoint_graph
        sorted_map = dict(sorted(graph_id_label_map.items()))
        self.graph_ids = np.array(list(sorted_map.keys()))
        self.graph_labels = np.array(list(sorted_map.values()))
        self.base_dir = base_dir
        self.last_completed_step = -1
        self.data_dir_path = f"{self.base_dir}/QSC-{self.dataset_name}"
        if not os.path.exists(self.data_dir_path):
            os.makedirs(self.data_dir_path)
        self.fvm_dir_path = os.path.join(self.data_dir_path, "feature_vector_matrices")
        self.qsc_file_path = os.path.join(self.data_dir_path, "qsc.pkl")
        self.refinement_results_file_path = os.path.join(self.data_dir_path, "refinement_results.csv")
        train_filename = os.path.join(self.data_dir_path, "train_results.csv")
        test_file_path = os.path.join(self.data_dir_path, "test_results.csv")
        log_file_path = os.path.join(self.data_dir_path, "evaluation_log.txt")
        log_file_exists = False
        if os.path.exists(log_file_path):
            log_file_exists = True
        self.logger = LoggerFactory.get_full_logger(__name__, log_file_path) if logging \
            else LoggerFactory.get_console_logger(__name__, "error")
        if log_file_exists:
            self.logger.info("====================")
            self.logger.info("CONTINUING REFINEMENT")
            self.logger.info("====================")
        log_machine_spec(self.logger)
        self.logger.info("--------------------")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info("Algorithm: QSC")
        self.logger.info("--------------------")
        self.path_to_qsc_graph = None

    def refine_and_create_feature_vector_matrices(self, max_step=np.inf, q: int = 1, max_color_count: int =4096):
        fvm_dir_exists = os.path.exists(self.fvm_dir_path)
        qsc_file_exists = os.path.isfile(self.qsc_file_path)
        refinement_results_exists = os.path.exists(self.refinement_results_file_path)
        if not fvm_dir_exists and not qsc_file_exists and not refinement_results_exists:
            self._do_normal_workflow(q, max_step, max_color_count)
        else:
            self._do_load_workflow(q, max_step, max_color_count)

    def _do_load_workflow(self, q, max_step, max_color_count):
        if not os.path.exists(self.fvm_dir_path):
            self.logger.error(f"Aborting: feature_vector_matrices folder not found.")
            return

        if not os.path.isfile(self.qsc_file_path):
            self.logger.error("Aborting: qsc.pkl file not found.")
            return

        try:
            with open(self.qsc_file_path, "rb") as f:
                qsc = pickle.load(f)
            self.logger.info("Loaded QSC object.")
        except Exception as e:
            self.logger.error(f"Aborting: failed to load qsc.pkl ({e})")
            return

        expected_steps = range(qsc.current_step + 1)  # includes step 0 to h
        missing_steps = []

        for step in expected_steps:
            filename = f"step_{step}.pkl"
            full_path = os.path.join(self.fvm_dir_path, filename)
            if not os.path.isfile(full_path):
                missing_steps.append(filename)

        if missing_steps:
            self.logger.error(f"Aborting: missing feature vector files: {', '.join(missing_steps)}")
            return

        current_step = qsc.current_step
        current_step_fvm_path = os.path.join(self.fvm_dir_path, f"step_{current_step}.pkl")

        with open(current_step_fvm_path, "rb") as f:
            fv_matrix_loaded, params = pickle.load(f)

        if params.get("step") != current_step:
            self.logger.error(f"Aborting: step mismatch — file has step={params.get('step')}, "
                              f"but qsc.current_step={current_step}.")
            return

        if round(qsc.current_max_q_error, 1) != round(params.get("max_q_error", -1), 1):
            self.logger.error("Aborting: max_q_error mismatch between qsc and saved params.")
            return

        if len(qsc.partitions) != params.get("partition_count"):
            self.logger.error("Aborting: partition count mismatch between qsc and saved params.")
            return

        fv_matrix_now = qsc.colored_graph.generate_feature_matrix()
        feature_dim_now = fv_matrix_now.shape[1] - 1
        if feature_dim_now != params.get("feature_dim"):
            self.logger.error(f"Aborting: feature_dim mismatch — CG reports {feature_dim_now}, "
                              f"but file has {params.get('feature_dim')}.")
            return

        if not os.path.isfile(self.refinement_results_file_path):
            self.logger.error("Aborting: refinement_results.csv not found.")
            return

        with open(self.refinement_results_file_path, "r", newline="") as f_csv:
            reader = list(csv.reader(f_csv))
            if len(reader) < 2:
                self.logger.error("Aborting: refinement_results.csv does not contain any data rows.")
                return
            header, *rows = reader
            last_row = rows[-1]

            expected_values = [
                str(params["step"]),
                str(params["feature_dim"]),
                str(params['max_q_error']),
                str(params["partition_count"]),
                str(params["witness_pair_count"]),
                str(params['calculation_time_in_seconds'])
            ]

            last_row_trimmed = [v.strip() for v in last_row]
            if last_row_trimmed != expected_values:
                self.logger.error(
                    f"Aborting: last row in refinement_results.csv does not match saved params.\n"
                    f"Expected: {expected_values}\n"
                    f"Found:    {last_row_trimmed}"
                )
                return

        self.logger.info(f"Step {current_step}: all files and logs match the current QSC state. Continuing...")

        self._do_common_workflow(q, max_step, max_color_count, qsc)


    def _do_normal_workflow(self, q, max_step, max_color_count):
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
            "max_q_error": round(qsc.current_max_q_error, 1),
            "partition_count": len(qsc.partitions),
            "witness_pair_count": 0,  # No witnesses at initial state
            "calculation_time_in_seconds": round(elapsed_time, 6)
        }

        # save feature vector matrix
        fvm_filename = f"step_0.pkl"
        fvm_path = os.path.join(self.fvm_dir_path, fvm_filename)
        if not os.path.exists(self.fvm_dir_path):
            os.makedirs(self.fvm_dir_path)
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

        # save refinement results
        with open(self.refinement_results_file_path, "w", newline="") as f_refine:
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

        # save latest QuasiStableColoringGraph
        with open(os.path.join(self.data_dir_path, "qsc.pkl"), "wb") as f:
            pickle.dump(qsc, f)

        self._do_common_workflow(q, max_step, max_color_count, qsc)

    def _do_common_workflow(self, q, max_step, max_color_count, qsc):
        self.last_completed_step = qsc.current_step
        self.last_max_q_error = qsc.current_max_q_error
        while self.last_completed_step < max_step and self.last_max_q_error > q and len(qsc.partitions) < max_color_count:

            start_time = time.time()
            self.logger.info(f"Refining for step={self.last_completed_step + 1}...")
            n_colors, current_step, max_q_error, witness_pair_count = qsc.refine_one_step()
            self.logger.info(f"Computing feature vector for step={current_step}...")

            fv_matrix = qsc.colored_graph.generate_feature_matrix()
            elapsed_time = round(time.time() - start_time, 6)

            params = {
                "step": current_step,
                "feature_dim": fv_matrix.shape[1] - 1,
                "max_q_error": max_q_error,
                "partition_count": len(qsc.partitions),
                "witness_pair_count": witness_pair_count,
                "calculation_time_in_seconds": elapsed_time
            }

            # --- Save FV ---
            fvm_filename = f"step_{current_step}.pkl"
            fvm_path = os.path.join(self.fvm_dir_path, fvm_filename)
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
            with open(self.refinement_results_file_path, "a", newline="") as f_refine:
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
            with open(os.path.join(self.data_dir_path, "qsc.pkl"), "wb") as f:
                pickle.dump(qsc, f)

            # --- Update for the next loop iteration ---
            self.last_completed_step = current_step
            self.last_max_q_error = max_q_error

            if max_q_error == 0.0:
                self.logger.info("Reached stable coloring: q-error is 0. Ending refinement.")
                break