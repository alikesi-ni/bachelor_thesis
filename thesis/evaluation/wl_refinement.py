import os
import csv
import pickle
import time

import networkx as nx
import numpy as np

from thesis.test.test_print_system_info import log_machine_spec
from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph
from thesis.utils.logger_config import LoggerFactory


class WlRefinement:
    def __init__(self, dataset_name, disjoint_graph: nx.Graph, graph_id_label_map: dict[int, int],
                 refinement_steps=np.inf, base_dir="../evaluation-results", logging=True):
        self.dataset_name = dataset_name
        self.disjoint_graph = disjoint_graph
        self.refinement_steps = refinement_steps
        sorted_map = dict(sorted(graph_id_label_map.items()))
        self.graph_ids = np.array(list(sorted_map.keys()))
        self.graph_labels = np.array(list(sorted_map.values()))
        self.base_dir = base_dir
        self.last_completed_step = -1
        self.data_dir_path = f"{self.base_dir}/WL-{self.dataset_name}"
        if not os.path.exists(self.data_dir_path):
            os.makedirs(self.data_dir_path)
        self.fvm_dir_path = os.path.join(self.data_dir_path, "feature_vector_matrices")
        self.wl_file_path = os.path.join(self.data_dir_path, "wl.pkl")
        self.refinement_results_file_path = os.path.join(self.data_dir_path, "refinement_results.csv")
        log_file_path = os.path.join(self.data_dir_path, "evaluation_log.txt")
        log_file_exists = os.path.exists(log_file_path)
        self.logger = LoggerFactory.get_full_logger(__name__, log_file_path) if logging \
            else LoggerFactory.get_console_logger(__name__, "error")
        if log_file_exists:
            self.logger.info("====================")
            self.logger.info("CONTINUING REFINEMENT")
            self.logger.info("====================")
        log_machine_spec(self.logger)
        self.logger.info("--------------------")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info("Algorithm: WL (Weisfeiler–Leman)")
        self.logger.info("--------------------")

    def run(self):
        fvm_dir_exists = os.path.exists(self.fvm_dir_path)
        wl_file_exists = os.path.isfile(self.wl_file_path)
        refinement_results_exists = os.path.exists(self.refinement_results_file_path)

        if not fvm_dir_exists and not wl_file_exists and not refinement_results_exists:
            self._do_normal_workflow()
        else:
            self._do_load_workflow()

    def _do_normal_workflow(self):
        cg = ColoredGraph(self.disjoint_graph.copy())
        wl = WeisfeilerLemanColoringGraph(cg)

        os.makedirs(self.fvm_dir_path, exist_ok=True)
        self.logger.info(f"Computing feature vector for step=0...")
        start_time = time.time()
        fv_matrix = cg.generate_feature_matrix()
        elapsed_time = time.time() - start_time

        params = {
            "step": 0,
            "feature_dim": fv_matrix.shape[1] - 1,
            "n_colors": cg.get_num_colors(),
            "calculation_time_in_seconds": round(elapsed_time, 6)
        }

        # Save FVM
        with open(os.path.join(self.fvm_dir_path, "step_0.pkl"), "wb") as f:
            pickle.dump((fv_matrix, params), f)

        self.logger.info(
            f"Saved feature vector for step=0 "
            f"(n_colors={params['n_colors']}, feature_dim={params['feature_dim']}, "
            f"time={params['calculation_time_in_seconds']} s)"
        )

        # Write CSV
        with open(self.refinement_results_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "feature_dim", "n_colors", "calculation_time_in_seconds"])
            writer.writerow([params["step"], params["feature_dim"], params["n_colors"], params["calculation_time_in_seconds"]])

        # Save WL
        with open(self.wl_file_path, "wb") as f:
            pickle.dump(wl, f)

        self._do_common_workflow(wl)

    def _do_load_workflow(self):
        if not os.path.exists(self.fvm_dir_path):
            self.logger.error("Aborting: feature_vector_matrices folder not found.")
            return

        if not os.path.isfile(self.wl_file_path):
            self.logger.error("Aborting: wl.pkl not found.")
            return

        try:
            with open(self.wl_file_path, "rb") as f:
                wl = pickle.load(f)
            self.logger.info("Loaded WL object.")
        except Exception as e:
            self.logger.error(f"Aborting: failed to load wl.pkl ({e})")
            return

        current_step = wl.current_step
        current_fvm_path = os.path.join(self.fvm_dir_path, f"step_{current_step}.pkl")

        if not os.path.isfile(current_fvm_path):
            self.logger.error(f"Aborting: FVM for step {current_step} is missing.")
            return

        if not os.path.isfile(self.refinement_results_file_path):
            self.logger.error("Aborting: refinement_results.csv not found.")
            return

        self.logger.info(f"Step {current_step}: all files and logs seem consistent. Continuing...")
        self._do_common_workflow(wl)

    def _do_common_workflow(self, wl):
        step = wl.current_step + 1
        while step <= self.refinement_steps and not wl.is_stable:
            self.logger.info(f"Refining for step={step}...")
            start_time = time.time()
            n_colors, _ = wl.refine_one_step()
            if wl.is_stable:
                self.logger.info(f"Reached stable coloring at step={wl.current_step}.")
                break
            fv_matrix = wl.colored_graph.generate_feature_matrix()
            elapsed_time = round(time.time() - start_time, 6)

            params = {
                "step": step,
                "feature_dim": fv_matrix.shape[1] - 1,
                "n_colors": n_colors,
                "calculation_time_in_seconds": elapsed_time
            }

            with open(os.path.join(self.fvm_dir_path, f"step_{step}.pkl"), "wb") as f:
                pickle.dump((fv_matrix, params), f)

            self.logger.info(
                f"Saved feature vector for step={step} "
                f"(n_colors={n_colors}, feature_dim={params['feature_dim']}, "
                f"time={elapsed_time} s)"
            )

            with open(self.refinement_results_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([params["step"], params["feature_dim"], params["n_colors"], elapsed_time])

            with open(self.wl_file_path, "wb") as f:
                pickle.dump(wl, f)

            step += 1

        # Check whether stable
        if not wl.is_stable:
            wl.refine_one_step()
            if not wl.is_stable:
                self.logger.info(f"Stopped at max steps ({self.refinement_steps}) without reaching stability.")
            else:
                self.logger.info(f"Reached stable coloring at step={wl.current_step}.")
