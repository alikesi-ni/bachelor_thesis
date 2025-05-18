import os
import csv
import pickle
import time
import networkx as nx
import numpy as np
import pandas as pd

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.gwl_coloring import GWLColoringGraph
from thesis.utils.logger_config import LoggerFactory
from tests.test_print_system_info import log_machine_spec


class GwlRefinement:
    def __init__(
        self,
        dataset_name: str,
        graph_id_label_map: dict[int, int],
        disjoint_graph: nx.Graph,
        n_clusters: int,
        cluster_init: str = "forgy",
        refinement_steps = 1,
        base_dir: str = "../evaluation-results",
        logging: bool = True
    ):
        self.dataset_name = dataset_name
        self.disjoint_graph = disjoint_graph
        self.n_clusters = n_clusters
        self.cluster_init = cluster_init
        self.refinement_steps = refinement_steps

        sorted_map = dict(sorted(graph_id_label_map.items()))
        self.graph_ids = np.array(list(sorted_map.keys()))
        self.graph_labels = np.array(list(sorted_map.values()))

        self.base_dir = base_dir
        self.data_dir_path = os.path.join(self.base_dir, f"GWL-{self.dataset_name}", cluster_init, f"k__{n_clusters}")
        self.fvm_dir_path = os.path.join(self.data_dir_path, "feature_vector_matrices")
        self.gwl_file_path = os.path.join(self.data_dir_path, "gwl.pkl")
        self.refinement_results_file_path = os.path.join(self.data_dir_path, "refinement_results.csv")
        self.log_path = os.path.join(self.data_dir_path, "evaluation_log.txt")

        os.makedirs(self.fvm_dir_path, exist_ok=True)

        self.logger = LoggerFactory.get_full_logger(__name__, self.log_path) if logging \
            else LoggerFactory.get_console_logger(__name__, "error")

        log_machine_spec(self.logger)
        self.logger.info("--------------------")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info("Algorithm: GWL (Gradual Weisfeiler Leman)")
        self.logger.info(f"Clusters: {n_clusters}")
        self.logger.info(f"Init method: {cluster_init}")
        self.logger.info(f"Refinement steps: {refinement_steps}")
        self.logger.info("--------------------")

    def run(self):
        if not os.path.exists(self.gwl_file_path) or \
           not os.path.exists(self.refinement_results_file_path):
            self._do_normal_workflow()
        else:
            self._do_load_workflow()

    def _do_normal_workflow(self):
        cg = ColoredGraph(self.disjoint_graph.copy())
        gwl = GWLColoringGraph(
            cg,
            refinement_steps=self.refinement_steps,
            n_cluster=self.n_clusters,
            cluster_initialization_method=self.cluster_init
        )

        self.logger.info("Starting fresh GWL refinement...")
        self._do_common_workflow(gwl, start_step=0)

    def _do_load_workflow(self):
        self.logger.info("Attempting to resume previous GWL refinement...")

        try:
            with open(self.gwl_file_path, "rb") as f:
                gwl = pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load gwl.pkl: {e}")
            return

        if not os.path.exists(self.refinement_results_file_path):
            self.logger.error("Missing refinement_results.csv")
            return

        try:
            df = pd.read_csv(self.refinement_results_file_path)
            completed_steps = df["step"].astype(int).tolist()
            start_step = max(completed_steps) + 1 if completed_steps else 0
        except Exception as e:
            self.logger.error(f"Failed to parse refinement_results.csv: {e}")
            return

        self.logger.info(f"Resuming from step {start_step}...")
        self._do_common_workflow(gwl, start_step=start_step)

    def _do_common_workflow(self, gwl: GWLColoringGraph, start_step: int):
        cg = gwl.colored_graph

        write_header = not os.path.exists(self.refinement_results_file_path)
        with open(self.refinement_results_file_path, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            if write_header:
                writer.writerow(["step", "feature_dim", "n_colors", "calculation_time_in_seconds"])

            for step in range(start_step, int(self.refinement_steps) + 1):
                self.logger.info(f"Refining step={step}...")

                start_time = time.time()
                if step > 0:
                    gwl.refine_one_step()

                fv_matrix = cg.generate_feature_matrix()
                elapsed_time = round(time.time() - start_time, 6)

                feature_dim = fv_matrix.shape[1] - 1
                n_colors = cg.get_num_colors()

                # Save FVM
                fvm_path = os.path.join(self.fvm_dir_path, f"step_{step}.pkl")
                with open(fvm_path, "wb") as f:
                    pickle.dump((fv_matrix, {
                        "step": step,
                        "feature_dim": feature_dim,
                        "n_colors": n_colors,
                        "calculation_time_in_seconds": elapsed_time
                    }), f)

                self.logger.info(
                    f"Saved FVM: step={step}, dim={feature_dim}, colors={n_colors}, time={elapsed_time:.2f}s"
                )

                # Save row to CSV
                writer.writerow([step, feature_dim, n_colors, elapsed_time])
                f_csv.flush()

                # Save GWL object
                with open(self.gwl_file_path, "wb") as f:
                    pickle.dump(gwl, f)

        self.logger.info("GWL refinement complete.")
