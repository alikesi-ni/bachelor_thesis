import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import csr_array, hstack

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.logger_config import LoggerFactory


class ColorStats:
    def __init__(self, v, n):
        self.v = v
        self.n = n
        self.neighbor = csr_array((v, n), dtype=np.float64)
        self.upper_base = np.zeros((n, n), dtype=np.float64)
        self.lower_base = np.full((n, n), np.inf, dtype=np.float64)
        self.counts_base = np.zeros((n, n), dtype=np.uint)
        self.errors_base = np.zeros((n, n), dtype=np.float64)

    def resize(self, v, n):
        new_stats = ColorStats(v, n)
        m = self.n
        new_stats.neighbor = self.neighbor  # assumes already resized
        new_stats.upper_base[:m, :m] = self.upper_base
        new_stats.lower_base[:m, :m] = self.lower_base
        new_stats.counts_base[:m, :m] = self.counts_base
        new_stats.errors_base[:m, :m] = self.errors_base
        return new_stats


class QuasiStableColoringGraph:
    def __init__(self, colored_graph: ColoredGraph, q=0.0, n_colors=np.inf, max_steps=np.inf,
                 q_tolerance=0.0, logger=None, logging_level="error"):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.q = q
        self.n_colors = n_colors
        self.max_steps = max_steps
        self.q_tolerance = q_tolerance

        self.blocks = []
        self.current_max_q_error = np.inf
        self.previous_max_q_error = np.inf
        self.current_step = 0
        self.color_stats = None
        self.weights = None

        self.logging_level = logging_level
        self.logger = logger if logger else LoggerFactory.get_console_logger(__name__, logging_level)

        self._assert_nodes_start_from_zero()
        self.is_set_up = False
        self._set_up()

    def _assert_nodes_start_from_zero(self):
        nodes = list(self.graph.nodes)
        sorted_nodes = sorted(nodes)
        assert sorted_nodes[0] == 0, "Node indices must start at 0"
        assert sorted_nodes == list(range(len(nodes))), "Node indices must be consecutive integers starting from 0"

    def _get_partition_matrix(self) -> csr_array:
        num_nodes = sum(len(block) for block in self.blocks)
        num_blocks = len(self.blocks)
        I = np.zeros(num_nodes, dtype=int)
        J = np.zeros(num_nodes, dtype=int)
        V = np.ones(num_nodes, dtype=np.float64)
        i = 0
        for block_idx, block in enumerate(self.blocks):
            for node_id in block:
                I[i] = node_id
                J[i] = block_idx
                i += 1
        return csr_array((V, (I, J)), shape=(self.graph.number_of_nodes(), num_blocks))

    def _initialize_partition_stats(self):
        P_sparse = self._get_partition_matrix()
        self.color_stats.neighbor = self.weights.dot(P_sparse)

        m = len(self.blocks)
        for i, block in enumerate(self.blocks):
            neighbor_matrix = self.color_stats.neighbor[block, :].toarray()
            upper_deg = np.max(neighbor_matrix, axis=0).reshape(1, -1)
            lower_deg = np.min(neighbor_matrix, axis=0).reshape(1, -1)
            self.color_stats.upper_base[i, :m] = upper_deg.flatten()
            self.color_stats.lower_base[i, :m] = lower_deg.flatten()

        self.color_stats.errors_base[:m, :m] = (self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m])

    def _update_partition_stats(self, indices_to_be_updated):
        start_time = time.time()
        m = len(self.blocks)

        for i in indices_to_be_updated:
            nodes = self.blocks[i]
            degs = np.array(self.weights[:, nodes].sum(axis=1)).flatten()
            self.color_stats.neighbor[:, i] = degs
            self.color_stats.upper_base[i, :m] = np.max(self.color_stats.neighbor[nodes, :], axis=0).toarray()
            self.color_stats.lower_base[i, :m] = np.min(self.color_stats.neighbor[nodes, :], axis=0).toarray()

        for i, block in enumerate(self.blocks):
            for j in indices_to_be_updated:
                self.color_stats.upper_base[i, j] = np.max(self.color_stats.neighbor[block, j])
                self.color_stats.lower_base[i, j] = np.min(self.color_stats.neighbor[block, j])

        self.color_stats.errors_base[:m, :m] = (self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m])

        for block in self.blocks:
            color_id = self.colored_graph.next_color_id
            for node in block:
                self.graph.nodes[node]["color-stack"].append(color_id)
            self.colored_graph.next_color_id += 1

        self.colored_graph.color_stack_height += 1
        elapsed = (time.time() - start_time)
        self.logger.info(
            f"[step {self.current_step}] update_stats_partitions completed in {elapsed:.2f} s"
        )

    def refine(self):
        self._set_up()

        while (
                len(self.blocks) < self.n_colors and
                self.current_step < self.max_steps and
                self.current_max_q_error > 0.0
        ):
            if self.current_max_q_error == 0.0:
                self.logger.info(
                    f"[step {self.current_step}] Reached stable state: no changes in color count."
                )
                break

            self.refine_one_step()

        self.logger.info(
            f"QSC REFINEMENT DONE: steps: {self.current_step}, color_count: {len(self.blocks)}, "
            f"max_q_error: {self.current_max_q_error}"
        )

        return len(self.blocks), self.current_step, self.current_max_q_error

    def refine_one_step(self):
        if not self.is_set_up:
            self._set_up()

        if self.current_max_q_error == 0.0:
            self.logger.warning(f"Q-error already 0.0 - no further refinement possible")
            return len(self.blocks), self.current_step, self.current_max_q_error

        self.previous_max_q_error = self.current_max_q_error
        self.current_step += 1

        m = len(self.blocks)
        errors = self.color_stats.errors_base[:m, :m]
        threshold = self.previous_max_q_error * (1 - self.q_tolerance)
        witness_pairs = np.argwhere(errors >= threshold)

        self.logger.info(
            f"[step {self.current_step}] Found {len(witness_pairs)} witness pairs with q-error >= {threshold:.1f} "
            f"(max={self.previous_max_q_error:.1f}, tolerance={self.q_tolerance:.2f})"
        )

        proposed_splits = self._propose_splits(witness_pairs)

        indices_with_node_changes = self._execute_splits(proposed_splits)

        self._resize_if_necessary()

        self._update_partition_stats(indices_with_node_changes)

        m = len(self.blocks)
        errors = self.color_stats.errors_base[:m, :m]
        self.current_max_q_error = np.max(errors)

        if self.current_max_q_error > self.previous_max_q_error:
            self.logger.warning(
                f"[step {self.current_step}] max q-error JUMPED from {self.previous_max_q_error:.1f} to {self.current_max_q_error:.1f}")

        self.logger.info(f"[step {self.current_step}] DONE: color_count: {len(self.blocks)}; max_q_error: {self.current_max_q_error}")
        self.logger.info("--------------------")

        return len(self.blocks), self.current_step, self.current_max_q_error, len(witness_pairs)

    def _resize_if_necessary(self):
        m = len(self.blocks)
        rows, cols = self.color_stats.neighbor.shape
        if self.color_stats.n < m:
            self.logger.info(
                f"[step {self.current_step}] Limit of {self.color_stats.n} reached: Updated color stats size")
            new_n = 1 if m == 0 else 2 ** (m - 1).bit_length()
            self.color_stats = self.color_stats.resize(self.color_stats.v, new_n)
        if cols < m:
            extra_cols = csr_array((rows, m - cols), dtype=np.float64)
            self.color_stats.neighbor = hstack([self.color_stats.neighbor, extra_cols])

    def _execute_splits(self, proposed_splits):
        changed_indices = set()
        for witness_i, split_groups in proposed_splits.items():
            first_group = split_groups.pop(0)
            self.blocks[witness_i] = first_group
            changed_indices.add(witness_i)
            for group in split_groups:
                if group:
                    self.blocks.append(group)
                    changed_indices.add(len(self.blocks) - 1)
        return changed_indices

    def _propose_splits(self, witness_pairs):
        split_conditions = defaultdict(list)
        for witness_i, witness_j in witness_pairs:
            degrees = self.color_stats.neighbor[self.blocks[witness_i], witness_j].toarray().flatten()
            split_deg = np.mean(degrees)
            split_conditions[witness_i].append((witness_j, split_deg))

        proposed_splits = {}
        for witness_i, conditions in split_conditions.items():
            node_buckets = defaultdict(list)
            nodes = self.blocks[witness_i]
            for node in nodes:
                signature = []
                for witness_j, split_deg in conditions:
                    degree = self.color_stats.neighbor[node, witness_j]
                    signature.append(int(degree > split_deg))
                signature = tuple(signature)
                node_buckets[signature].append(node)

            proposed_splits[witness_i] = list(node_buckets.values())
        return proposed_splits

    def _set_up(self):
        color_groups = defaultdict(list)
        for node, color_stack in nx.get_node_attributes(self.graph, "color-stack").items():
            color = color_stack[-1]
            color_groups[color].append(node)

        self.blocks = list(color_groups.values())

        self.weights = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes), dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), max(len(self.blocks), int(min(self.n_colors, 128))))
        self._initialize_partition_stats()
        m = len(self.blocks)
        errors = self.color_stats.errors_base[:m, :m]
        self.current_max_q_error = np.max(errors)
        self.is_set_up = True
