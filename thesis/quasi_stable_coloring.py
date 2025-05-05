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
    def __init__(self, colored_graph: ColoredGraph, q=0.0, n_colors=np.inf, refinement_steps=np.inf, weighting=False, q_tolerance=0.0, logger=None, logging_level="error"):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.q = q
        self.n_colors = n_colors
        self.refinement_steps = refinement_steps
        self.weighting = weighting
        self.q_tolerance = q_tolerance

        self.partitions = []
        self.q_error = np.inf
        self.q_error_before = np.inf
        self.refinement_step = 0
        self.color_stats = None
        self.weights = None

        self.logging_level = logging_level
        self.logger = logger if logger else LoggerFactory.get_console_logger(__name__, logging_level)

        self.__assert_nodes_start_from_zero()
        self.is_set_up = False

    def __assert_nodes_start_from_zero(self):
        nodes = list(self.graph.nodes)
        sorted_nodes = sorted(nodes)
        assert sorted_nodes[0] == 0, "Node indices must start at 0"
        assert sorted_nodes == list(range(len(nodes))), "Node indices must be consecutive integers starting from 0"

    def partition_matrix(self):
        num_nodes = sum(len(partition) for partition in self.partitions)
        num_partitions = len(self.partitions)
        I = np.zeros(num_nodes, dtype=int)
        J = np.zeros(num_nodes, dtype=int)
        V = np.ones(num_nodes, dtype=np.float64)
        i = 0
        for partition_idx, partition in enumerate(self.partitions):
            for node_id in partition:
                I[i] = node_id
                J[i] = partition_idx
                i += 1
        return csr_array((V, (I, J)), shape=(self.graph.number_of_nodes(), num_partitions))

    def update_stats(self):
        P_sparse = self.partition_matrix()
        self.color_stats.neighbor = self.weights.dot(P_sparse)

        m = len(self.partitions)
        for i, partition in enumerate(self.partitions):
            neighbor_matrix = self.color_stats.neighbor[partition, :].toarray()
            upper_deg = np.max(neighbor_matrix, axis=0).reshape(1, -1)
            lower_deg = np.min(neighbor_matrix, axis=0).reshape(1, -1)
            self.color_stats.upper_base[i, :m] = upper_deg.flatten()
            self.color_stats.lower_base[i, :m] = lower_deg.flatten()

        if self.weighting:
            self.color_stats.errors_base[:m, :m] = (
                self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]
            ) * np.array([len(P_i) for P_i in self.partitions]).reshape(-1, 1)
        else:
            self.color_stats.errors_base[:m, :m] = (
                self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]
            )

    def update_stats_partitions(self, partitions_to_be_updated):
        m = len(self.partitions)

        for partition in partitions_to_be_updated:
            nodes = self.partitions[partition]
            degs = np.array(self.weights[:, nodes].sum(axis=1)).flatten()
            self.color_stats.neighbor[:, partition] = degs
            self.color_stats.upper_base[partition, :m] = np.max(self.color_stats.neighbor[nodes, :], axis=0).toarray()
            self.color_stats.lower_base[partition, :m] = np.min(self.color_stats.neighbor[nodes, :], axis=0).toarray()

        for i, partition in enumerate(self.partitions):
            for partition_to_be_updated in partitions_to_be_updated:
                self.color_stats.upper_base[i, partition_to_be_updated] = np.max(self.color_stats.neighbor[partition, partition_to_be_updated])
                self.color_stats.lower_base[i, partition_to_be_updated] = np.min(self.color_stats.neighbor[partition, partition_to_be_updated])

        if self.weighting:
            sizes = np.array([len(p) for p in self.partitions]).reshape(-1, 1)
            self.color_stats.errors_base[:m, :m] = (self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]) * sizes
        else:
            self.color_stats.errors_base[:m, :m] = (self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m])

        for partition in partitions_to_be_updated:
            partition_nodes = self.partitions[partition]
            color_id = self.colored_graph.next_color_id
            for node in partition_nodes:
                self.graph.nodes[node]["color-stack"].append(color_id)
            self.colored_graph.next_color_id += 1

        self.colored_graph.color_stack_height += 1

    def refine(self):
        self.partitions = []

        color_groups = defaultdict(list)
        for node, color_stack in nx.get_node_attributes(self.graph, "color-stack").items():
            color = color_stack[-1]
            color_groups[color].append(node)

        self.partitions = list(color_groups.values())

        self.weights = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes), dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), max(len(self.partitions), int(min(self.n_colors, 128))))
        self.update_stats()

        while len(self.partitions) < self.n_colors and self.refinement_step < self.refinement_steps:
            self.q_error_before = self.q_error
            self.refinement_step += 1

            m = len(self.partitions)
            errors = self.color_stats.errors_base[:m, :m]
            self.q_error = np.max(errors)

            if self.q_error > self.q_error_before:
                self.logger.warning(f"[refinement_step {self.refinement_step}] Q-ERROR JUMPED: from {self.q_error_before:.1f} to {self.q_error:.1f}")
                self.logger.warning("--------------------")

            threshold = self.q_error * (1 - self.q_tolerance)
            witness_pairs = np.argwhere(errors >= threshold)

            self.logger.info(
                f"[refinement_step {self.refinement_step}] Found {len(witness_pairs)} witness pairs with q-error >= {threshold:.1f} "
                f"(max={self.q_error:.1f}, tolerance={self.q_tolerance:.2f})"
            )

            if self.q_error <= self.q:
                break

            split_conditions = defaultdict(list)
            for witness_i, witness_j in witness_pairs:
                degrees = self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray().flatten()
                split_deg = np.mean(degrees)
                split_conditions[witness_i].append((witness_j, split_deg))

            proposed_splits = {}
            for witness_i, conditions in split_conditions.items():
                node_buckets = defaultdict(list)
                nodes = self.partitions[witness_i]
                for node in nodes:
                    signature = []
                    for witness_j, split_deg in conditions:
                        degree = self.color_stats.neighbor[node, witness_j]
                        signature.append(int(degree > split_deg))
                    signature = tuple(signature)
                    node_buckets[signature].append(node)

                proposed_splits[witness_i] = list(node_buckets.values())

            changed_indices = set()
            for witness_i, split_groups in proposed_splits.items():
                first_group = split_groups.pop(0)
                self.partitions[witness_i] = first_group
                changed_indices.add(witness_i)
                for group in split_groups:
                    if group:
                        self.partitions.append(group)
                        changed_indices.add(len(self.partitions) - 1)

            m = len(self.partitions)
            rows, cols = self.color_stats.neighbor.shape
            if self.color_stats.n < m:
                self.logger.info(f"[refinement_step {self.refinement_step}] Limit of {self.color_stats.n} reached: Updated color stats size")
                new_n = 1 if m == 0 else 2**(m - 1).bit_length()
                self.color_stats = self.color_stats.resize(self.color_stats.v, new_n)
            if cols < m:
                extra_cols = csr_array((rows, m - cols), dtype=np.float64)
                self.color_stats.neighbor = hstack([self.color_stats.neighbor, extra_cols])

            self.update_stats_partitions(changed_indices)

            self.logger.info(f"[refinement_step {self.refinement_step}] color count: {len(self.partitions)}; q-error: {self.q_error}")
            self.logger.info("--------------------")

        self.logger.info(f"QSC DONE: color count: {len(self.partitions)}, max q-error={self.q_error}, refinement_steps={self.refinement_step}")

        return len(self.partitions), self.refinement_step, self.q_error

    def refine_one_step(self):
        if not self.is_set_up:
            self.__set_up()

        if self.q_error == 0.0:
            self.logger.warning(f"Q-error already 0.0 - no further refinement possible")
            return len(self.partitions), self.refinement_step, self.q_error

        self.q_error_before = self.q_error
        self.refinement_step += 1

        m = len(self.partitions)
        errors = self.color_stats.errors_base[:m, :m]
        self.q_error = np.max(errors)

        if self.q_error > self.q_error_before:
            self.logger.warning(
                f"[refinement_step {self.refinement_step}] Q-ERROR JUMPED: from {self.q_error_before:.1f} to {self.q_error:.1f}")
            self.logger.warning("--------------------")

        threshold = self.q_error * (1 - self.q_tolerance)
        witness_pairs = np.argwhere(errors >= threshold)

        self.logger.info(
            f"[refinement_step {self.refinement_step}] Found {len(witness_pairs)} witness pairs with q-error >= {threshold:.1f} "
            f"(max={self.q_error:.1f}, tolerance={self.q_tolerance:.2f})"
        )

        split_conditions = defaultdict(list)
        for witness_i, witness_j in witness_pairs:
            degrees = self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray().flatten()
            split_deg = np.mean(degrees)
            split_conditions[witness_i].append((witness_j, split_deg))

        proposed_splits = {}
        for witness_i, conditions in split_conditions.items():
            node_buckets = defaultdict(list)
            nodes = self.partitions[witness_i]
            for node in nodes:
                signature = []
                for witness_j, split_deg in conditions:
                    degree = self.color_stats.neighbor[node, witness_j]
                    signature.append(int(degree > split_deg))
                signature = tuple(signature)
                node_buckets[signature].append(node)

            proposed_splits[witness_i] = list(node_buckets.values())

        changed_indices = set()
        for witness_i, split_groups in proposed_splits.items():
            first_group = split_groups.pop(0)
            self.partitions[witness_i] = first_group
            changed_indices.add(witness_i)
            for group in split_groups:
                if group:
                    self.partitions.append(group)
                    changed_indices.add(len(self.partitions) - 1)

        m = len(self.partitions)
        rows, cols = self.color_stats.neighbor.shape
        if self.color_stats.n < m:
            self.logger.info(
                f"[refinement_step {self.refinement_step}] Limit of {self.color_stats.n} reached: Updated color stats size")
            new_n = 1 if m == 0 else 2 ** (m - 1).bit_length()
            self.color_stats = self.color_stats.resize(self.color_stats.v, new_n)
        if cols < m:
            extra_cols = csr_array((rows, m - cols), dtype=np.float64)
            self.color_stats.neighbor = hstack([self.color_stats.neighbor, extra_cols])

        self.update_stats_partitions(changed_indices)

        self.logger.info(f"[refinement_step {self.refinement_step}] color count: {len(self.partitions)}; q-error: {self.q_error}")
        self.logger.info("--------------------")

        return len(self.partitions), self.refinement_step, self.q_error

    def __set_up(self):
        color_groups = defaultdict(list)
        for node, color_stack in nx.get_node_attributes(self.graph, "color-stack").items():
            color = color_stack[-1]
            color_groups[color].append(node)

        self.partitions = list(color_groups.values())

        self.weights = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes), dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), max(len(self.partitions), int(min(self.n_colors, 128))))
        self.update_stats()
        self.is_set_up = True
