import logging
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.sparse import csr_array

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.logger_config import setup_logger


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
    def __init__(self, colored_graph: ColoredGraph, q=0.0, n_colors=np.inf, weighting=False, verbose=False):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.q = q
        self.n_colors = n_colors
        self.weighting = weighting

        self.partitions = []
        self.color_stats = None
        self.weights = None
        self.verbose = verbose

        self.logger = setup_logger(self.__class__.__name__) if logging else None

        self.__assert_nodes_start_from_zero()

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

    def pick_witness(self):
        m = len(self.partitions)
        errors = self.color_stats.errors_base[:m, :m]
        max_error = np.max(errors)

        # Find all (i,j) pairs with the same max error
        max_error_indices = np.argwhere(errors == max_error)
        num_max = len(max_error_indices)

        if num_max > 1:
            print(f"Number of witness candidates with max error ({max_error}): {num_max}")

        # Pick just one as before
        witness_i, witness_j = max_error_indices[0]
        q_error = max_error

        if self.verbose:
            print(f"Witness i: {self.partitions[witness_i]}")
            print(f"Witness j: {self.partitions[witness_j]}")
            print(f"Q-error: {q_error}")

        split_deg = np.mean(self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray())
        return witness_i, witness_j, split_deg, q_error

    def split_color(self, witness_i, witness_j, threshold):
        retained, ejected = [], []
        for node_id in self.partitions[witness_i]:
            if self.color_stats.neighbor[node_id, witness_j] > threshold:
                ejected.append(node_id)
            else:
                retained.append(node_id)

        if self.verbose:
            print(f"{self.partitions[witness_i]} split into {retained} and {ejected}")

        assert retained and ejected

        self.partitions[witness_i] = retained
        self.partitions.append(ejected)

    def update_stats_partitions(self, partition_indices):
        """
        Update the statistics (upper_base, lower_base, errors_base, neighbor matrix, color stacks)
        for the specified partitions.

        partition_indices : list of partition indices to update
        """
        m = len(self.partitions)

        # Resize neighbor matrix if needed
        rows, cols = self.color_stats.neighbor.shape
        if cols < m:
            self.color_stats.neighbor.resize((rows, m))

        # 1️⃣ Update neighbor columns for changed partitions
        for idx in partition_indices:
            nodes = self.partitions[idx]
            degs = np.array(self.weights[:, nodes].sum(axis=1)).flatten()
            self.color_stats.neighbor[:, idx] = degs

        # 2️⃣ Update upper and lower bounds for changed partitions (row-wise)
        for idx in partition_indices:
            nodes = self.partitions[idx]
            self.color_stats.upper_base[idx, :m] = np.max(self.color_stats.neighbor[nodes, :], axis=0).toarray()
            self.color_stats.lower_base[idx, :m] = np.min(self.color_stats.neighbor[nodes, :], axis=0).toarray()

        # 3️⃣ Update upper and lower bounds for all partitions relative to changed ones (column-wise)
        for i, partition in enumerate(self.partitions):
            for idx in partition_indices:
                self.color_stats.upper_base[i, idx] = np.max(self.color_stats.neighbor[partition, idx])
                self.color_stats.lower_base[i, idx] = np.min(self.color_stats.neighbor[partition, idx])

        # 4️⃣ Update the error base
        if self.weighting:
            sizes = np.array([len(p) for p in self.partitions]).reshape(-1, 1)
            self.color_stats.errors_base[:m, :m] = (
                                                           self.color_stats.upper_base[:m,
                                                           :m] - self.color_stats.lower_base[:m, :m]
                                                   ) * sizes
        else:
            self.color_stats.errors_base[:m, :m] = (
                    self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]
            )

        # 5️⃣ Update color stacks for the changed partitions
        for idx in partition_indices:
            partition = self.partitions[idx]
            color_id = self.colored_graph.next_color_id
            for node in partition:
                self.graph.nodes[node]["color-stack"].append(color_id)
            self.colored_graph.next_color_id += 1

        self.colored_graph.color_stack_height += 1

    def refine(self, verbose=False):
        self.partitions = []

        # initialize partitions by grouping nodes by their initial color
        color_groups = defaultdict(list)
        for node, color_stack in nx.get_node_attributes(self.graph, "color-stack").items():
            color = color_stack[-1]
            color_groups[color].append(node)

        self.partitions = list(color_groups.values())

        self.weights = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes), dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), max(len(self.partitions), int(min(self.n_colors, 128))))
        self.update_stats()

        q_error_before = np.inf
        q_error = np.inf
        iteration = 0
        while len(self.partitions) < self.n_colors:
            iteration+=1
            if len(self.partitions) == self.color_stats.n:
                print(f"[iteration {iteration}] Limit of {self.color_stats.n} reached: Updated color stats size")
                self.color_stats = self.color_stats.resize(self.color_stats.v, self.color_stats.n * 2)

            m = len(self.partitions)
            errors = self.color_stats.errors_base[:m, :m]
            max_error = np.max(errors)

            witness_indices = np.argwhere(errors == max_error)
            if self.verbose:
                print(f"[iteration {iteration}] Found {len(witness_indices)} witness pairs with q-error = {max_error}")

            q_error = max_error

            if q_error > q_error_before:
                self.logger.info(f"[iteration {iteration}] Q-error JUMPED! {q_error_before:.3f} # {q_error:.3f}")
            q_error_before = q_error

            if q_error <= self.q:
                break

            # Collect for each witness_i the set of neighbor degrees toward different witness_j
            split_data = defaultdict(list)

            for witness_i, witness_j in witness_indices:
                # Store neighbor degrees to later compute a combined split_deg
                neighbor_degrees = self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray().flatten()
                split_data[witness_i].append(neighbor_degrees)

            # Prepare all splits
            splits = {}

            for witness_i, neighbor_degree_lists in split_data.items():
                # Concatenate all neighbor degrees across different witness_j's
                all_neighbor_degrees = np.concatenate(neighbor_degree_lists)

                # Compute split degree — you could also use median instead of mean if more robust
                split_deg = np.mean(all_neighbor_degrees)

                splits[witness_i] = split_deg

            for witness_i, witness_j in witness_indices:

                split_deg = np.mean(
                    self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray()
                )

                self.split_color(witness_i, witness_j, split_deg)
                self.update_stats_partitions([witness_i, len(self.partitions) - 1])


            if self.verbose:
                print(f"[iteration {iteration}] Number of partitions: {len(self.partitions)}; Q-error: {q_error}")
                print("--------------------")
            if (len(self.partitions) % 10 == 0):
                self.logger.info(f"[iteration {iteration}] Number of partitions: {len(self.partitions)}; Q-error: {q_error}")

        self.logger.info(f"QSC DONE: color count: {len(self.partitions)}, max q-error={q_error}, iterations={iteration}")

        return self.partitions
