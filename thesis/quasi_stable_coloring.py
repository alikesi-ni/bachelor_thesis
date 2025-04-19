from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.sparse import csr_array

from thesis.colored_graph.colored_graph import ColoredGraph


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
    def __init__(self, colored_graph: ColoredGraph, q=0.0, n_colors=np.inf, weighting=False):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.q = q
        self.n_colors = n_colors
        self.weighting = weighting

        self.partitions = []
        self.color_stats = None
        self.weights = None

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
        witness = np.unravel_index(np.argmax(errors), errors.shape)
        q_error = errors[witness]
        witness_i, witness_j = witness[0], witness[1]
        split_deg = np.mean(self.color_stats.neighbor[self.partitions[witness_i], witness_j].toarray())
        return witness_i, witness_j, split_deg, q_error

    def split_color(self, witness_i, witness_j, threshold):
        retained, ejected = [], []
        for node_id in self.partitions[witness_i]:
            if self.color_stats.neighbor[node_id, witness_j] > threshold:
                ejected.append(node_id)
            else:
                retained.append(node_id)

        assert retained and ejected

        self.partitions[witness_i] = retained
        self.partitions.append(ejected)

    def update_stats_split(self, old, new):
        old_nodes = self.partitions[old]
        new_nodes = self.partitions[new]

        rows, cols = self.color_stats.neighbor.shape
        self.color_stats.neighbor.resize((rows, cols + 1))

        old_degs = np.array(self.weights[:, old_nodes].sum(axis=1)).flatten()
        new_degs = np.array(self.weights[:, new_nodes].sum(axis=1)).flatten()
        self.color_stats.neighbor[:, old] = old_degs
        self.color_stats.neighbor[:, new] = new_degs

        m = len(self.partitions)
        self.color_stats.upper_base[old, :m] = np.max(self.color_stats.neighbor[old_nodes, :], axis=0).toarray()
        self.color_stats.lower_base[old, :m] = np.min(self.color_stats.neighbor[old_nodes, :], axis=0).toarray()

        self.color_stats.upper_base[new, :m] = np.max(self.color_stats.neighbor[new_nodes, :], axis=0).toarray()
        self.color_stats.lower_base[new, :m] = np.min(self.color_stats.neighbor[new_nodes, :], axis=0).toarray()

        for i, partition in enumerate(self.partitions):
            self.color_stats.upper_base[i, old] = np.max(self.color_stats.neighbor[partition, old])
            self.color_stats.lower_base[i, old] = np.min(self.color_stats.neighbor[partition, old])
            self.color_stats.upper_base[i, new] = np.max(self.color_stats.neighbor[partition, new])
            self.color_stats.lower_base[i, new] = np.min(self.color_stats.neighbor[partition, new])

        if self.weighting:
            sizes = np.array([len(p) for p in self.partitions]).reshape(-1, 1)
            self.color_stats.errors_base[:m, :m] = (
                self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]
            ) * sizes
        else:
            self.color_stats.errors_base[:m, :m] = (
                self.color_stats.upper_base[:m, :m] - self.color_stats.lower_base[:m, :m]
            )

        for partition in self.partitions:
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

        self.weights = nx.adjacency_matrix(self.graph, dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), max(len(self.partitions), int(min(self.n_colors, 128))))
        self.update_stats()

        q_error_before = np.inf
        while len(self.partitions) < self.n_colors:
            if len(self.partitions) == self.color_stats.n:
                print(f"Limit of {self.color_stats.n} reached: Updated color stats size")
                self.color_stats = self.color_stats.resize(self.color_stats.v, self.color_stats.n * 2)

            witness_i, witness_j, split_deg, q_error = self.pick_witness()
            if q_error > q_error_before:
                print(f"Q-error JUMPED! {q_error_before:.3f} → {q_error:.3f}")
            q_error_before = q_error
            if q_error <= self.q:
                break

            self.split_color(witness_i, witness_j, split_deg)
            self.update_stats_split(witness_i, len(self.partitions) - 1)
            verbose and print(f"Number of partitions: {len(self.partitions)}; Q-error: {q_error}")

        print(f"QSC DONE: color count: {len(self.partitions)}, max q-error={q_error}")

        return self.partitions
