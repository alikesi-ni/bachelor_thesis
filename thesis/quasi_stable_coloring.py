import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, csr_array


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
        new_stats.neighbor = self.neighbor  # assuming already resized outside
        new_stats.upper_base[:m, :m] = self.upper_base
        new_stats.lower_base[:m, :m] = self.lower_base
        new_stats.counts_base[:m, :m] = self.counts_base
        new_stats.errors_base[:m, :m] = self.errors_base
        return new_stats


class QuasiStableColoringGraph:
    def __init__(self, graph: nx.Graph, q=1.0, n_colors=np.inf, weighting=False):
        self.graph = graph
        self.q = q
        self.n_colors = n_colors
        self.weighting = weighting

        self.partitions = []
        self.color_stats = None

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
        return csr_array((V, (I, J)), shape=(max(I) + 1, num_partitions))

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

    def refine(self):
        self.partitions = [list(self.graph.nodes)]
        self.weights = nx.adjacency_matrix(self.graph, dtype=np.float64)
        self.color_stats = ColorStats(len(self.graph), min(self.n_colors, 128))
        self.update_stats()

        while len(self.partitions) < self.n_colors:
            if len(self.partitions) == self.color_stats.n:
                self.color_stats = self.color_stats.resize(self.color_stats.v, self.color_stats.n * 2)

            witness_i, witness_j, split_deg, q_error = self.pick_witness()

            if q_error <= self.q:
                break

            self.split_color(witness_i, witness_j, split_deg)
            self.update_stats_split(witness_i, len(self.partitions) - 1)

        return self.partitions



# Create an undirected graph
g = nx.Graph()

# Add nodes
g.add_nodes_from(range(8))

# Add edges based on the structure provided
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5), (4, 7),
    (5, 6)
]

# Add edges to the graph
g.add_edges_from(edges)

qsc = QuasiStableColoringGraph(g, 1.0)
qsc.refine()

