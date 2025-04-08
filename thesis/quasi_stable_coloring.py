import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, hstack, csr_array


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
        new_stats.neighbor = self.neighbor # no need to resize - is already dynamically resized in code
        new_stats.upper_base[:m, :m] = self.upper_base
        new_stats.lower_base[:m, :m] = self.lower_base
        new_stats.counts_base[:m, :m] = self.counts_base
        new_stats.errors_base[:m, :m] = self.errors_base
        return new_stats



def partition_matrix(partitions):
    num_nodes = sum(len(partition) for partition in partitions)
    num_partitions = len(partitions)
    I = np.zeros(num_nodes, dtype=int)
    J = np.zeros(num_nodes, dtype=int)
    V = np.ones(num_nodes, dtype=np.float64)
    i = 0
    for partition_idx, partition in enumerate(partitions):
        for node_id in partition:
            I[i] = node_id
            J[i] = partition_idx
            i += 1
    return csr_array((V, (I, J)), shape=(num_nodes, num_partitions))


def update_stats(stats, weights, partitions, weighting=False):
    P_sparse = partition_matrix(partitions)
    stats.neighbor = weights.dot(P_sparse)

    m = len(partitions)  # Number of partitions

    # Iterate through each partition
    for i, partition in enumerate(partitions):
        # Extract the relevant rows from the neighbor matrix
        neighbor_matrix = stats.neighbor[partition, :].toarray()

        # Calculate the upper and lower degree (maximum and minimum per column)
        upper_deg = np.max(neighbor_matrix, axis=0).reshape(1, -1)
        lower_deg = np.min(neighbor_matrix, axis=0).reshape(1, -1)

        # Update the base matrices in place (simulating Julia's @view)
        stats.upper_base[i, :m] = upper_deg.flatten()
        stats.lower_base[i, :m] = lower_deg.flatten()

    if weighting:
        stats.errors_base[:m, :m] = (stats.upper_base[:m, :m] - stats.lower_base[:m, :m]) * np.array([len(P_i) for P_i in partitions]).reshape(-1, 1)
    else:
        stats.errors_base[:m, :m] = stats.upper_base[:m, :m] - stats.lower_base[:m, :m]


def split_color(P, stats, witness_i, witness_j, threshold):
    retained = []
    ejected = []
    for node_id in P[witness_i]:
        if stats.neighbor[node_id, witness_j] > threshold:
            ejected.append(node_id)
        else:
            retained.append(node_id)
    P[witness_i] = retained
    P.append(ejected)


def pick_witness(P, stats):
    m = len(P)
    upper_deg = stats.upper_base[:m, :m]
    lower_deg = stats.lower_base[:m, :m]
    errors = stats.errors_base[:m, :m]

    witness = np.unravel_index(np.argmax(errors), errors.shape)
    q_error = errors[witness]
    witness_i, witness_j = witness[0], witness[1]
    split_deg = np.mean(stats.neighbor[P[witness_i], witness_j].toarray())

    return witness_i, witness_j, split_deg, q_error

def update_stats_split(stats, weights, partitions, old, new, weighting=False):
    # Extract old and new nodes from partitions
    old_nodes = partitions[old]
    new_nodes = partitions[new]

    rows, cols = stats.neighbor.shape
    stats.neighbor.resize(rows, cols + 1)

    # Update columns for old and new colors
    old_degs = np.array(weights[:, old_nodes].sum(axis=1)).flatten()
    new_degs = np.array(weights[:, new_nodes].sum(axis=1)).flatten()
    stats.neighbor[:, old] = old_degs
    stats.neighbor[:, new] = new_degs

    m = len(partitions)  # Number of partitions

    stats.upper_base[old, :m] = np.max(stats.neighbor[partitions[old], :], axis=0).toarray()
    stats.lower_base[old, :m] = np.min(stats.neighbor[partitions[old], :], axis=0).toarray()

    stats.upper_base[new, :m] = np.max(stats.neighbor[partitions[new], :], axis=0).toarray()
    stats.lower_base[new, :m] = np.min(stats.neighbor[partitions[new], :], axis=0).toarray()

    # Iterate through each partition
    for i, partition in enumerate(partitions):
        stats.upper_base[i, old] = np.max(stats.neighbor[partition, old])
        stats.lower_base[i, old] = np.min(stats.neighbor[partition, old])

        stats.upper_base[i, new] = np.max(stats.neighbor[partition, new])
        stats.lower_base[i, new] = np.min(stats.neighbor[partition, new])

    # Update errors
    if weighting:
        sizes = np.array([len(part) for part in partitions]).reshape(-1, 1)
        stats.errors_base[:m, :m] = (stats.upper_base[:m, :m] - stats.lower_base[:m, :m]) * sizes
    else:
        stats.errors_base[:m, :m] = stats.upper_base[:m, :m] - stats.lower_base[:m, :m]

    # Ensure no NaN or Inf values are present
    assert np.all(np.isfinite(stats.errors_base)), "Error matrix contains NaN or Inf"


def q_color(graph: nx.Graph, q=0.0, n_colors=np.inf, warm_start=None, weighting=False):
    vertices = set(graph.nodes())
    partitions = []
    if warm_start is None:
        partitions.append(list(vertices))
    else:
        partitions = [list(group) for group in warm_start]

    weights = nx.adjacency_matrix(graph, dtype=np.float64)

    color_stats = ColorStats(len(graph), min(n_colors, 128))
    update_stats(color_stats, weights, partitions, weighting=weighting)

    while len(partitions) < n_colors:
        if len(partitions) == color_stats.n:
            color_stats = color_stats.resize(color_stats.v, color_stats.n * 2)
        witness_i, witness_j, split_deg, q_error = pick_witness(partitions, color_stats)
        if q_error <= q:
            break
        split_color(partitions, color_stats, witness_i, witness_j, split_deg)
        update_stats_split(color_stats, weights, partitions, witness_i, len(partitions) - 1)

    return partitions


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

q_color(g)

