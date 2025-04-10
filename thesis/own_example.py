import networkx as nx

from thesis.colored_graph import ColoredGraph
from thesis.other_utils import generate_feature_vector
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.read_data_utils import dataset_to_graphs

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

g = dataset_to_graphs("./data", "PTC_FM")[1]
g = nx.convert_node_labels_to_integers(g, first_label=0)

# qsc = QuasiStableColoringGraph(g, 1.0)
# print(qsc.refine())
# feature_vector = generate_feature_vector(g)
# print(dict(sorted(feature_vector.items())))

colored_graph = ColoredGraph(g)
colored_graph.draw()