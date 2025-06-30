import networkx as nx

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph

G = nx.Graph()

G.add_edges_from([
    (0, 1),
    (1, 2), (1, 3),
    (2, 3), (2, 4),
    (3, 4),
    (4, 5), (4, 6),
    (5, 6)
])

# Assign labels
labels = {
    0: 0,
    2: 0,
    3: 0,
    1: 1,
    4: 1,
    5: 1,
    6: 1
}
nx.set_node_attributes(G, labels, "label")


cg = ColoredGraph(G)
qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=30)
qsc.refine()

color_stack_height = cg.color_stack_height
for i in range (color_stack_height):
    cg.draw(i, show_color_id=False, true_coloring=True)