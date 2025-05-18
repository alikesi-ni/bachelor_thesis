import networkx as nx

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph

G = nx.Graph()

G.add_edges_from([
    (0, 1), (0, 3), (0, 5),
    (1, 2),
    (2, 3), (2, 6),
    (4, 5), (4, 6)
])


cg = ColoredGraph(G)
qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=30)
qsc.refine()

color_stack_height = cg.color_stack_height
for i in range (color_stack_height):
    cg.draw(i, show_color_id=False, true_coloring=True)