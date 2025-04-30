import networkx as nx

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph


def build_qerror_increase_graph():
    """
    Construct a graph where q_error increases after a refinement step.
    This happens because the first split causes a symmetric structure to become asymmetric.
    """
    G = nx.Graph()

    # Left triangle
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Right triangle
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])

    # Central line
    G.add_edges_from([(6, 7), (7, 8)])

    # Connect each triangle to one end of the line
    G.add_edge(2, 6)  # node from left triangle to line
    G.add_edge(5, 8)  # node from right triangle to line

    return G

def build_qerror_jump_complex():
    """
    Build a larger graph designed to produce a visible q_error increase
    mid-way through refinement.
    """
    G = nx.Graph()

    # Cluster 1: A 5-node cycle
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0)
    ])

    G.add_edge(0, 5)
    G.add_edge(4, 5)

    G.add_edge(2, 6)
    G.add_edge(4, 6)

    return G

G = nx.Graph()
G.add_edges_from([
    (0, 1), (1, 2), (2, 3), (2, 4), (1, 5), (2, 5)
])
G=build_qerror_jump_complex()
cg = ColoredGraph(G)
qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=30, verbose=True)
qsc.refine(verbose=True)

color_stack_height = cg.color_stack_height
for i in range (color_stack_height):
    cg.draw(i, show_color_id=False, true_coloring=True)