import networkx as nx
import numpy as np

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.logger_config import LoggerFactory
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph

G = nx.Graph()

G.add_edges_from([
    (0, 1), (0,5), (0, 6),
    (1, 2), (1, 7),
    (2, 3),
    (3, 4), (3, 8),
    (4, 5)
])

logger = LoggerFactory.get_console_logger(__name__);

cg = ColoredGraph(G)
# qsc = QuasiStableColoringGraph(cg, q=0.0, n_colors=30, logger=logger)
# qsc.refine()
wl = WeisfeilerLemanColoringGraph(cg, refinement_steps=np.inf)
wl.refine()

color_stack_height = cg.color_stack_height
for i in range (color_stack_height):
    cg.draw(i, show_color_id=False, true_coloring=True)