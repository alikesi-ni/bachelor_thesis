import networkx as nx

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.utils.other_utils import analyze
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph


def run_qsc_diagnostics(disjoint_graph: nx.Graph, max_iter: int = 10):
    wl_colored_graph = ColoredGraph(disjoint_graph)

    for iteration in range(max_iter):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Copy for QSC
        qsc_colored_graph = wl_colored_graph.copy()
        num_colors = qsc_colored_graph.get_num_colors()
        print(f"WL Color Count = {num_colors}")

        qsc = QuasiStableColoringGraph(colored_graph=qsc_colored_graph, n_colors=num_colors + 1)
        qsc.refine(verbose=True)

        # WL Step
        wl = WeisfeilerLemanColoringGraph(wl_colored_graph, refinement_steps=1)
        wl.refine(verbose=False)

dataset_name = "IMDB-BINARY"

graphs = dataset_to_graphs("../data", dataset_name)

analyze(graphs)

disjoint_graph = nx.disjoint_union_all(graphs)

run_qsc_diagnostics(disjoint_graph)

