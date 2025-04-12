import networkx as nx
from GWL_python.clustering.kmeans import KMeans
import GWL_python.gwl.utils as gwl_utils


class GradualWeisfeilerLemanGraph:
    """
    Refines a ColoredGraph using Gradual Weisfeiler-Leman refinement with k-means clustering.
    Updates only the 'color-stack' in-place.
    """

    def __init__(self, colored_graph, refinement_steps: int, n_cluster: int, seed: int = 4321):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.h = refinement_steps
        self.k = n_cluster
        self.seed = seed

    def refine(self):
        """
        Runs h iterations of gradual refinement using k-means clustering on neighbor color histograms.
        Updates the 'color-stack' in-place on the graph.
        """
        if "color-stack" not in next(iter(self.graph.nodes(data=True)))[1]:
            raise ValueError("Graph must be initialized with a 'color-stack' (use ColoredGraph first).")

        for _ in range(self.h):
            self.__renep()

        self.colored_graph.color_stack_height = len(self.graph.nodes[next(iter(self.graph.nodes))]["color-stack"])
        self.colored_graph.next_color_id = max(
            self.graph.nodes[n]["color-stack"][-1] for n in self.graph.nodes
        ) + 1

    def __renep(self):
        edge_labels = nx.get_edge_attributes(self.graph, "label")
        color_to_nodes = {}

        # Group nodes by their current color
        for node, data in self.graph.nodes(data=True):
            last_color = data["color-stack"][-1]
            color_to_nodes.setdefault(last_color, []).append(node)

        next_color_id = max(color_to_nodes.keys()) + 1

        for _, nodes in color_to_nodes.items():
            if len(nodes) > 1:
                neighbor_counts = {
                    node: gwl_utils.generate_neighbor_color_count(self.graph, node, edge_labels)
                    for node in nodes
                }

                # Build a sorted key index (vocabulary) across all neighbor histograms
                all_keys = sorted({key for hist in neighbor_counts.values() for key in hist})
                node_to_vector = {}

                for node in nodes:
                    vector = [neighbor_counts[node].get(key, 0) for key in all_keys]
                    node_to_vector[node] = vector

                # Now it's in the correct format: dict[int, list[int]]
                kmeans = KMeans(k=self.k, initialization_method="kmeans++", seed=self.seed)
                clusters = kmeans.fit_and_assign_cluster(node_to_vector)
            else:
                clusters = {0: nodes}

            for cluster_nodes in clusters.values():
                if not cluster_nodes:
                    continue
                for node in cluster_nodes:
                    self.graph.nodes[node]["color-stack"].append(next_color_id)
                next_color_id += 1
