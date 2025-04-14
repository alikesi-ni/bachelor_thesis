import networkx as nx

from thesis.clustering.kmeans import KMeans
from thesis.color_hierarchy.color_hierarchy_tree import ColorHierarchyTree
from thesis.color_hierarchy.color_node import ColorNode
from thesis.colored_graph.colored_graph import ColoredGraph


class GWLColoringGraph:
    def __init__(self, colored_graph: ColoredGraph, refinement_steps: int, n_cluster: int,
                 cluster_initialization_method: str = "kmeans++",
                 num_forgy_iterations: int = 15, seed: int = 4321) -> None:

        self.colored_graph = colored_graph
        self.graph = colored_graph.graph

        self.refinement_steps = refinement_steps # h in the paper
        self.n_cluster = n_cluster # k in the paper
        self.cluster_initialization_method = cluster_initialization_method
        self.num_forgy_iterations = num_forgy_iterations
        self.seed = seed

        self.__is_refined = None
        self.color_hierarchy_tree = None

    def refine(self, verbose: bool = False):
        """
        Refines the color of the graph stored in self.colored_graph using the GWL method.
        """
        self.colored_graph.build_color_hierarchy_tree()

        self.color_hierarchy_tree = self.colored_graph.color_hierarchy_tree

        # --- iterative refinement ---
        for _ in range(self.refinement_steps):
            self.__renep()
            self.update_colors()

        # Finalize
        self.__is_refined = True

        if verbose:
            self.color_hierarchy_tree.print_tree()

    def __renep(self) -> None:
        leaves = self.color_hierarchy_tree.get_leaves()
        edge_labels = nx.get_edge_attributes(self.graph, "label")

        for leaf in leaves:
            if len(leaf.associated_vertices) > 1:
                neighbor_color_count = {
                    vertex: self.generate_neighbor_color_count(vertex, edge_labels)
                    for vertex in leaf.associated_vertices
                }

                unique_colors = sorted(set().union(*neighbor_color_count.values()))

                for key, value in neighbor_color_count.items():
                    full_count_vector = [value.get(color, 0) for color in unique_colors]
                    neighbor_color_count[key] = full_count_vector

                kmeans = KMeans(
                    k=self.n_cluster,
                    initialization_method=self.cluster_initialization_method,
                    num_forgy_iterations=self.num_forgy_iterations,
                    seed=self.seed
                )
                clusters = kmeans.fit_and_assign_cluster(neighbor_color_count)

            else:
                clusters = {0: leaf.associated_vertices}

            for refined_nodes in clusters.values():
                if not refined_nodes:
                    continue
                node = ColorNode(self.colored_graph.next_color_id)
                node.update_associated_vertices(refined_nodes)
                leaf.add_child(node)
                self.colored_graph.next_color_id += 1

    def update_colors(self) -> None:
        """
        Updates the color of the associated nodes for all leaves of the ColorHierarchyTree.

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph instance that is being refined

        color_hierarchy_tree : ColorHierarchyTree
            Corresponding instance of ColorHierarchyTree

        """

        for leaf in self.color_hierarchy_tree.get_leaves():
            for node in leaf.associated_vertices:
                self.graph.nodes[node]["color-stack"].extend([leaf.color])
        self.colored_graph.color_stack_height += 1

    def generate_neighbor_color_count(self, vertex: int, edge_labels: dict) -> dict:
        """
        Generates the neighbor color count for a given vertex.

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph instance that is being refined

        vertex : int
            ID of the node/ vertex for which the neighbor color count need to be generated

        edge_labels : dict
            A dictionary, where the key is the edge, and the corresponding value is its assigned label

        Returns
        -------
        out : dict
            A dictionary, where a key-value pair represent a node, and the count of node colors of its neighbors

        """

        color_neighbor_count = dict()

        for neighbor in nx.neighbors(self.graph, vertex):

            if len(edge_labels) != 0:
                edge = (vertex, neighbor) if (vertex, neighbor) in edge_labels.keys() else (neighbor, vertex)
                neighbor_color = (self.graph.nodes[neighbor]["color-stack"][-1], edge_labels[edge])

            else:
                neighbor_color = self.graph.nodes[neighbor]["color-stack"][-1]

            if neighbor_color in color_neighbor_count.keys():
                color_neighbor_count[neighbor_color] += 1
            else:
                color_neighbor_count[neighbor_color] = 1

        return color_neighbor_count
