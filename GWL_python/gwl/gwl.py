import itertools
from typing import Union

import networkx as nx

from GWL_python.clustering.kmeans import KMeans
from GWL_python.color_hierarchy.color_node import ColorNode
from GWL_python.color_hierarchy.color_hierarchy_tree import ColorHierarchyTree

import GWL_python.gwl.utils as gwl_utils


class GradualWeisfeilerLeman:
    """
    Implementation of Gradual Weisfeiler Leman algorithm inspired from the paper:

    Bause, Franka & Kriege, Nils. (2022). Gradual Weisfeiler-Leman: Slow and Steady Wins the Race.
    10.48550/arXiv.2209.09048.

    Abstract
    --------
    The classical Weisfeiler-Leman algorithm aka color refinement is fundamental for graph learning and central
    for successful graph kernels and graph neural networks. Originally developed for graph isomorphism testing,
    the algorithm iteratively refines vertex colors. On many datasets, the stable coloring is reached after a few
    iterations and the optimal number of iterations for machine learning tasks is typically even lower. This
    suggests that the colors diverge too fast, defining a similarity that is too coarse. We generalize the concept
    of color refinement and propose a framework for gradual neighborhood refinement, which allows a slower
    convergence to the stable coloring and thus provides a more fine-grained refinement hierarchy and vertex
    similarity. We assign new colors by clustering vertex neighborhoods, replacing the original injective color
    assignment function. Our approach is used to derive new variants of existing graph kernels and to approximate
    the graph edit distance via optimal assignments regarding vertex similarity. We show that in both tasks,
    our method outperforms the original color refinement with only moderate increase in running time advancing
    the state of the art.

    Parameters
    ----------
    refinement_steps : int
        Number of refinement steps to be performed

    n_cluster : int
        Number of clusters

    cluster_initialization_method : str
        Method to use for cluster center initialization (default: "kmeans++")

    num_forgy_iterations : int
        Number of clustering iterations to be performed for selecting the best clustering using "forgy" initialization.
        Has effect only if the initialization method is "forgy" (default: 15)

    seed : int
        Seed for Numpy random functions to ensure reproducibility (default: 4321)

    """

    def __init__(self, refinement_steps: int, n_cluster: int, cluster_initialization_method: str = "kmeans++",
                 num_forgy_iterations: int = 15, seed: int = 4321) -> None:

        self.h = refinement_steps
        self.k = n_cluster

        self.cluster_initialization_method = cluster_initialization_method
        self.num_forgy_iterations = num_forgy_iterations
        self.seed = seed

        # for indicating that the refinement process is completed
        self.__is_refined = None

    def refine_color(self, graph: nx.Graph, verbose: bool = False) -> Union[ColorHierarchyTree, None]:

        """
        Refines the color of the nodes of the given graph according to GWL for given number of refinement steps.

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph instance to be refined

        verbose : bool
            If True, prints and visualizes the generated color hierarchy tree at the end of the color refinement.
            Useful for debugging (default: False)

        Returns
        -------
        out : None or ColorHierarchyTree
            If verbose is True then the final ColorHierarchyTree is returned, otherwise None

        """

        # checks whether the nodes of the graph are labeled
        node_labels = nx.get_node_attributes(graph, "label")
        create_artificial_root_node = (len(node_labels) != 0) and (len(set(node_labels.values())) > 1)

        initial_color = 0
        color = 0

        # --- initial vertex coloring & color hierarchy initialization (T_0) --- #

        # graph with labeled vertices
        if create_artificial_root_node:

            node_color_attributes = dict()
            label_color_map = dict()

            color_node_map = dict()

            for node, label in node_labels.items():

                if label in label_color_map.keys():
                    node_color_attributes[node] = label_color_map[label]
                else:
                    color += 1
                    node_color_attributes[node] = color
                    label_color_map[label] = color

                if color in color_node_map.keys():
                    color_node_map[node_color_attributes[node]] += [node]
                else:
                    color_node_map[node_color_attributes[node]] = [node]

            node_color_attributes = {k: [v] for k, v in node_color_attributes.items()}

            nx.set_node_attributes(graph, node_color_attributes, "color-stack")

            root_color_node = ColorNode(color=initial_color)
            root_color_node.update_associated_vertices(graph.nodes())

            for color in color_node_map.keys():
                color_node = ColorNode(color=color)
                color_node.update_associated_vertices(color_node_map[color])

                root_color_node.add_child(color_node)

            color_hierarchy = ColorHierarchyTree(root_node=root_color_node, is_root_node_artificial=True)

        # graph with unlabeled vertices
        else:
            nx.set_node_attributes(graph, {node: [initial_color] for node in graph.nodes}, "color-stack")

            root_color_node = ColorNode(color=initial_color)
            root_color_node.update_associated_vertices(graph.nodes())

            color_hierarchy = ColorHierarchyTree(root_node=root_color_node, is_root_node_artificial=False)

        # refining for given number of refinement steps
        for _ in range(self.h):
            # update color-hierarchy T_i
            color_hierarchy = self.__renep(graph, color_hierarchy)

            # assign colors to vertices
            gwl_utils.update_colors(graph, color_hierarchy)

        # indicating that the refinement process is completed
        self.__is_refined = True

        if verbose:
            color_hierarchy.print_tree()
            return color_hierarchy

    def __renep(self, graph: nx.Graph, color_hierarchy_tree: ColorHierarchyTree) -> ColorHierarchyTree:

        """
        Color neighborhood preserving function as presented in the paper. Generates neighborhood color multiset,
        Performs clustering on them, and updates the corresponding ColorHierarchyTree.

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph instance to be refined

        color_hierarchy_tree : ColorHierarchyTree
            Corresponding instance of ColorHierarchyTree

        Returns
        -------
        out : ColorHierarchyTree
            Updated ColorHierarchyTree instance

        """

        last_color = color_hierarchy_tree.get_last_color()

        leaves = color_hierarchy_tree.get_leaves()

        # edge labels of the graph
        edge_labels = nx.get_edge_attributes(graph, "label")

        for leaf in leaves:

            if len(leaf.associated_vertices) > 1:

                neighbor_color_count = dict()

                for vertex in leaf.associated_vertices:
                    neighbor_color_count[vertex] = gwl_utils.generate_neighbor_color_count(graph, vertex, edge_labels)

                # add missing colors to color count mapping
                unique_colors = sorted(set().union(*neighbor_color_count.values()))

                # generate neighbor color multiset
                for key, value in neighbor_color_count.items():

                    available_colors = value.keys()
                    all_color_count = list()

                    for color in unique_colors:
                        if color in available_colors:
                            all_color_count.append(value[color])
                        else:
                            all_color_count.append(0)

                    neighbor_color_count[key] = all_color_count

                kmeans = KMeans(k=self.k, initialization_method=self.cluster_initialization_method,
                                num_forgy_iterations=self.num_forgy_iterations, seed=self.seed)
                clusters = kmeans.fit_and_assign_cluster(neighbor_color_count)

            else:
                # assign each vertex its own cluster
                clusters = {0: leaf.associated_vertices}

            for _, refined_nodes in clusters.items():

                if len(refined_nodes) != 0:
                    last_color += 1
                    node = ColorNode(last_color)
                    node.update_associated_vertices(refined_nodes)

                    leaf.add_child(node)

        return color_hierarchy_tree

    def generate_feature_vector(self, refined_graph: nx.Graph) -> dict:

        """
        Generates the feature vector for a single refined graph.

        Parameters
        ----------
        refined_graph : nx.Graph
            A nx.Graph instance that has already been refined using GWL

        Returns
        -------
        out : dict
            A dictionary, where a key-value pair represent a color, and its count.

        """

        assert self.__is_refined, ("The color refinement step must be executed before the feature vector for the "
                                   "given graph can be generated !!!")

        fv = dict()

        for color in list(itertools.chain(*nx.get_node_attributes(refined_graph, "color-stack").values())):

            if color in fv.keys():
                fv[color] += 1
            else:
                fv[color] = 1

        return fv

    def generate_feature_vectors(self, refined_disjoint_graph: nx.Graph) -> dict:

        """
        Generates the feature vectors for all the graphs.

        Parameters
        ----------
        refined_disjoint_graph : nx.Graph
            A nx.Graph instance (which is disjoint union of all the graphs) that has already been refined using GWL

        Returns
        -------
        out : dict
            A dictionary, where a key is the graph id and value is a dictionary containing the corresponding
            feature vector.

        """

        assert self.__is_refined, ("The color refinement step must be executed before the feature vector for the "
                                   "given graph can be generated !!!")

        fvs = dict()

        nodes = refined_disjoint_graph.nodes

        for node in nodes:

            gid = nodes[node]["gid"]
            colors = nodes[node]["color-stack"]

            if gid not in fvs.keys():
                fvs[gid] = dict()

            for color in colors:
                if color in fvs[gid].keys():
                    fvs[gid][color] += 1
                else:
                    fvs[gid][color] = 1

        return fvs
