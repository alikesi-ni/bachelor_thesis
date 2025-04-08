import itertools

import networkx as nx


class WeisfeilerLeman:
    """
    Implementation of Weisfeiler Leman that also takes edge label into account for updating the colors.

    Parameters
    ----------
    refinement_steps : int
        Number of refinement steps to be performed

    """

    def __init__(self, refinement_steps: int) -> None:
        self.h = refinement_steps

        self.__is_refined = None

    def refine_color(self, graph: nx.Graph) -> None:

        """
        Refines the color of the nodes of the given graph for the given number of refinement steps.

        Parameters
        ----------
        graph : nx.Graph
            A nx.Graph instance to be refined

        """

        # checks whether the nodes and edges of the graph are labeled
        node_labels = nx.get_node_attributes(graph, "label")
        are_nodes_labeled = (len(node_labels) != 0) and (len(set(node_labels.values())) > 1)

        edge_labels = nx.get_edge_attributes(graph, "label")
        are_edges_labeled = len(edge_labels) != 0

        color = 0

        # --- initial vertex coloring --- #

        # graph with labeled vertices
        if are_nodes_labeled:

            node_color_attributes = dict()
            label_color_map = dict()

            for node, label in node_labels.items():

                if label in label_color_map.keys():
                    node_color_attributes[node] = [label_color_map[label]]
                else:
                    node_color_attributes[node] = [color]
                    label_color_map[label] = color
                    color += 1

            nx.set_node_attributes(graph, node_color_attributes, "color-stack")

        # graph with unlabeled vertices
        else:
            nx.set_node_attributes(graph, {node: [color] for node in graph.nodes}, "color-stack")
            color += 1

        print(f"Number of colors after iteration 0: {color}")

        # refine color for given number of refinement steps
        for i in range(self.h):

            color_hashes = dict()
            color_map = dict()

            for node in graph.nodes:

                own_color_hash = str(graph.nodes[node]["color-stack"][-1])

                if are_edges_labeled:

                    neighbor_color_hashes = list()

                    for neighbor in graph.neighbors(node):
                        edge = (node, neighbor) if (node, neighbor) in edge_labels.keys() else (neighbor, node)
                        neighbor_color_hash = "".join(
                            (str(graph.nodes[neighbor]["color-stack"][-1]), str(edge_labels[edge])))

                        neighbor_color_hashes.append(neighbor_color_hash)

                else:
                    neighbor_color_hashes = [str(graph.nodes[neighbor]["color-stack"][-1]) for neighbor in
                                             graph.neighbors(node)]

                neighbor_color_hashes = "".join(sorted(neighbor_color_hashes))
                new_color_hash = own_color_hash + neighbor_color_hashes

                color_hashes[node] = new_color_hash

                if new_color_hash not in color_map.keys():
                    color_map[new_color_hash] = color
                    color += 1

            # assign the refined color to the nodes
            for node in graph.nodes:
                graph.nodes[node]["color-stack"].extend([color_map[color_hashes[node]]])

            # Count the number of unique colors after refinement
            unique_colors = set()

            # Iterate through each node to collect the last color in the color-stack
            for node in graph.nodes:
                # Access the last color in the color-stack
                last_color = graph.nodes[node]["color-stack"][-1]
                # Add the last color to the set (to ensure uniqueness)
                unique_colors.add(last_color)

            print(f"Number of colors after iteration {i + 1}: {len(unique_colors)}")

        # indicating that the refinement process is completed
        self.__is_refined = True

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
