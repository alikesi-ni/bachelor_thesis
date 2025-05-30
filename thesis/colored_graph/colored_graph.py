import itertools
from collections import defaultdict
from typing import Dict

import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse import csr_array, vstack

from thesis.color_hierarchy.color_hierarchy_tree import ColorHierarchyTree
from thesis.color_hierarchy.color_node import ColorNode
from thesis.colored_graph.color_palette import ColorPalette
from thesis.utils.other_utils import has_distinct_node_labels, convert_to_feature_matrix


class ColoredGraph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.next_color_id = 0
        self.color_stack_height = 0
        self.color_hierarchy_tree = None
        self._color()

    def _color(self):
        node_labels = nx.get_node_attributes(self.graph, "label")
        are_nodes_labeled = has_distinct_node_labels(self.graph)

        if are_nodes_labeled:
            node_color_attributes = {}
            label_color_map = {}
            color_nodes_map = defaultdict(list)

            self.next_color_id = 1  # Reserve 0 for the artificial root

            for node, label in node_labels.items():
                if label in label_color_map:
                    color_id = label_color_map[label]
                else:
                    color_id = self.next_color_id
                    label_color_map[label] = color_id
                    self.next_color_id += 1

                node_color_attributes[node] = [color_id]
                color_nodes_map[color_id].append(node)

            nx.set_node_attributes(self.graph, node_color_attributes, "color-stack")

        else:
            nx.set_node_attributes(self.graph, {node: [self.next_color_id] for node in self.graph.nodes}, "color-stack")
            self.next_color_id += 1

        self.color_stack_height = 1

    def draw(self, hierarchy_level: int = -1, node_size=500, show_color_id=True, show_node_id=True, true_coloring=False):
        """
        Draws the graph with nodes colored based on a given hierarchy level of their color-stack.
        If no level is provided, the last color is used.

        Parameters
        ----------
        hierarchy_level : int
            The level (index) of the color-stack to visualize. Default is the final level (-1).
        node_size : int
            Size of the nodes in the plot.
        show_color_id : bool
            Whether to show the color id of each node.
        show_node_id : bool
            Whether to show the node id.
        """
        if self.color_stack_height == 0:
            raise ValueError("Color stack is not initialized. Has the graph been colored yet?")

        if hierarchy_level == -1:
            level = self.color_stack_height - 1
        else:
            level = hierarchy_level

        if not (0 <= level < self.color_stack_height):
            raise ValueError(f"Invalid hierarchy_level {level}. Must be between 0 and {self.color_stack_height - 1}.")

        node_color_id_map = {node: data["color-stack"][level] for node, data in self.graph.nodes(data=True)}

        if true_coloring:
            unique_colors = set()
            for _, data in self.graph.nodes(data=True):
                for color_id in data["color-stack"][:level + 1]:
                    unique_colors.add(color_id)

            unique_colors = sorted(unique_colors)
            color_map = ColorPalette.map_color_id_to_hex_color_consecutively(unique_colors)
        else:
            color_counts = {}
            for color in node_color_id_map.values():
                color_counts[color] = color_counts.get(color, 0) + 1

            print(f"Number of different colors at level {level}: {len(color_counts)}")

            color_map = ColorPalette.map_color_id_to_hex_color_by_frequency(color_counts)

        # Build node color list and labels
        node_color_list = []
        node_labels = {}

        for node, color_id in node_color_id_map.items():
            hex_color = color_map.get(color_id)
            if hex_color:
                node_color_list.append(hex_color)
            else:
                node_color_list.append("#cccccc")  # fallback

            label_parts = []
            if show_node_id:
                label_parts.append(str(node))
            if show_color_id:
                label_parts.append(f"[{color_id}]")
            if label_parts:
                node_labels[node] = " ".join(label_parts)

        # Plot layout
        pos = nx.spring_layout(self.graph, seed=42)
        with_labels = show_node_id or show_color_id
        nx.draw(self.graph, pos, node_color=node_color_list, with_labels=False, node_size=node_size)

        if with_labels:
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=20, font_color="black")

        plt.title(f"Colored Graph at Hierarchy Level {level}")
        plt.show()

    def get_num_colors(self, hierarchy_level: int = -1) -> int:
        """
        Returns the number of unique colors at a given hierarchy level.

        Parameters
        ----------
        hierarchy_level : int
            The level of the color-stack to evaluate.
            Defaults to the final level (-1).

        Returns
        -------
        int
            Number of unique colors at that hierarchy level.
        """
        if self.color_stack_height == 0:
            raise ValueError("Color stack is not initialized.")

        if hierarchy_level == -1:
            level = self.color_stack_height - 1
        else:
            level = hierarchy_level

        if not (0 <= level < self.color_stack_height):
            raise ValueError(f"Invalid hierarchy_level {level}. Must be between 0 and {self.color_stack_height - 1}.")

        unique_colors = {data["color-stack"][level] for _, data in self.graph.nodes(data=True)}
        return len(unique_colors)

    def assert_consistent_color_stack_height(self):
        """
        Asserts that all nodes in the graph have a color-stack of the same length as self.color_stack_height.

        Raises
        ------
        AssertionError
            If any node's color-stack length does not match self.color_stack_height.
        """
        for node, data in self.graph.nodes(data=True):
            actual_height = len(data.get("color-stack", []))
            assert actual_height == self.color_stack_height, (
                f"Node {node} has color-stack of height {actual_height}, "
                f"expected {self.color_stack_height}."
            )

    def build_color_hierarchy_tree(self):
        """
        Reconstructs the ColorHierarchyTree based on the node color-stacks in the graph.
        Sets the result to self.color_hierarchy_tree.
        """

        node_map = {}  # color_id -> ColorNode
        edge_links = defaultdict(set)  # parent_id -> set of child_ids
        level0_colors = set()

        # Step 1: Create ColorNode objects and register parent # child relationships
        for node, data in self.graph.nodes(data=True):
            color_stack = data["color-stack"]
            if len(color_stack) < 1:
                continue

            level0_colors.add(color_stack[0])

            for i, color_id in enumerate(color_stack):
                if color_id not in node_map:
                    node_map[color_id] = ColorNode(color_id)
                if i > 0:
                    parent = color_stack[i - 1]
                    edge_links[parent].add(color_id)  # build mapping

        # Step 2: Assign associated vertices to leaf nodes (last color in stack)
        for node, data in self.graph.nodes(data=True):
            leaf_color = data["color-stack"][-1]
            node_map[leaf_color].associated_vertices.append(node)

        # Step 3: Link parent # children in sorted order
        for parent_id, child_ids in edge_links.items():
            for child_id in sorted(child_ids):
                node_map[parent_id].add_child(node_map[child_id])

        # Step 4: Create artificial root if necessary
        if len(level0_colors) == 1:
            root = node_map[list(level0_colors)[0]]
            is_root_artificial = False
        else:
            root = ColorNode(0)  # convention: 0 is reserved for root
            for color_id in level0_colors:
                root.add_child(node_map[color_id])
            root.associated_vertices = list(self.graph.nodes)
            node_map[0] = root
            is_root_artificial = True

        # Register all in tree
        tree = ColorHierarchyTree(root_node=root, is_root_node_artificial=is_root_artificial)
        tree.node_map = {cid: cnode for cid, cnode in node_map.items()}

        self.color_hierarchy_tree = tree

    def generate_feature_vector(self) -> Dict[int, int]:
        """
    Generates a color histogram feature vector for the current graph.

    Returns
    -------
    Dict[int, int]
        A dictionary where keys are color IDs (from the color-stack)
        and values are their frequency counts across all nodes.
    """

        fv = dict()

        for color in list(itertools.chain(*nx.get_node_attributes(self.graph, "color-stack").values())):

            if color in fv.keys():
                fv[color] += 1
            else:
                fv[color] = 1

        return fv

    def generate_gid_to_feature_vector_map(self) -> Dict[int, Dict[int, int]]:
        """
        Generates color histogram feature vectors for all subgraphs
        contained within the disjoint union graph.

        Assumes that each node has:
        - a 'gid' attribute indicating which graph it belongs to
        - a 'color-stack' attribute produced by color refinement

        Returns
        -------
        Dict[int, Dict[int, int]]
            A dictionary mapping each graph ID (gid) to its corresponding
            color histogram {color_id: count}.
        """

        fvs: Dict[int, Dict[int, int]] = dict()

        nodes = self.graph.nodes

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

    def generate_feature_matrix(self) -> csr_array:
        """
        Converts the gid-to-feature-vector mapping into a sparse feature matrix.

        Each row corresponds to a graph (gid), and each column corresponds to a
        color ID. The color ID is used directly as the column index.

        Assumes:
        - Color IDs are consecutive integers starting at 0 or 1
        - GIDs are unique integers, possibly starting at 0

        Returns
        -------
        csr_array
            A sparse matrix of shape [max_gid + 1 x max_color_id + 1]
            where row i corresponds to gid = i, and column j corresponds to color ID = j.
        """

        gid_to_feature_vector_map = self.generate_gid_to_feature_vector_map()

        return convert_to_feature_matrix(gid_to_feature_vector_map)

    def copy(self):
        """
        Creates a deep copy of the ColoredGraph, including node attributes and color stack state.

        Returns
        -------
        ColoredGraph
            A new instance of ColoredGraph with identical graph structure and coloring.
        """
        new_graph = self.graph.copy()
        new_colored_graph = ColoredGraph(new_graph)

        # Manually copy additional attributes
        new_colored_graph.next_color_id = self.next_color_id
        new_colored_graph.color_stack_height = self.color_stack_height

        # Copy color-stack explicitly (deep copy of lists)
        for node in self.graph.nodes:
            color_stack = self.graph.nodes[node].get("color-stack", []).copy()
            new_colored_graph.graph.nodes[node]["color-stack"] = color_stack

        new_colored_graph.color_hierarchy_tree = None

        return new_colored_graph
