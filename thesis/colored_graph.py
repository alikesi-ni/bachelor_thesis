from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt

from GWL_python.color_hierarchy.color_hierarchy_tree import ColorHierarchyTree
from GWL_python.color_hierarchy.color_node import ColorNode
from thesis.color_palette import ColorPalette


class ColoredGraph:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.next_color_id = 0
        self.color_stack_height = 0
        self.color_hierarchy_tree = None
        self.__color()

    def __color(self):
        node_labels = nx.get_node_attributes(self.graph, "label")
        are_nodes_labeled = (len(node_labels) != 0) and (len(set(node_labels.values())) > 1)

        if are_nodes_labeled:
            node_color_attributes = {}
            label_color_map = {}
            color_nodes_map = defaultdict(list)

            root = ColorNode(0)
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

            for color_id, group_nodes in color_nodes_map.items():
                child = ColorNode(color_id)
                child.update_associated_vertices(group_nodes)
                root.add_child(child)

            self.color_hierarchy_tree = ColorHierarchyTree(
                root_node=root,
                is_root_node_artificial=True
            )

        else:
            nx.set_node_attributes(self.graph, {node: [self.next_color_id] for node in self.graph.nodes}, "color-stack")

            root = ColorNode(self.next_color_id)
            root.update_associated_vertices(list(self.graph.nodes))
            self.color_hierarchy_tree = ColorHierarchyTree(
                root_node=root,
                is_root_node_artificial=False
            )

            self.next_color_id += 1

        self.color_stack_height = 1

    def draw(self, hierarchy_level: int = -1, node_size=100):
        """
        Draws the graph with nodes colored based on a given hierarchy level of their color-stack.
        If no level is provided, the last color is used.

        Parameters
        ----------
        hierarchy_level : int
            The level (index) of the color-stack to visualize. Default is the final level (-1).
        with_labels : bool
            Whether to show labels for nodes.
        node_size : int
            Size of the nodes in the plot.
        """
        if self.color_stack_height == 0:
            raise ValueError("Color stack is not initialized. Has the graph been colored yet?")

        if hierarchy_level == -1:
            level = self.color_stack_height - 1
        else:
            level = hierarchy_level

        if not (0 <= level < self.color_stack_height):
            raise ValueError(f"Invalid hierarchy_level {level}. Must be between 0 and {self.color_stack_height - 1}.")

        # Extract colors at the selected level
        node_colors = {node: data["color-stack"][level] for node, data in self.graph.nodes(data=True)}

        # Count frequency of each color
        color_counts = {}
        for color in node_colors.values():
            color_counts[color] = color_counts.get(color, 0) + 1

        print(f"Number of different colors at level {level}: {len(color_counts)}")

        # Assign display colors
        color_map = ColorPalette.assign_color_map(color_counts)

        # Build node color list
        node_color_list = []
        node_labels = {}
        overflow_label_map = {}
        label_index = 1

        for node, color_id in node_colors.items():
            hex_color = color_map.get(color_id)

            if hex_color:
                node_color_list.append(hex_color)
                # no label for colored nodes
            else:
                node_color_list.append("#cccccc")

                if color_id not in overflow_label_map:
                    overflow_label_map[color_id] = f"[{label_index}]"
                    label_index += 1

                node_labels[node] = overflow_label_map[color_id]

        # Plot layout
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, node_color=node_color_list, with_labels=False, node_size=node_size)

        # Show labels only for overflow nodes
        if node_labels:
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10, font_color="black")

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


