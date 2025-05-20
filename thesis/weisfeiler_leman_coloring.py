import networkx as nx

from thesis.colored_graph.colored_graph import ColoredGraph
from thesis.utils.other_utils import has_distinct_edge_labels


class WeisfeilerLemanColoringGraph:
    def __init__(self, colored_graph: ColoredGraph, refinement_steps = 1):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.refinement_steps = refinement_steps
        self.current_step = 0  # Track the number of refinement steps performed so far
        self.is_stable = False  # Flag to prevent refinement beyond stability
        self.previous_num_colors = colored_graph.get_num_colors()


    def refine_one_step(self, verbose=False):
        if self.is_stable:
            if verbose:
                print("Graph is already stable. No further refinement.")
            return self.previous_num_colors, self.current_step

        are_edges_labeled = has_distinct_edge_labels(self.graph)

        color_hashes = {}
        color_map = {}

        for node in self.graph.nodes:
            own_color = str(self.graph.nodes[node]["color-stack"][-1])

            if are_edges_labeled:
                edge_to_label_map = nx.get_edge_attributes(self.graph, "label")
                neighbor_hashes = []
                for neighbor in self.graph.neighbors(node):
                    edge = (node, neighbor) if (node, neighbor) in edge_to_label_map else (neighbor, node)
                    neighbor_color = str(self.graph.nodes[neighbor]["color-stack"][-1])
                    edge_label = str(edge_to_label_map[edge])
                    neighbor_hashes.append(neighbor_color + edge_label)
            else:
                neighbor_hashes = [str(self.graph.nodes[neighbor]["color-stack"][-1])
                                   for neighbor in self.graph.neighbors(node)]

            combined_hash = own_color + "".join(sorted(neighbor_hashes))
            color_hashes[node] = combined_hash

            if combined_hash not in color_map:
                color_map[combined_hash] = self.colored_graph.next_color_id
                self.colored_graph.next_color_id += 1

        num_colors = len(color_map)

        # Check if coloring changed — if not, mark as stable
        if num_colors == self.previous_num_colors:
            self.is_stable = True
            verbose and print(
                f"Refinement step {self.current_step}: Coloring stabilized at {num_colors} colors."
            )
            return num_colors, self.current_step

        # Otherwise, update the color stack
        for node in self.graph.nodes:
            new_color = color_map[color_hashes[node]]
            self.graph.nodes[node]["color-stack"].append(new_color)

        self.colored_graph.color_stack_height += 1
        self.current_step += 1
        self.previous_num_colors = num_colors

        verbose and print(f"Number of colors after iteration {self.current_step}: {num_colors}")

        return num_colors, self.current_step

    def refine(self, verbose=False):
        while self.current_step < self.refinement_steps and not self.is_stable:
            self.refine_one_step(verbose=verbose)

    def refine_until_stable(self, verbose=False):
        while not self.is_stable:
            self.refine_one_step(verbose=verbose)
