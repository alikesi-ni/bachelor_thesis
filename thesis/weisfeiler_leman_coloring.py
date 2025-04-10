import networkx as nx

from thesis.colored_graph import ColoredGraph


class WeisfeilerLemanColoring:
    def __init__(self, colored_graph: ColoredGraph, refinement_steps: int = 1):
        self.colored_graph = colored_graph
        self.graph = colored_graph.graph
        self.refinement_steps = refinement_steps
        self.edge_labels = nx.get_edge_attributes(self.graph, "label")
        self.are_edges_labeled = len(self.edge_labels) != 0

    def refine(self):
        for i in range(self.refinement_steps):
            color_hashes = {}
            color_map = {}

            for node in self.graph.nodes:
                own_color = str(self.graph.nodes[node]["color-stack"][-1])

                if self.are_edges_labeled:
                    neighbor_hashes = []
                    for neighbor in self.graph.neighbors(node):
                        edge = (node, neighbor) if (node, neighbor) in self.edge_labels else (neighbor, node)
                        neighbor_color = str(self.graph.nodes[neighbor]["color-stack"][-1])
                        edge_label = str(self.edge_labels[edge])
                        neighbor_hashes.append(neighbor_color + edge_label)
                else:
                    neighbor_hashes = [str(self.graph.nodes[neighbor]["color-stack"][-1])
                                       for neighbor in self.graph.neighbors(node)]

                combined_hash = own_color + "".join(sorted(neighbor_hashes))
                color_hashes[node] = combined_hash

                if combined_hash not in color_map:
                    color_map[combined_hash] = self.colored_graph.next_color_id
                    self.colored_graph.next_color_id += 1

            # Assign refined colors
            for node in self.graph.nodes:
                new_color = color_map[color_hashes[node]]
                self.graph.nodes[node]["color-stack"].append(new_color)

            # Update state in ColoredGraph
            self.colored_graph.color_stack_height += 1
