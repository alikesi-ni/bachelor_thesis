import networkx as nx

from thesis.colored_graph import ColoredGraph
from thesis.other_utils import generate_feature_vector, remove_node_labels
from thesis.quasi_stable_coloring import QuasiStableColoringGraph
from thesis.read_data_utils import dataset_to_graphs
from thesis.weisfeiler_leman_coloring import WeisfeilerLemanColoringGraph

# Create an undirected graph
g = nx.Graph()

# Add nodes
g.add_nodes_from(range(8))

# Add edges based on the structure provided
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5), (4, 7),
    (5, 6)
]

# Add edges to the graph
g.add_edges_from(edges)

g = dataset_to_graphs("./data", "PTC_FM")[1]
g = nx.convert_node_labels_to_integers(g, first_label=0)

print("Number of nodes:", g.number_of_nodes())

# remove_node_labels(g)

# qsc = QuasiStableColoringGraph(g, 1.0)
# print(qsc.refine())
# feature_vector = generate_feature_vector(g)
# print(dict(sorted(feature_vector.items())))

colored_graph_a = ColoredGraph(g)
qsc = QuasiStableColoringGraph(colored_graph_a, q=0)
qsc.refine()

colored_graph_a.assert_consistent_color_stack_height()
unique_colors = qsc.colored_graph.get_num_colors()

print("### Quasi Stable Coloring ###")
print(f"Number of unique colors: {unique_colors}")
print(f"Color-stack height: {qsc.colored_graph.color_stack_height}")

colored_graph_a.draw()

colored_graph_b = ColoredGraph(g)
wl = WeisfeilerLemanColoringGraph(colored_graph_b, refinement_steps=1) # 9 for stable coloring
wl.refine()

colored_graph_b.assert_consistent_color_stack_height()
unique_colors = wl.colored_graph.get_num_colors()

print("### Quasi Stable Coloring ###")
print(f"Number of unique colors: {unique_colors}")
print(f"Color-stack height: {qsc.colored_graph.color_stack_height}")

colored_graph_b.draw()

colored_graph_b.build_color_hierarchy_tree()
colored_graph_b.color_hierarchy_tree.visualize_tree()

# # Draw the graph at each refinement level
# for level in range(colored_graph_a.color_stack_height):
#     print(f"\nDrawing graph at refinement level {level}")
#     colored_graph_a.draw(hierarchy_level=level)
