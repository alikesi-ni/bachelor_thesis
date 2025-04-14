import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.data.data import Data


def analyze(dataset):
    """
    Analyze a PyTorch Geometric dataset by printing basic statistics.
    """
    num_graphs = len(dataset)
    num_nodes = 0
    num_edges = 0

    # Calculate total number of nodes and edges
    for data in dataset:
        num_nodes += data.num_nodes
        num_edges += data.num_edges

    avg_nodes = num_nodes / num_graphs if num_graphs > 0 else 0
    avg_edges = num_edges / num_graphs if num_graphs > 0 else 0

    print(f"Dataset: {dataset}")
    print(f"Number of node labels: {dataset.num_node_labels}")
    print(f"Number of edge labels: {dataset.num_edge_labels}")
    print(f"Number of graph labels: {dataset.num_classes}")
    print(f"Number of graphs: {num_graphs}")
    print(f"Total nodes: {num_nodes}")
    print(f"Total edges: {num_edges}")
    print(f"Average nodes per graph: {avg_nodes:.2f}")
    print(f"Average edges per graph: {avg_edges:.2f}")
    print("-" * 50)


def convert_to_networkx_graph(data: Data):
    """
    Convert a single graph data object to a NetworkX graph with labeled nodes and edges.
    Transforms edge attributes to a readable format and assigns them to 'label'.
    """
    try:
        G = to_networkx(data, to_undirected=True)

        # Convert graph label
        if hasattr(data, 'y'):
            try:
                if data.y.size(0) != 1:
                    raise ValueError(f"Graph has an invalid 'y' attribute: {data.y}")
                G.graph['label'] = int(data.y.item())
            except Exception:
                raise ValueError(f"Graph has an invalid 'y' attribute: {data.y}")
        else:
            raise ValueError(f"Graph does not have a 'y' attribute.")

        # Convert node labels
        if hasattr(data, 'x') and data.x is not None:
            node_attrs = data.x.numpy()
            for node in G.nodes():
                try:
                    # One-hot decoding
                    label_positions = [i for i, value in enumerate(node_attrs[node]) if value == 1.0]
                    if len(label_positions) != 1:
                        raise ValueError(f"Node {node} has invalid one-hot encoding: {node_attrs[node]}")
                    label = label_positions[0]
                    G.nodes[node]["label"] = label
                except Exception:
                    raise ValueError(f"Node {node} has an invalid one-hot encoding: {node_attrs[node]}")
        # else:
        #     raise ValueError("Data object does not contain valid node attributes 'x'.")

        # Convert edge labels
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attrs = data.edge_attr.numpy()
            for idx, (u, v) in enumerate(G.edges()):
                try:
                    # One-hot decoding for edge attributes
                    label_positions = [i for i, value in enumerate(edge_attrs[idx]) if value == 1.0]
                    if len(label_positions) != 1:
                        raise ValueError(f"Edge ({u}, {v}) has invalid one-hot encoding: {edge_attrs[idx]}")
                    label = label_positions[0]
                    G[u][v]["label"] = label
                except Exception:
                    raise ValueError(f"Edge ({u}, {v}) has an invalid one-hot encoding: {edge_attrs[idx]}")
        # else:
        #     raise ValueError("Data object does not contain valid edge attributes 'edge_attr'.")

        return G

    except ValueError as e:
        print(f"Error processing graph: {e}")
        raise


def convert_dataset_to_networkx_graphs(dataset: TUDataset):
    """
    Convert an entire dataset into a list of NetworkX graphs.
    """
    graphs = []
    for graph_id, data in enumerate(dataset):
        try:
            graph = convert_to_networkx_graph(data)
            graph.graph['id'] = graph_id
            graphs.append(graph)
        except ValueError as e:
            print(f"Skipping graph {graph_id} due to error: {e}")
    return graphs


def convert_to_disjoint_union_graph(dataset):
    """
    Convert the entire dataset to a single disjoint union graph.
    """
    networkx_graphs = convert_dataset_to_networkx_graphs(dataset)
    disjoint_union_graph = nx.disjoint_union_all(networkx_graphs)
    return disjoint_union_graph

def count_labeled_edges(G):
    """
    Count the number of edges in a NetworkX graph that have a 'label' attribute.
    """
    labeled_edge_count = sum(1 for _, _, data in G.edges(data=True) if "label" in data)
    return labeled_edge_count

def draw_graph_with_labels(G):
    """
    Draw a NetworkX graph with both node labels and edge labels.
    """
    pos = nx.spring_layout(G)  # Position nodes using the spring layout

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=700, font_size=10)

    # Extract and display node labels
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color="black")

    # Extract and display edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

    plt.title("Graph with Node and Edge Labels")
    plt.show()