import networkx as nx
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


dataset_names = [
    "KKI", "PTC_FM", "COLLAB", "DD", "IMDB-BINARY", "MSRC_9",
    "NCI1", "REDDIT-BINARY", "MUTAG", "ENZYMES", "PROTEINS"
]

for dataset_name in dataset_names:
    try:
        print(f"Loading dataset: {dataset_name}")
        root = './data/'
        dataset = TUDataset(root, dataset_name)
        analyze(dataset)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")

# dataset_name = 'MUTAG'
dataset_name = 'KKI'
# dataset_name = 'PTC_FM'
# dataset_name = 'COLLAB'
# dataset_name = 'DD'
# dataset_name = 'IMDB-BINARY'

root = './data/'
dataset = TUDataset(root, dataset_name)

# G = nx.complete_graph(5)

# graphs = convert_dataset_to_networkx_graphs(dataset)
disjoint_union_graph = convert_to_disjoint_union_graph(dataset)
# print(graphs)

# root = './'
# dataset = TUDataset(root, 'PROTEINS', cleaned=True)
# print(dataset)
