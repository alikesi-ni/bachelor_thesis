import os

import networkx as nx
from pathlib import Path


def dataset_to_graphs(data_directory, dataset) -> [nx.Graph]:
    """
    Parses and loads graphs, including with their node and edge labels if available.

    """
    graph_id_node_id_mapping = dict()
    node_id_graph_id_mapping = dict()

    with open(os.path.join(data_directory, dataset, f"{dataset}_graph_indicator.txt"), mode="r") as f:

        for node_id, graph_id in enumerate(f.readlines()):

            node_id = node_id + 1  # +1 since the index starts from 0
            graph_id = int(graph_id.rstrip())

            if graph_id in graph_id_node_id_mapping.keys():
                graph_id_node_id_mapping[graph_id] += [node_id]
            else:
                graph_id_node_id_mapping[graph_id] = [node_id]

            node_id_graph_id_mapping[node_id] = graph_id

    graph_id_edge_mapping = dict()
    graph_id_edge_label_mapping = None

    # edge labels (optional)
    has_edge_labels = f"{dataset}_edge_labels.txt" in os.listdir(os.path.join(data_directory, dataset))

    if has_edge_labels:
        graph_id_edge_label_mapping = {gid: dict() for gid in graph_id_node_id_mapping.keys()}

        with open(os.path.join(data_directory, dataset, f"{dataset}_edge_labels.txt"), mode="r") as f:
            eid_to_edge_label_mapping = {eid: int(label.rstrip()) for eid, label in enumerate(f.readlines())}

    with open(os.path.join(data_directory, dataset, f"{dataset}_A.txt"), mode="r") as f:

        for eid, edge in enumerate(f.readlines()):

            v1, v2 = map(int, edge.rstrip().split(","))

            assert node_id_graph_id_mapping[v1] == node_id_graph_id_mapping[
                v2], "Vertices in an edge must in the same graph !!!"

            graph_id = node_id_graph_id_mapping[v1]
            edge = (v1, v2)

            if graph_id in graph_id_edge_mapping.keys():
                graph_id_edge_mapping[graph_id] += [edge]
            else:
                graph_id_edge_mapping[graph_id] = [edge]

            if has_edge_labels:
                graph_id_edge_label_mapping[graph_id].update({edge: eid_to_edge_label_mapping[eid]})

    # node label (optional)
    has_node_labels = f"{dataset}_node_labels.txt" in os.listdir(os.path.join(data_directory, dataset))

    if has_node_labels:
        with open(os.path.join(data_directory, dataset, f"{dataset}_node_labels.txt"), mode="r") as f:
            # +1 since the index starts from 0
            node_id_label_mapping = {
                node_id + 1: int(label.rstrip()) for node_id, label in enumerate(f.readlines())}

    # graph label mapping
    with open(os.path.join(data_directory, dataset, f"{dataset}_graph_labels.txt"), mode="r") as f:

        # +1 since the index starts from 0
        graph_id_label_mapping = {graph_id + 1: int(label.rstrip()) for graph_id, label in enumerate(f.readlines())}

    # generating graphs
    graphs = []

    for graph_id, edge_list in graph_id_edge_mapping.items():
        graph = nx.Graph(edge_list, graph_id=graph_id, graph_label=graph_id_label_mapping[graph_id])

        nx.set_node_attributes(graph, {node: graph_id for node in graph_id_node_id_mapping[graph_id]}, "gid")

        if has_node_labels:
            node_labels = {node_id: node_id_label_mapping[node_id] for node_id in graph.nodes}
            nx.set_node_attributes(graph, node_labels, "label")

        if has_edge_labels:
            nx.set_edge_attributes(graph, graph_id_edge_label_mapping[graph_id], "label")

        graphs.append(graph)

    return graphs
