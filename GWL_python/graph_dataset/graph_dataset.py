import os
import networkx as nx


class GraphDataset:
    """
    Utility class for reading and loading the graph datasets (customized to TUDataset).

    Parameters
    ----------
    directory : str
        Directory, where the dataset is located

    dataset : str
        Name of the dataset to load

    Example
    ----------
    graphs = GraphDataset("./data", "PTC_FM")
    graphs.get_graphs()

    """

    def __init__(self, directory: str, dataset: str) -> None:
        self.directory = directory
        self.dataset = dataset

        self.__graphs = None

        self.__load_data()

    def get_graphs(self) -> dict:

        """
        Returns the loaded graphs.

        Returns
        -------
        out : dict[int, nx.Graph]
            A dictionary, where the keys are graph ids, and values are corresponding nx.Graph instance

        """

        assert self.__graphs is not None, "Error loading graphs !!!"

        return self.__graphs

    def get_graphs_as_disjoint_union(self) -> nx.Graph:

        """
        Returns the disjoint union of all the loaded graphs.

        Returns
        -------
        out : nx.Graph
            A nx.Graph instance

        """

        assert self.__graphs is not None, "Error loading graphs !!!"

        return nx.disjoint_union_all(self.__graphs.values())

    def get_graphs_labels(self) -> dict:

        """
        Returns label of the graphs.

        Returns
        -------
        out : dict
            A dictionary, where the key is the graph id, and value is the label of the corresponding graph

        """

        return {graph_id: graph.graph["graph_label"] for graph_id, graph in self.__graphs.items()}

    def __load_data(self) -> None:

        """
        Parses and loads graphs, including with their node and edge labels if available.

        """

        graph_id_node_id_mapping = dict()
        node_id_graph_id_mapping = dict()

        with open(os.path.join(self.directory, self.dataset, f"{self.dataset}_graph_indicator.txt"), mode="r") as f:

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
        has_edge_labels = f"{self.dataset}_edge_labels.txt" in os.listdir(os.path.join(self.directory, self.dataset))

        if has_edge_labels:
            graph_id_edge_label_mapping = {gid: dict() for gid in graph_id_node_id_mapping.keys()}

            with open(os.path.join(self.directory, self.dataset, f"{self.dataset}_edge_labels.txt"), mode="r") as f:
                eid_to_edge_label_mapping = {eid: int(label.rstrip()) for eid, label in enumerate(f.readlines())}

        with open(os.path.join(self.directory, self.dataset, f"{self.dataset}_A.txt"), mode="r") as f:

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
        has_node_labels = f"{self.dataset}_node_labels.txt" in os.listdir(os.path.join(self.directory, self.dataset))

        if has_node_labels:
            with open(os.path.join(self.directory, self.dataset, f"{self.dataset}_node_labels.txt"), mode="r") as f:
                # +1 since the index starts from 0
                node_id_label_mapping = {
                    node_id + 1: int(label.rstrip()) for node_id, label in enumerate(f.readlines())}

        # graph label mapping
        with open(os.path.join(self.directory, self.dataset, f"{self.dataset}_graph_labels.txt"), mode="r") as f:

            # +1 since the index starts from 0
            graph_id_label_mapping = {graph_id + 1: int(label.rstrip()) for graph_id, label in enumerate(f.readlines())}

        # generating graphs
        graphs = dict()

        for graph_id, edge_list in graph_id_edge_mapping.items():
            graph = nx.Graph(edge_list, graph_id=graph_id, graph_label=graph_id_label_mapping[graph_id])

            nx.set_node_attributes(graph, {node: graph_id for node in graph_id_node_id_mapping[graph_id]}, "gid")

            if has_node_labels:
                node_labels = {node_id: node_id_label_mapping[node_id] for node_id in graph.nodes}
                nx.set_node_attributes(graph, node_labels, "label")

            if has_edge_labels:
                nx.set_edge_attributes(graph, graph_id_edge_label_mapping[graph_id], "label")

            graphs[graph_id] = graph

        self.__graphs = graphs
