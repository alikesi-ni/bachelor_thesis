import networkx as nx
from pathlib import Path

"""
adapted from https://github.com/mlai-bonn/GenWL
"""


def graph_data_to_graph_list(path) -> [nx.Graph]:
    splits = path.split('/')
    db = splits[-1]

    # return variables
    graph_list = []

    # open the data files and read first line
    edge_file = open(path + "/" + db + "_A.txt", "r")
    edge = edge_file.readline().strip().split(",")

    # graph indicator
    graph_indicator = open(path + "/" + db + "_graph_indicator.txt", "r")
    graph = graph_indicator.readline()

    # graph labels
    graph_label_file = open(path + "/" + db + "_graph_labels.txt", "r")
    graph_label = graph_label_file.readline()

    # node labels
    node_labels = False
    if Path(path + "/" + db + "_node_labels.txt").is_file():
        node_label_file = open(path + "/" + db + "_node_labels.txt", "r")
        node_labels = True
        node_label = node_label_file.readline()

    # edge labels
    edge_labels = False
    if Path(path + "/" + db + "_edge_labels.txt").is_file():
        edge_label_file = open(path + "/" + db + "_edge_labels.txt", "r")
        edge_labels = True
        edge_label = edge_label_file.readline()

    # go through the data and read out the graphs
    node_counter = 1
    # all node_id will start with 0 for all graphs
    node_id_subtractor = 1
    while graph_label:
        G = nx.Graph()
        old_graph = graph
        new_graph = False

        # read out one complete graph
        while not new_graph and edge:
            # set all node labels with possibly node attributes
            while max(int(edge[0]), int(edge[1])) >= node_counter and not new_graph:
                if graph == old_graph:
                    if node_labels:
                        G.add_node(node_counter - node_id_subtractor, label=int(node_label))
                        node_label = node_label_file.readline()
                    else:
                        G.add_node(node_counter - node_id_subtractor)
                    node_counter += 1
                    graph = graph_indicator.readline()
                else:
                    old_graph = graph
                    new_graph = True
                    node_id_subtractor = node_counter

            if not new_graph:
                # set edge with possibly edge label and attributes and get next line
                if edge_labels:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor,
                               label=int(edge_label))
                    edge_label = edge_label_file.readline()
                else:
                    G.add_edge(int(edge[0]) - node_id_subtractor, int(edge[1]) - node_id_subtractor)

                # get new edge
                edge = edge_file.readline()
                if edge:
                    edge = edge.strip().split(",")

        G.graph['label'] = int(graph_label)

        # add graph to list
        graph_list.append(G)

        graph_label = graph_label_file.readline()

    # close all files
    edge_file.close()
    graph_indicator.close()
    graph_label_file.close()

    if node_labels:
        node_label_file.close()
    if edge_labels:
        edge_label_file.close()

    # returns list of the graphs of the db
    return graph_list
