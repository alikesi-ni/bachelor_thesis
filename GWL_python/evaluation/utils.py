import os
import pickle

import networkx as nx

from GWL_python.graph_dataset.graph_dataset import GraphDataset
from GWL_python.gwl.gwl import GradualWeisfeilerLeman


def generate_feature_vectors(dataset: GraphDataset, gwl_params: dict, fv_dir: str) -> None:
    """
    Generates and persists feature vectors for a given GWL parameter combination.

    Parameters
    ----------
    dataset: GraphDataset
        Directory where the instance of pickled GraphDataset is located

    gwl_params: dict
        GWL parameter configuration for which the feature vectors will be generated

    fv_dir: str
        Directory where the generated feature vectors will be persisted

    """

    graphs: nx.Graph = dataset.get_graphs_as_disjoint_union()

    gwl = GradualWeisfeilerLeman(
        refinement_steps=gwl_params["refinement-steps"],
        n_cluster=gwl_params["num-clusters"],
        cluster_initialization_method=gwl_params["cluster-init-method"]
    )

    gwl.refine_color(graphs)

    feature_vectors = gwl.generate_feature_vectors(graphs)

    with open(os.path.join(fv_dir, "-".join(map(str, gwl_params.values()))), mode="wb") as fv_file:
        pickle.dump(feature_vectors, fv_file)
