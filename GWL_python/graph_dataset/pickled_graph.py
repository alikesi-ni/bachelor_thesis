import os
import pickle
from graph_dataset.graph_dataset import GraphDataset


def pickle_graph_datasets(data_directory: str, destination_directory: str, verbose: bool = True) -> None:
    """
    Pickles all the datasets present in the given data directory.

    Parameters
    ----------
    data_directory : str
        Directory, where the datasets are located

    destination_directory : str
        Directory, where the pickled datasets should be stored

    verbose : bool
        For debugging (default: True)

    """

    datasets = [dataset for dataset in os.listdir(data_directory) if
                os.path.isdir(os.path.join(data_directory, dataset))]

    if verbose:
        print(f"{len(datasets)} datasets found: {datasets}")

    for dataset in datasets:
        pickle_graph_dataset(data_directory, destination_directory, dataset)

        if verbose:
            print(f"{dataset} dataset pickled successfully as an GraphDataset instance.")


def pickle_graph_dataset(data_directory: str, destination_directory: str, dataset: str) -> None:
    """
    Pickles the given dataset.

    Parameters
    ----------
    data_directory : str
        Directory, where the dataset is located

    destination_directory : str
        Directory, where the pickled dataset should be stored

    dataset : str
        Name of the dataset to pickle

    """

    with open(os.path.join(destination_directory, dataset), 'wb') as f:
        pickle.dump(GraphDataset(data_directory, dataset), f)
