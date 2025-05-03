import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from GWL_python.graph_dataset.graph_dataset import GraphDataset
from GWL_python.kernels.gwl_subtree import GWLSubtreeKernel

from GWL_python.evaluation.utils import *


class EvaluationNestedCV:
    """Utility class for evaluating the accuracy of GWL on a given dataset using nested cross-validation"""

    @staticmethod
    def evaluate_hyperparameters(graph_ids: np.array, graph_labels: np.array, gwl_param_grid: dict,
                                 model_param_grid: list[dict], inner_folds: int, fv_dir: str) -> list[dict]:

        """
        Evaluates a given hyperparameter combination on GWL.

        Parameters
        ----------
        graph_ids : np.array
            Ids of the graphs to be used for hyperparameter evaluation

        graph_labels : np.array
            Labels of the graphs to be used for hyperparameter evaluation

        gwl_param_grid: dict
            Parameters to use for GWL evaluation

        model_param_grid: list[dict]
            Parameters of the SVM model for evaluation

        inner_folds: int
            Number of folds to be used for cross-validation (inner folds of the nested fold)

        fv_dir: str
            Directory of the persisted feature vectors

        Returns
        -------
        out : list[dict]
            A list of dictionaries, where each dictionary contain the parameters used and the average accuracy from CV

        """

        # load feature vectors generated for the given GWL parameter combination
        with open(os.path.join(fv_dir, "-".join(map(str, gwl_param_grid.values()))), mode="rb") as fv_file:

            feature_vectors = pickle.load(fv_file)

        results = list()

        for model_params in model_param_grid:

            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=1234)

            accuracies = list()

            for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(graph_ids, graph_labels)):
                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_val, y_val = graph_ids[val_idx], graph_labels[val_idx]

                kernel = GWLSubtreeKernel(normalize=True)
                k_train = kernel.fit_transform(feature_vectors, x_train)
                k_test = kernel.transform(feature_vectors, x_train, x_val)

                model = SVC(kernel="precomputed", cache_size=1500, max_iter=500, **model_params)
                model.fit(k_train, y_train)
                y_predictions = model.predict(k_test)

                # classification accuracy
                accuracy = np.round(accuracy_score(y_val, y_predictions) * 100, 2)

                accuracies.append(accuracy)

            combination = gwl_param_grid.copy()
            combination.update(model_params)
            combination.update({"average-accuracy": np.round(np.mean(accuracies), 2)})

            results.append(combination)

        return results

    @staticmethod
    def evaluate_gwl(dataset_dir: str, dataset_name: str, outer_folds: int = 10, inner_folds: int = 10,
                     num_trials: int = 10) -> None:

        """
        Evaluates GWL on a given dataset and persists the results.

        Parameters
        ----------
        dataset_dir: str
            Directory where the instance of pickled GraphDataset is located

        dataset_name: str
            Name of the pickled dataset to load

        outer_folds: int
            Number of outer folds of the nested fold

        inner_folds: int
            Number of inner folds of the nested fold

        num_trials: int = 10
            Number of trials/ repetitions to perform

        """

        # create directory
        main_dir = f"{dataset_name}-Evaluation-{int(datetime.now().timestamp())}"

        if main_dir not in os.listdir():
            os.mkdir(main_dir)

        # load GraphDataset instance
        # with open(os.path.join(dataset_dir, dataset_name), mode="rb") as pickled_data:
        #     dataset: GraphDataset = pickle.load(pickled_data)
        dataset = GraphDataset("../data", dataset_name)

        graph_id_label_mapping: dict = dataset.get_graphs_labels()

        graph_ids: np.array = np.fromiter(graph_id_label_mapping.keys(), int)
        graph_labels: np.array = np.fromiter(graph_id_label_mapping.values(), int)

        # define search spaces
        gwl_search_space = {
            "refinement-steps": (4,),
            "num-clusters": (3,),
            "cluster-init-method": ("forgy",)
        }
        gwl_permuted_search_space = [dict(zip(gwl_search_space.keys(), elem)) for elem in
                                     itertools.product(*gwl_search_space.values())]

        num_gwl_params = len(gwl_permuted_search_space)

        model_search_space = {
            "C": (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
        }
        model_permuted_search_space = [dict(zip(model_search_space.keys(), elem)) for elem in
                                       itertools.product(*model_search_space.values())]

        all_params_keys = list(gwl_search_space.keys()) + list(model_search_space.keys())

        # generate and persist feature vectors for each configuration
        fv_dir = os.path.join(main_dir, "feature_vectors")

        if "feature_vectors" not in os.listdir(main_dir):
            os.mkdir(fv_dir)

        with ProcessPoolExecutor() as executor:
            executor.map(generate_feature_vectors,
                         [dataset] * num_gwl_params, gwl_permuted_search_space, [fv_dir] * num_gwl_params)

        results = list()
        trial_accuracies = list()

        # nested CV
        for num_trial in range(num_trials):

            outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=1234)

            outer_fold_accuracies = list()

            for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels)):

                x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
                x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

                # obtain the best hyperparameter combination using inner cross-validation
                combinations = list()

                with ProcessPoolExecutor() as executor:

                    for combination in executor.map(
                            EvaluationNestedCV.evaluate_hyperparameters,
                            [x_train] * num_gwl_params,
                            [y_train] * num_gwl_params,
                            gwl_permuted_search_space,
                            [model_permuted_search_space] * num_gwl_params,
                            [inner_folds] * num_gwl_params,
                            [fv_dir] * num_gwl_params):
                        combinations.extend(combination)

                best_combination = max(combinations, key=lambda x: x["average-accuracy"])

                # load the feature vectors for best combination
                best_fv_file = os.path.join(fv_dir,
                                            "-".join([str(best_combination[key]) for key in gwl_search_space.keys()]))

                with open(best_fv_file, mode="rb") as fv_file:
                    feature_vectors = pickle.load(fv_file)

                kernel = GWLSubtreeKernel(normalize=True)
                k_train = kernel.fit_transform(feature_vectors, x_train)
                k_test = kernel.transform(feature_vectors, x_train, x_test)

                model = SVC(kernel="precomputed", cache_size=1500, max_iter=500,
                            **{key: best_combination[key] for key in model_search_space.keys()})

                model.fit(k_train, y_train)
                y_predictions = model.predict(k_test)

                # classification accuracy
                accuracy = np.round(accuracy_score(y_test, y_predictions) * 100, 2)

                outer_fold_accuracies.append(accuracy)

                result = {"Trial": num_trial, "Outer Fold": outer_fold}
                result.update({key: best_combination[key] for key in all_params_keys})
                result.update({"accuracy": accuracy})

                results.append(result)

            trial_accuracies.append(
                {"Trial": num_trial + 1, "Average-Accuracy": np.round(np.mean(outer_fold_accuracies), 2)})

        df = pd.DataFrame.from_records(results)
        df.to_csv(os.path.join(main_dir, "results.csv"), sep=";", index=False)

        df = pd.DataFrame.from_records(trial_accuracies)
        df.to_csv(os.path.join(main_dir, "trial-accuracies.csv"), sep=";", index=False)
