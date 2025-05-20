import os
import pickle
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from GWL_python.graph_dataset.graph_dataset import GraphDataset
from GWL_python.kernels.gwl_subtree import GWLSubtreeKernel
from GWL_python.evaluation.utils import generate_feature_vectors


def evaluate_gwl_serial(dataset_dir: str, dataset_name: str, outer_folds: int = 10, inner_folds: int = 10,
                        num_trials: int = 10):

    """
    Serial (non-parallel) version of evaluate_gwl for clarity.
    """

    #### 0️⃣ Prepare output folders ####

    main_dir = f"{dataset_name}-Evaluation-Serial-{int(datetime.now().timestamp())}"
    os.makedirs(main_dir, exist_ok=True)

    #### 1️⃣ Load dataset ####

    dataset = GraphDataset(dataset_dir, dataset_name)

    graph_id_label_mapping: dict = dataset.get_graphs_labels()
    graph_ids: np.array = np.fromiter(graph_id_label_mapping.keys(), int)
    graph_labels: np.array = np.fromiter(graph_id_label_mapping.values(), int)

    #### 2️⃣ Define search spaces ####

    gwl_search_space = {
        "refinement-steps": (4,), # tuple(range(11)),   # h
        "num-clusters": (3,), # tuple([2 ** i for i in range(1, 5)]),  # k
        "cluster-init-method": ("forgy",) #("kmeans++", "forgy")
    }
    gwl_param_combinations = [dict(zip(gwl_search_space.keys(), values))
                              for values in itertools.product(*gwl_search_space.values())]

    model_search_space = {
        "C": (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
    }
    model_param_combinations = [dict(zip(model_search_space.keys(), values))
                                for values in itertools.product(*model_search_space.values())]

    #### 3️⃣ Precompute feature vectors for all GWL combinations ####

    fv_dir = os.path.join(main_dir, "feature_vectors")
    os.makedirs(fv_dir, exist_ok=True)

    for gwl_params in gwl_param_combinations:
        fv_file = os.path.join(fv_dir, "-".join(map(str, gwl_params.values())))
        if not os.path.exists(fv_file):
            generate_feature_vectors(dataset, gwl_params, fv_dir)

    #### 4️⃣ Nested cross-validation ####

    results = []
    trial_accuracies = []

    for trial in range(num_trials):
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=1234)
        outer_fold_accuracies = []

        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(graph_ids, graph_labels)):

            x_train, y_train = graph_ids[train_idx], graph_labels[train_idx]
            x_test, y_test = graph_ids[test_idx], graph_labels[test_idx]

            ##### 4a INNER CV — hyperparameter search #####

            best_score = -1
            best_params = None

            for gwl_params in gwl_param_combinations:

                fv_file = os.path.join(fv_dir, "-".join(map(str, gwl_params.values())))
                with open(fv_file, "rb") as f:
                    feature_vectors = pickle.load(f)

                for model_params in model_param_combinations:

                    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=1234)
                    inner_accuracies = []

                    for inner_train_idx, inner_val_idx in inner_cv.split(x_train, y_train):
                        x_inner_train = x_train[inner_train_idx]
                        y_inner_train = y_train[inner_train_idx]
                        x_val = x_train[inner_val_idx]
                        y_val = y_train[inner_val_idx]

                        kernel = GWLSubtreeKernel(normalize=True)
                        K_train = kernel.fit_transform(feature_vectors, x_inner_train)
                        K_val = kernel.transform(feature_vectors, x_inner_train, x_val)

                        model = SVC(kernel="precomputed", cache_size=1500, max_iter=500, **model_params)
                        model.fit(K_train, y_inner_train)
                        y_pred = model.predict(K_val)
                        acc = accuracy_score(y_val, y_pred) * 100
                        inner_accuracies.append(acc)

                    avg_inner_acc = np.mean(inner_accuracies)

                    if avg_inner_acc > best_score:
                        best_score = avg_inner_acc
                        best_params = (gwl_params, model_params)

            ##### 4b OUTER TEST EVALUATION #####

            gwl_best_params, model_best_params = best_params

            fv_file = os.path.join(fv_dir, "-".join(map(str, gwl_best_params.values())))
            with open(fv_file, "rb") as f:
                feature_vectors = pickle.load(f)

            kernel = GWLSubtreeKernel(normalize=True)
            K_train = kernel.fit_transform(feature_vectors, x_train)
            K_test = kernel.transform(feature_vectors, x_train, x_test)

            model = SVC(kernel="precomputed", cache_size=1500, max_iter=500, **model_best_params)
            model.fit(K_train, y_train)
            y_pred = model.predict(K_test)
            outer_acc = accuracy_score(y_test, y_pred) * 100

            outer_fold_accuracies.append(outer_acc)

            result = {
                "Trial": trial,
                "Outer Fold": outer_fold,
                **gwl_best_params,
                **model_best_params,
                "Outer Test Accuracy": outer_acc
            }
            results.append(result)

        ##### 4c️⃣ Average outer fold accuracy for the trial #####
        trial_avg = np.round(np.mean(outer_fold_accuracies), 2)
        trial_accuracies.append({"Trial": trial + 1, "Average Accuracy": trial_avg})

    #### 5️⃣ Save results ####

    pd.DataFrame(results).to_csv(os.path.join(main_dir, "results_serial.csv"), sep=";", index=False)
    pd.DataFrame(trial_accuracies).to_csv(os.path.join(main_dir, "trial-accuracies_serial.csv"), sep=";", index=False)
