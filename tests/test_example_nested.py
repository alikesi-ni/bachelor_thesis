import time

import networkx as nx

from GWL_python.evaluation.evaluation_nested_cv import EvaluationNestedCV
from GWL_python.evaluation.evaluation_nested_cv_serial import evaluate_gwl_serial
from thesis.utils.read_data_utils import dataset_to_graphs
from thesis.utils.test_utils import evaluate_gwl_cv

if __name__ == "__main__":

    dataset_name = "MSRC_9"

    # # -----------------------
    # # PARALLEL version timing
    # # -----------------------
    # start_parallel = time.perf_counter()
    #
    # EvaluationNestedCV.evaluate_gwl(
    #     dataset_dir="../data",
    #     dataset_name=dataset_name,
    #     outer_folds=10,
    #     inner_folds=10,
    #     num_trials=1
    # )
    #
    # end_parallel = time.perf_counter()
    # print(f"Parallel version runtime: {end_parallel - start_parallel:.2f} seconds")
    #
    # -----------------------
    # SERIAL version timing
    # -----------------------
    start_serial = time.perf_counter()

    evaluate_gwl_serial(
        dataset_dir="../data",
        dataset_name=dataset_name,
        outer_folds=10,
        inner_folds=10,
        num_trials=1
    )

    end_serial = time.perf_counter()
    print(f"Serial version runtime: {end_serial - start_serial:.2f} seconds")

    start_own_serial = time.perf_counter()
    graphs = dataset_to_graphs("../data", dataset_name)
    disjoint_graph = nx.disjoint_union_all(graphs)
    graph_id_label_map = {g.graph["graph_id"]: g.graph["graph_label"] for g in graphs}

    h_grid = [4]
    k_grid = [3]

    c_grid = [10 ** i for i in range(-3, 4)]  # SVM C ∈ {1e-3 to 1e3}

    evaluate_gwl_cv(disjoint_graph, graph_id_label_map, h_grid, k_grid, c_grid, dataset_name=dataset_name, folds=10, repeats=1, start_repeat=1)

    end_own_serial = time.perf_counter()
    print(f"Own serial version runtime: {end_own_serial - start_own_serial:.2f} seconds")
