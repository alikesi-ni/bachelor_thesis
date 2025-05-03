import time
from GWL_python.evaluation.evaluation_nested_cv import EvaluationNestedCV
from GWL_python.evaluation.evaluation_nested_cv_serial import evaluate_gwl_serial

if __name__ == "__main__":

    dataset_name = "MSRC_9"

    # -----------------------
    # PARALLEL version timing
    # -----------------------
    start_parallel = time.perf_counter()

    EvaluationNestedCV.evaluate_gwl(
        dataset_dir="../data",
        dataset_name=dataset_name,
        outer_folds=10,
        inner_folds=10,
        num_trials=1
    )

    end_parallel = time.perf_counter()
    print(f"Parallel version runtime: {end_parallel - start_parallel:.2f} seconds")

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

    # -----------------------
    # Comparison
    # -----------------------
    speedup = (end_serial - start_serial) / (end_parallel - start_parallel)
    print(f"Speedup factor (Serial / Parallel): {speedup:.2f}x")
