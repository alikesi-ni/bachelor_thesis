import os
import pickle
import string
from typing import List

import pandas as pd
from scipy.sparse import hstack


def pick_steps_q_half(data_dir_path: string, q_strictly_descending: bool = True, include_inbetween_steps: bool = False):
    fvm_dir_path = os.path.join(data_dir_path, "feature_vector_matrices")
    refinement_results_file_path = os.path.join(data_dir_path, "refinement_results.csv")
    assert os.path.exists(fvm_dir_path)
    assert os.path.isfile(refinement_results_file_path)
    df = pd.read_csv(refinement_results_file_path)

    if q_strictly_descending:
        filtered_rows = [df.iloc[0]]
        last_q = df.loc[0, "max_q_error"]

        for i in range(1, len(df)):
            current_q = df.loc[i, "max_q_error"]
            if current_q < last_q:  # strictly descending
                filtered_rows.append(df.iloc[i])
                last_q = current_q

        df = pd.DataFrame(filtered_rows)

    base_q = df.loc[df["step"] == 0, "max_q_error"].values[0]
    threshold = base_q / 2
    anchor_steps = [0]

    for _, row in df.iterrows():
        if row["max_q_error"] <= threshold:
            anchor_steps.append(int(row["step"]))
            threshold /= 2

    parameter_associated_steps = []
    for i, step in enumerate(anchor_steps):
        if include_inbetween_steps:
            associated_steps = df.iloc[:i + 1]["step"].astype(int).tolist()
        else:
            associated_steps = anchor_steps[:i + 1]
        parameter_associated_steps.append((i, associated_steps))  # i represents q = base_q / 2^i

    return parameter_associated_steps

def pick_steps_h_grid(data_dir_path: string, h_grid: List[int], q_strictly_descending: bool = False, include_inbetween_steps: bool = True):
    fvm_dir_path = os.path.join(data_dir_path, "feature_vector_matrices")
    refinement_results_file_path = os.path.join(data_dir_path, "refinement_results.csv")
    assert os.path.exists(fvm_dir_path)
    assert os.path.isfile(refinement_results_file_path)
    df = pd.read_csv(refinement_results_file_path)

    if q_strictly_descending:
        filtered_rows = [df.iloc[0]]
        last_q = df.loc[0, "max_q_error"]

        for i in range(1, len(df)):
            current_q = df.loc[i, "max_q_error"]
            if current_q < last_q:  # strictly descending
                filtered_rows.append(df.iloc[i])
                last_q = current_q

        df = pd.DataFrame(filtered_rows)

    parameter_associated_steps = []
    for i in h_grid:
        if i >= len(df):
            continue  # skip invalid indices
        if include_inbetween_steps:
            associated_steps = df.iloc[:i + 1]["step"].astype(int).tolist()
        else:
            associated_steps = df.iloc[h_grid[:h_grid.index(i) + 1]]["step"].astype(int).tolist()
        parameter_associated_steps.append((i, associated_steps))

    return parameter_associated_steps

def stitch_feature_vectors(data_dir_path: string, associated_steps: List[int]):
    fvm_dir_path = os.path.join(data_dir_path, "feature_vector_matrices")

    fvm = None

    for step in associated_steps:
        fv_filename = os.path.join(fvm_dir_path, f"step_{step}.pkl")
        with open(fv_filename, "rb") as f:
            single_step_fvm, _ = pickle.load(f)

        if fvm is None:
            fvm = single_step_fvm
        else:
            fvm = hstack([fvm, single_step_fvm])

    return fvm.tocsr()


def generate_report(data_dir_path: str, output_dir: str):
    test_results_path = os.path.join(data_dir_path, "test_results.csv")
    df = pd.read_csv(test_results_path)

    # count frequency of each param
    param_counts = df["param"].value_counts().sort_index()
    total = param_counts.sum()
    param_freq_section = ["# Parameter Frequency"]
    for param, count in param_counts.items():
        param_freq_section.append(f"param={param}: {count} ({count / total:.2%})")

    # compute average accuracy per trial
    trial_group = df.groupby("trial")["accuracy"]
    trial_averages = trial_group.mean().sort_index()
    per_trial_section = ["\n# Per-Trial Accuracy Averages"]
    for trial, avg in trial_averages.items():
        per_trial_section.append(f"trial {trial}: {avg:.2f}")

    # compute overall mean and std of trial averages
    mean_of_means = trial_averages.mean()
    std_of_means = trial_averages.std()
    summary_section = [
        "\n# Summary",
        f"Average of trial averages: {mean_of_means:.2f}",
        f"Standard deviation of averages: {std_of_means:.2f}"
    ]

    # combine and save to report.txt
    report_lines = param_freq_section + per_trial_section + summary_section
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return mean_of_means, std_of_means
