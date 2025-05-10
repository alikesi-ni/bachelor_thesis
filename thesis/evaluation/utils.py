import os
import string
from typing import List

import pandas as pd


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
            associated_steps = df[df["step"] <= step]["step"].tolist()
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
            associated_steps = df.iloc[:i + 1]["step"].tolist()
        else:
            associated_steps = df.iloc[h_grid[:h_grid.index(i) + 1]]["step"].tolist()
        parameter_associated_steps.append((i, associated_steps))

    return parameter_associated_steps

def get_feature_vector_matrix(data_dir_path: string,step_set: List[int]):
    fvm_dir_path = os.path.join(data_dir_path, "feature_vector_matrices")



def check_fvm_and_refinement_results(data_dir_path: string):
    fvm_dir_pat = os.path.join(data_dir_path, "fvm")
