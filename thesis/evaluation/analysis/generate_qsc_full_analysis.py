import os
import re
import pandas as pd


def parse_param_steps(evaluation_log_path):
    param_steps = {}

    if not os.path.exists(evaluation_log_path):
        return param_steps

    with open(evaluation_log_path, "r") as f:
        for line in f:
            match = re.match(r".*param=(\d+)\s+->\s+steps=\[(.*?)\]", line)
            if match:
                param = int(match.group(1))
                steps_str = match.group(2)
                steps = [int(s.strip()) for s in steps_str.split(",") if s.strip().isdigit()]
                param_steps[param] = steps

    return param_steps


def generate_qsc_full_analysis(cosine_dir):
    kernel_dir = cosine_dir  # e.g., QSC-DATASET/cosine
    dataset_dir = os.path.dirname(kernel_dir)  # QSC-DATASET
    refinement_path = os.path.join(dataset_dir, "refinement_results.csv")
    output_path = os.path.join(dataset_dir, "full_analysis.csv")

    all_subfolders = [
        f for f in os.listdir(kernel_dir)
        if os.path.isdir(os.path.join(kernel_dir, f)) and (
            f.startswith("q_ratio__") or f.startswith("h_grid__"))
    ]

    if not os.path.exists(refinement_path):
        print(f"[!] Missing refinement_results.csv in {dataset_dir}")
        return

    df_ref = pd.read_csv(refinement_path)
    trial_fold_set = set()
    all_results = []

    for folder in all_subfolders:
        train_path = os.path.join(kernel_dir, folder, "train_results.csv")
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            trial_fold_set.update((int(t), int(f)) for t, f in zip(df_train["trial"], df_train["fold"]))

    for trial, fold in sorted(trial_fold_set):
        best_acc = -1
        best_folder = None
        best_param = None

        for folder in all_subfolders:
            train_file = os.path.join(kernel_dir, folder, "train_results.csv")
            test_file = os.path.join(kernel_dir, folder, "test_results.csv")

            if not os.path.exists(train_file) or not os.path.exists(test_file):
                continue

            df_train = pd.read_csv(train_file)

            df_filtered = df_train[(df_train["trial"] == trial) & (df_train["fold"] == fold)]
            if df_filtered.empty:
                continue

            best_row = df_filtered.loc[df_filtered["accuracy"].idxmax()]
            if best_row["accuracy"] > best_acc:
                best_acc = best_row["accuracy"]
                best_folder = folder
                best_param = int(best_row["param"])

        if best_folder:
            test_path = os.path.join(kernel_dir, best_folder, "test_results.csv")
            df_test = pd.read_csv(test_path)
            test_row = df_test[
                (df_test["trial"] == trial) &
                (df_test["fold"] == fold) &
                (df_test["param"] == best_param)
            ]
            if test_row.empty:
                continue
            test_acc = float(test_row.iloc[0]["accuracy"])

            eval_log_path = os.path.join(kernel_dir, best_folder, "evaluation_log.txt")
            param_to_steps = parse_param_steps(eval_log_path)
            steps = param_to_steps.get(best_param, [])

            if steps:
                highest_step = max(steps)
                relevant_rows = df_ref[df_ref["step"].isin(steps)]
                feature_dim = int(relevant_rows["partition_count"].sum())
                n_colors = int(df_ref[df_ref["step"] == highest_step]["partition_count"].values[0])
                total_time = round(df_ref[df_ref["step"] <= highest_step]["calculation_time_in_seconds"].sum(), 6)
            else:
                feature_dim = n_colors = total_time = None

            if best_folder.startswith("q_ratio__"):
                category = "q_ratio"
                subtype = best_folder.split("__")[1]
            elif best_folder.startswith("h_grid__"):
                category = "h_grid"
                if "desc" in best_folder:
                    subtype = "desc"
                elif "no_desc" in best_folder:
                    subtype = "no_desc"
                else:
                    subtype = "plain"
            else:
                category = subtype = None

            all_results.append({
                "trial": trial,
                "fold": fold,
                "category": category,
                "subtype": subtype,
                "param": best_param,
                "steps": str(steps),
                "accuracy": round(test_acc, 4),
                "feature_dim": feature_dim,
                "n_colors": n_colors,
                "total_time_seconds": total_time
            })

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out = df_out[
            ["trial", "fold", "category", "subtype", "param", "steps",
             "accuracy", "feature_dim", "n_colors", "total_time_seconds"]
        ]
        df_out.to_csv(output_path, index=False)
        print(f"[✓] Saved full_analysis.csv to {output_path}")
    else:
        print(f"[!] No results found to save in {dataset_dir}")


dataset_names = [
    # # small datasets
    # "KKI",
    "PTC_FM",
    # "MSRC_9",
    # "MUTAG",
    #
    # # large datasets
    # "COLLAB",
    # "DD",
    # "REDDIT-BINARY",
    #
    # # medium datsets
    # "IMDB-BINARY",
    # "NCI1",
    #
    # # social network datasets
    # "EGO-1",
    # "EGO-2",
    # "EGO-3",
    # "EGO-4",
    #
    # # new datasets
    # "ENZYMES",
    # "PROTEINS",
]

base_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "evaluation-results"))
for dataset in dataset_names:
    for sim_type in ["cosine"]:
        dataset_dir = os.path.join(base_path, f"QSC-{dataset}", sim_type)
        if os.path.isdir(dataset_dir):
            print(f"\n[+] Processing QSC-{dataset} [{sim_type}]")
            generate_qsc_full_analysis(dataset_dir)
            break  # Stop after the first one found
    else:
        print(f"[!] Skipping: QSC-{dataset} — no 'cosine' found")
