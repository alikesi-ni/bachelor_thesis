import os
import pandas as pd

def generate_gwl_full_analysis(dataset_root_dir):
    all_results = []
    trial_fold_set = set()

    init_method = "kmeans++"
    init_path = os.path.join(dataset_root_dir, init_method)
    if not os.path.isdir(init_path):
        print(f"[!] Missing '{init_method}' directory in {dataset_root_dir}")
        return

    for k_dir in os.listdir(init_path):
        if not k_dir.startswith("k__"):
            continue
        cosine_dir = os.path.join(init_path, k_dir, "cosine")
        if not os.path.isdir(cosine_dir):
            continue

        for h_grid_folder in os.listdir(cosine_dir):
            if not h_grid_folder.startswith("h_grid__"):
                continue
            train_path = os.path.join(cosine_dir, h_grid_folder, "train_results.csv")
            if os.path.exists(train_path):
                df_train = pd.read_csv(train_path)
                trial_fold_set.update(set(zip(df_train['trial'], df_train['fold'])))

    for trial, fold in sorted(trial_fold_set):
        best_acc = -1
        best_info = None

        for k_dir in os.listdir(init_path):
            if not k_dir.startswith("k__"):
                continue
            k_value = k_dir.replace("k__", "")
            cosine_dir = os.path.join(init_path, k_dir, "cosine")
            if not os.path.isdir(cosine_dir):
                continue

            for h_grid_folder in os.listdir(cosine_dir):
                if not h_grid_folder.startswith("h_grid__"):
                    continue

                folder_path = os.path.join(cosine_dir, h_grid_folder)
                train_path = os.path.join(folder_path, "train_results.csv")
                test_path = os.path.join(folder_path, "test_results.csv")

                if not os.path.exists(train_path) or not os.path.exists(test_path):
                    continue

                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)

                df_filtered = df_train[(df_train['trial'] == trial) & (df_train['fold'] == fold)]
                if df_filtered.empty:
                    continue

                best_row = df_filtered.loc[df_filtered['accuracy'].idxmax()]
                if best_row['accuracy'] > best_acc:
                    best_acc = best_row['accuracy']
                    best_param = int(best_row['param'])

                    df_test_filtered = df_test[
                        (df_test['trial'] == trial) &
                        (df_test['fold'] == fold) &
                        (df_test['param'] == best_param)
                    ]
                    if df_test_filtered.empty:
                        continue

                    test_acc = float(df_test_filtered.iloc[0]['accuracy'])

                    refinement_path = os.path.join(init_path, f"k__{k_value}", "refinement_results.csv")
                    if os.path.exists(refinement_path):
                        try:
                            df_ref = pd.read_csv(refinement_path)
                            df_ref = df_ref.sort_values("step")
                            row_h = df_ref[df_ref["step"] == best_param]
                            if not row_h.empty:
                                feature_dim = int(row_h.iloc[0]["feature_dim"]) if "feature_dim" in row_h.columns else None
                                n_colors = int(row_h.iloc[0]["n_colors"]) if "n_colors" in row_h.columns else None
                                total_time = round(df_ref[df_ref["step"] <= best_param]["calculation_time_in_seconds"].sum(), 6)
                            else:
                                feature_dim = n_colors = total_time = None
                        except Exception as e:
                            print(f"[!] Error reading refinement step={best_param}: {e}")
                            feature_dim = n_colors = total_time = None
                    else:
                        feature_dim = n_colors = total_time = None

                    best_info = {
                        "trial": trial,
                        "fold": fold,
                        "k": int(k_value),
                        "h": best_param,
                        "accuracy": round(test_acc, 4),
                        "feature_dim": feature_dim,
                        "n_colors": n_colors,
                        "total_time_seconds": total_time
                    }

        if best_info:
            all_results.append(best_info)

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out = df_out[["trial", "fold", "k", "h", "accuracy", "feature_dim", "n_colors", "total_time_seconds"]]
        output_path = os.path.join(dataset_root_dir, "full_analysis.csv")
        df_out.to_csv(output_path, index=False)
        print(f"[✓] Saved full_analysis.csv to {output_path}")
    else:
        print(f"[!] No valid results found for {dataset_root_dir}")


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
    dataset_dir = os.path.join(base_path, f"GWL-{dataset}")
    if os.path.isdir(dataset_dir):
        print(f"\n[+] Processing {dataset}")
        generate_gwl_full_analysis(dataset_dir)
    else:
        print(f"[!] Skipping missing dataset: {dataset_dir}")
