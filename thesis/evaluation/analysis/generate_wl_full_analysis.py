import os
import pandas as pd

def generate_wl_full_analysis(dataset_root_dir):
    h_grid_base = os.path.join(dataset_root_dir, "cosine")
    refinement_path = os.path.join(dataset_root_dir, "refinement_results.csv")
    output_path = os.path.join(dataset_root_dir, "full_analysis.csv")

    if not os.path.exists(refinement_path):
        print(f"[!] Missing refinement_results.csv in {dataset_root_dir}")
        return

    h_grid_folder = None
    for folder in os.listdir(h_grid_base):
        if folder.startswith("h_grid__"):
            h_grid_folder = os.path.join(h_grid_base, folder)
            break

    if not h_grid_folder:
        print(f"[!] No h_grid__ folder found in {h_grid_base}")
        return

    test_path = os.path.join(h_grid_folder, "test_results.csv")
    if not os.path.exists(test_path):
        print(f"[!] Missing test_results.csv in {h_grid_folder}")
        return

    try:
        df_test = pd.read_csv(test_path)
        df_ref = pd.read_csv(refinement_path).sort_values("step")

        results = []

        for _, row in df_test.iterrows():
            trial = int(row["trial"])
            fold = int(row["fold"])
            h = int(row["param"])
            acc = float(row["accuracy"])

            row_h = df_ref[df_ref["step"] == h]
            if not row_h.empty:
                feature_dim = int(row_h.iloc[0]["feature_dim"])
                n_colors = int(row_h.iloc[0]["n_colors"])
                total_time = round(df_ref[df_ref["step"] <= h]["calculation_time_in_seconds"].sum(), 6)
            else:
                feature_dim = n_colors = total_time = None

            results.append({
                "trial": trial,
                "fold": fold,
                "h": h,
                "accuracy": acc,
                "feature_dim": feature_dim,
                "n_colors": n_colors,
                "total_time_seconds": total_time
            })

        df_out = pd.DataFrame(results)
        df_out = df_out[["trial", "fold", "h", "accuracy", "feature_dim", "n_colors", "total_time_seconds"]]
        df_out.to_csv(output_path, index=False)
        print(f"[✓] Saved full_analysis.csv to {output_path}")

    except Exception as e:
        print(f"[!] Error processing {dataset_root_dir}: {e}")


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
    dataset_dir = os.path.join(base_path, f"WL-{dataset}")
    if os.path.isdir(dataset_dir):
        print(f"\n[+] Processing {dataset}")
        generate_wl_full_analysis(dataset_dir)
    else:
        print(f"[!] Skipping missing dataset: {dataset_dir}")