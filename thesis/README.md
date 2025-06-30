# Bachelor's Thesis
**Topic:** Graph Classification Using Kernels Based on Quasi-Stable Coloring

**Student:**
- Alexander Nittmann

**Supervisor:**
- Assoz. Prof. Dipl-Inf. Dr. Nils Kriege

---

## Getting Started

### Prerequisites

- Python version: 3.12.7

### Install Dependencies

Run pip install -r requirements.txt to install the required dependencies.

---

## How to Use

### Dataset Setup

- The dataset PTC_FM is already included in the repository.
- Other datasets must be downloaded individually and placed in the same TUDataset format.

### Step 1: Run Quasi-Stable Coloring Refinement

To refine a dataset, open the script test_qsc_refinement.py and uncomment the desired dataset.
By default, the refinement process continues until the number of colors exceeds 4096 or until maxErr = 0 is reached.

When completed, a folder evaluation-results/QSC-{dataset_name}/ will be created, containing:
- evaluation_log.txt
- refinement_results.csv

The file refinement_results.csv contains useful metrics such as the number of refinement steps, maximum error, number of colors, number of witness pairs, runtime, and more.

### Step 2: Run 10-Fold Nested Cross-Validation

To evaluate the model, open the script test_qsc_evaluation.py and uncomment the desired dataset.
After execution, a folder is created for each strategy used, containing:
- evaluation_log.txt
- train_results.csv
- test_results.csv

Note: It is recommended not to use the report.txt files, as the next step provides a clearer analysis.

### Step 3: Generate Full Analysis

To perform a full analysis of the evaluation results, run the script generate_qsc_full_analysis.py and uncomment the desired dataset.
This generates a detailed report showing the best parameter settings per trial and fold, the strategies used, and which refinement steps were filtered out.

---

## Additional Information

Files and methods also exist for the Gradual Weisfeiler-Leman (GWL) and Weisfeiler-Leman (WL) approaches.
Feel free to experiment with those variants.

---

## Noteworthy Files

- quasi_stable_coloring.py: Contains the main implementation of the adapted Rothko algorithm for Quasi-Stable Coloring.
- examples/: This folder includes scripts to visualize how different simple graphs are colored across refinement iterations. Running the scripts displays the graphs and their colorings. You can also create your own custom graph using NetworkX and run the algorithm on it.
