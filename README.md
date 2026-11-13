# Empirical Evaluation of Reject Option Classification (ROC)

This repository contains the reproducible Python implementation and empirical evaluation of **Reject Option based Classification (ROC)** (Kamiran et al., 2012) using IBM's **AI Fairness 360 (`aif360`)** toolkit across three benchmark datasets: **German Credit**, **COMPAS**, and **Titanic**.

## Repository Contents
- `roc_fairness_evaluation.ipynb` — Complete Jupyter/Colab notebook with all code and saved execution outputs.
- `README.md` — Setup, dataset preparation, and reproduction instructions.

## Environment Setup & Installation

The experiments were executed in Python 3 using Google Colab. To reproduce the environment locally or in Colab, install the required packages:

pip install aif360 scikit-learn pandas numpy seaborn

### Dataset Setup
- **Titanic:** Loaded directly via `seaborn.load_dataset('titanic')` (pre-processed by dropping missing `age` values and one-hot encoding categorical predictors).
- **German Credit & COMPAS:** Loaded via `aif360.datasets` (`GermanDataset` and `CompasDataset`). If running locally for the first time, place the raw data files (`german.data` and `compas-scores-two-years.csv`) into the corresponding `aif360/data/raw/` directories as prompted by `aif360`.

## Experimental Pipeline
1. **Baseline Raw Metrics:** Computes Demographic Parity Difference (`DP_Diff`), Disparate Impact Ratio (`DIR`), and Consistency ($k=5$) directly on the unmodified ground-truth dataset labels.
2. **Stratified Splitting:** Splits each dataset into **60% Train / 20% Validation / 20% Test**, stratified jointly on the target label and sensitive attribute (`seed = 42`).
3. **Model Training & Post-Processing:** Standardizes features using `StandardScaler` fit strictly on the training split, trains a baseline `LogisticRegression` classifier (`max_iter = 1000`, `random_state = 42`), and fits `RejectOptionClassification` on the validation predictions (`num_class_thresh = 100`, `num_ROC_margin = 50`, `DP_Diff` constrained to `[-0.05, 0.05]`).
4. **Held-Out Evaluation:** Compares group fairness (`DP_Diff`, `DIR`), individual fairness (`Consistency`), and predictive performance (`Accuracy`, `Balanced Accuracy`, `F1-Score`) before and after ROC post-processing on the held-out test sets.

## References
- Kamiran, F., Karim, A., & Zhang, X. (2012). Decision theory for discrimination-aware classification. In *2012 IEEE 12th International Conference on Data Mining* (pp. 924–929). IEEE. https://doi.org/10.1109/ICDM.2012.45
