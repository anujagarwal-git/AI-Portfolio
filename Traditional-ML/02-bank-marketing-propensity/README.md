# Term Deposit Propensity Model

Predicts which bank customers will subscribe to a term deposit after a marketing call, so a campaign can call the most likely buyers first.

## Data

`data/bank-full.csv` — [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) data set: 45,211 customers of a Portuguese bank, of whom 5,289 (11.7%) subscribed. Semicolon-separated.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Exploratory analysis in 8 steps: class imbalance, the `pdays = -1` "never contacted" segment, subscription rate by category, outliers, correlations, campaign contact patterns, unknown values |
| `02_modelling.ipynb` | Feature selection, three classifiers, threshold tuning, and an exploratory revenue regression |

## Key design decision — no leakage from call duration

`duration` (length of the call) is the strongest feature in every ranking, but it is only known *after* the call has ended. A model that uses it cannot help decide whom to call. It is **left out of all models**.

## Approach

1. **Feature selection** — nine methods compared: correlation, decision-tree importance, Lasso, RFE, SHAP, mutual information, permutation importance, VIF and chi-square.
2. **Models** — logistic regression, decision tree and XGBoost in scikit-learn pipelines, with class weights for the imbalanced target and 5-fold cross-validated grid search.
3. **Threshold tuning** — the default 0.5 threshold flags too many customers. The best threshold (0.63) was found on the precision-recall curve and confirmed with `TunedThresholdClassifierCV` on training data only.

## Results (test set, 9,043 customers)

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.50 | 0.78 | 0.29 | 0.57 | 0.38 |
| Logistic regression | 0.63 | 0.87 | 0.46 | 0.41 | 0.43 |
| Decision tree, tuned | 0.50 | — | — | — | 0.43–0.44 |
| XGBoost, tuned | 0.50 | 0.84 | 0.38 | 0.52 | **0.44** |

Logistic regression ROC-AUC: 0.74. The decision tree has no fixed random seed, so its F1 varies slightly between runs.

Threshold tuning lifted logistic regression F1 from 0.38 to 0.43, close to XGBoost, so the simpler and more explainable model is a reasonable choice.

**Part 2 (exploratory):** two regressions to estimate expected revenue reached R² ≈ 0.05: customer demographics explain very little of account balance. This is kept in the notebook as a record of what was tried.
