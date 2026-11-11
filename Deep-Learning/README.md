# Deep Learning

Feed-forward neural networks (TensorFlow / Keras, wrapped with SciKeras so they work inside scikit-learn pipelines) applied to two credit-risk problems.

| # | Project | Business question | Result (test set) |
|---|---|---|---|
| 01 | [Credit Card Fraud Detection](01-credit-card-fraud-detection/) | Is this card transaction fraud? (0.17% of transactions) | Precision 0.77, recall 0.74, F1 0.76 |
| 02 | [Credit Default (PD) Model](02-credit-default-pd-model/) | Will this personal loan default? (1.3M loans) — champion–challenger: XGBoost vs neural network | Neural network: AUC 0.70, KS 28.9% (XGBoost champion AUC 0.70) |

## Techniques covered

- Dense networks with batch normalisation, dropout and early stopping
- Extreme class imbalance: class weights, threshold tuning on out-of-fold predictions
- Leakage control: scaling inside cross-validation folds; removing look-ahead variables with credit judgement
- Credit-risk validation: ROC-AUC, KS statistic, PSI (in-time only, as OOT data is not available in the public sample; in production PSI would run on OOT data)
- Champion–challenger model comparison
- Explainability: SHAP on the XGBoost champion model

## Setup

```bash
pip install pandas numpy pyarrow scikit-learn tensorflow scikeras xgboost shap scipy matplotlib jupyter
```

The data sets are too large for GitHub. Each project's `data/README.md` says where to download them and where to put them. Open a notebook from inside its project folder so the relative `data/` paths resolve.
