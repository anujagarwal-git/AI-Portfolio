# Traditional Machine Learning

Classic machine learning on banking problems, built with scikit-learn pipelines and XGBoost. Each project has its own folder with a README, the notebook(s) and the data.

| # | Project | Business question | Best result |
|---|---|---|---|
| 01 | [Loan Approval Prediction](01-loan-approval-prediction/) | Will this home-loan application be approved? | F1 0.90 (test) |
| 02 | [Term Deposit Propensity Model](02-bank-marketing-propensity/) | Which customers will buy a term deposit after a call? | F1 0.44, AUC 0.74 (test) |

## Techniques covered

- Logistic regression, decision trees, random forest, XGBoost, linear / ridge / lasso regression
- Feature selection: correlation, Lasso, RFE, chi-square, mutual information, VIF, permutation importance, SHAP
- Pipelines with `ColumnTransformer` (imputation, encoding, scaling inside cross-validation)
- Class imbalance: class weights, `scale_pos_weight`, decision-threshold tuning
- Model evaluation: precision, recall, F1, ROC-AUC, cross-validation, grid search

## Setup

```bash
pip install pandas numpy scikit-learn xgboost shap statsmodels matplotlib seaborn jupyter
```

Open a notebook from inside its project folder, so the relative `data/` paths resolve.
