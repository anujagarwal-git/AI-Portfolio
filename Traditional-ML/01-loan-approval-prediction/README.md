# Loan Approval Prediction

Predicts whether a home-loan application is approved, using the applicant's income, credit history, employment and property details.

## Data

`data/loan_data.csv` — 614 loan applications (422 approved, 192 rejected). Public "Loan Prediction" practice data set.

## What the notebook does

1. Combines applicant and co-applicant income into one household income feature.
2. Explores correlations and a quick decision-tree importance check.
3. Trains three models inside scikit-learn pipelines (imputation and encoding happen inside the pipeline):
   - Logistic regression on just three features (income, credit history, self-employment)
   - Decision tree
   - XGBoost
4. Tunes each model with 5-fold cross-validated grid search.

## Results (test set, 123 applications)

| Model | Accuracy | F1 |
|---|---|---|
| Logistic regression (3 features) | 0.81 | 0.88 |
| Decision tree, max depth 3 | 0.85 | 0.90 |
| XGBoost, tuned | — | 0.90 |

**Credit history is by far the strongest driver of approval.** A three-feature logistic regression is nearly as good as the tree models and is much easier to explain. With only 123 test rows, the gaps between models are within sampling noise.

## Files

| File | Contents |
|---|---|
| `loan_approval_prediction.ipynb` | Full analysis and models |
| `data/loan_data.csv` | Input data |
