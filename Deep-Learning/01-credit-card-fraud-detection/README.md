# Credit Card Fraud Detection — Neural Network

Flags fraudulent card transactions where fraud is only 0.17% of all transactions.

## Data

Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB): 284,807 transactions over two days, 492 of them fraud. Features `V1`–`V28` are anonymised PCA components; `Time` and `Amount` are raw. The file (144 MB) is not stored here; see [`data/README.md`](data/README.md).

## Approach

1. **Feature selection** — consensus of decision-tree importance, correlation and Lasso; 9 low-signal features dropped.
2. **Model** — Keras network (32 → 16 → 8 → 1, dropout), `Time` in hours, `Amount` log-transformed, early stopping on validation AUC.
3. **Imbalance** — fraud class weighted heavily; weight tuned between 1:50 and 1:200.
4. **Leakage-proof cross-validation** — the scaler sits inside the pipeline, so each fold is scaled with its own training rows only. The decision threshold is chosen on these out-of-fold predictions, never on the test set.
5. **Tuning** — grid search over batch size, epochs and class weight, optimising F1.

## Results (test set: 56,962 transactions, 98 fraud)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Initial network (class weight 1:150) | 0.50 | 0.22 | 0.91 | 0.35 | 0.97 |
| Initial network | 0.80 | 0.46 | 0.87 | 0.60 | 0.97 |
| **Tuned network (class weight 1:50)** | **0.525** | **0.77** | **0.74** | **0.76** | 0.90 |

Accuracy is not useful here: predicting "not fraud" for everything scores 99.8%. The tuned model raises 22 false alarms while catching 73 frauds, against 315 false alarms in the first version. Which version is better is a business choice between catching more fraud and running a smaller review queue.

## Files

| File | Contents |
|---|---|
| `fraud_detection_nn.ipynb` | Full analysis and models |
| `data/` | Place `creditcard.csv` here (not tracked) |
