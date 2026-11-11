# Data

Not stored in Git (about 1.6 GB).

1. Download the LendingClub loan data from Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Put `accepted_2007_to_2018Q4.csv` in this folder.
3. Run the first cell of `pd_model_nn.ipynb` once (uncommented) to convert it to `lendingclubfull.parquet`, which loads much faster.

`rejected_2007_to_2018Q4.csv` from the same download is not used.
