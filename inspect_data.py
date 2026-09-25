import pandas as pd
from pathlib import Path

files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]

for file in files:
    print("\n" + "=" * 70)
    print(file)

    df = pd.read_csv(file, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))