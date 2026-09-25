"""
train_model.py — Person 2
 
Trains a LogisticRegression matcher on the labeled feature table produced by
features.py. Splits by source1_entity_id (not by pair) so that all pairs
belonging to a given Source 1 entity stay in the same split — otherwise the
model could see some of an entity's candidates at train time and be scored
on the rest, which leaks information and inflates validation F0.5.
 
Usage:
 
    python3 src/train_model.py \
        --features output/train_features.tsv \
        --model-out output/matcher_model.joblib \
        --val-out output/val_features.tsv
"""
import argparse
 
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
 
FEATURE_COLS = [
    "name_token_sort_ratio",
    "addr_token_sort_ratio",
    "name_jaro_winkler",
    "postal_exact_match",
]
 
 
def split_by_entity(df, test_size=0.2, random_state=42):
    """80/20 split grouped by source1_entity_id (no leakage across splits)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["source1_entity_id"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
 
    # sanity check: no S1 entity should appear on both sides
    overlap = set(train_df["source1_entity_id"]) & set(val_df["source1_entity_id"])
    assert not overlap, f"leakage: {len(overlap)} S1 entities appear in both splits"
 
    return train_df, val_df
 
 
def prepare_xy(df):
    X = df[FEATURE_COLS].astype(float)
    y = df["label"].astype(int)
    return X, y
 
 
def train(df):
    X, y = prepare_xy(df)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    return model
 
 
def main():
    parser = argparse.ArgumentParser(description="Train the logistic regression entity matcher.")
    parser.add_argument("--features", default="output/train_features.tsv")
    parser.add_argument("--model-out", default="output/matcher_model.joblib")
    parser.add_argument("--val-out", default="output/val_features.tsv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
 
    df = pd.read_csv(args.features, sep="\t")
    train_df, val_df = split_by_entity(df, test_size=args.test_size, random_state=args.random_state)
 
    print(f"[train_model.py] train: {len(train_df)} pairs / "
          f"{train_df['source1_entity_id'].nunique()} S1 entities "
          f"({int(train_df['label'].sum())} positive)")
    print(f"[train_model.py] val:   {len(val_df)} pairs / "
          f"{val_df['source1_entity_id'].nunique()} S1 entities "
          f"({int(val_df['label'].sum())} positive)")
 
    model = train(train_df)
    joblib.dump(model, args.model_out)
    val_df.to_csv(args.val_out, sep="\t", index=False)
 
    print(f"[train_model.py] saved model to {args.model_out}")
    print(f"[train_model.py] saved held-out validation pairs to {args.val_out}")
 
    coefs = dict(zip(FEATURE_COLS, model.coef_[0]))
    print(f"[train_model.py] coefficients: {coefs}")
    print(f"[train_model.py] intercept: {model.intercept_[0]:.4f}")
 
 
if __name__ == "__main__":
    main()
 