"""
train_model.py — Train the entity matcher on the labeled feature table.

Uses HistGradientBoostingClassifier with entity-grouped 80/20 split.
Auto-tunes classification threshold on the held-out validation set.

Usage:
    python train_model.py \\
        --features output/train_features.tsv \\
        --model-out output/matcher_model.joblib \\
        --val-out output/val_features.tsv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import fbeta_score, precision_score, recall_score

from features import FEATURE_COLS


def split_by_entity(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["source1_entity_id"]
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    overlap  = set(train_df["source1_entity_id"]) & set(val_df["source1_entity_id"])
    assert not overlap, f"leakage: {len(overlap)} S1 entities in both splits"
    return train_df, val_df


def prepare_xy(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].astype(float).fillna(0.0)
    y = df["label"].astype(int)
    return X, y


def train(df):
    X, y = prepare_xy(df)
    pos = int(y.sum())
    neg = len(y) - pos
    sample_weight = np.where(y == 1, neg / max(pos, 1), 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=7,
        min_samples_leaf=20, l2_regularization=0.1, random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def tune_threshold(model, val_df, beta=0.5):
    X_val, y_val = prepare_xy(val_df)
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f = 0.5, -1.0
    rows = []
    for t in np.arange(0.01, 1.00, 0.01):
        preds = (proba >= t).astype(int)
        f = fbeta_score(y_val, preds, beta=beta, zero_division=0)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        rows.append((t, f, p, r))
        if f > best_f:
            best_f, best_thresh = f, t
    rows.sort(key=lambda x: -x[1])
    print("[train_model.py] Top thresholds by F0.5:")
    for t, f, p, r in rows[:5]:
        print(f"  thresh={t:.2f}  F0.5={f:.4f}  P={p:.4f}  R={r:.4f}")
    return float(best_thresh), float(best_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",     default="output/train_features.tsv")
    parser.add_argument("--model-out",    default="output/matcher_model.joblib")
    parser.add_argument("--val-out",      default="output/val_features.tsv")
    parser.add_argument("--test-size",    type=float, default=0.2)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    print(f"[train_model.py] loading {args.features} ...")
    df = pd.read_csv(args.features, sep="\t")
    print(f"[train_model.py] {len(df):,} rows, {int(df['label'].sum()):,} positives")

    train_df, val_df = split_by_entity(df, test_size=args.test_size,
                                        random_state=args.random_state)
    print(f"[train_model.py] train: {len(train_df):,} rows ({int(train_df['label'].sum()):,} pos)")
    print(f"[train_model.py] val:   {len(val_df):,} rows ({int(val_df['label'].sum()):,} pos)")

    print("[train_model.py] training ...")
    model = train(train_df)
    joblib.dump(model, args.model_out)
    val_df.to_csv(args.val_out, sep="\t", index=False)
    print(f"[train_model.py] model -> {args.model_out}")

    best_thresh, best_f = tune_threshold(model, val_df, beta=0.5)
    print(f"[train_model.py] best val F0.5={best_f:.4f} at threshold={best_thresh:.2f}")

    with open("output/best_threshold.txt", "w") as fh:
        fh.write(str(best_thresh))


if __name__ == "__main__":
    main()
