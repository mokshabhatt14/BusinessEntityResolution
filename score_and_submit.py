"""
score_and_submit.py
Final test scoring and submission generation for Business Entity Resolution.

Uses the exact same 14 features as features.py / train_model.py.
Processes test candidates in chunks.
Auto-tunes threshold via F0.5 sweep on val_features.tsv if present.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLS, compute_features_batch


# ---------------------------------------------------------------------
# Load test entity lookup
# ---------------------------------------------------------------------

def load_entity_lookup(source1_path, source2_path, source3_path):
    print("[score_and_submit.py] loading test source files...")
    s1 = pd.read_csv(source1_path, sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(source2_path, sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(source3_path, sep="\t", dtype=str).fillna("")

    lookup = pd.concat([s1, s2, s3], ignore_index=True)[[
        "entity_id", "business_name", "business_address", "country"
    ]].set_index("entity_id")

    print(f"[score_and_submit.py] {len(lookup):,} entities loaded")
    return lookup, s1["entity_id"].tolist()


# ---------------------------------------------------------------------
# Score one batch
# ---------------------------------------------------------------------

def score_batch(source1_ids, candidate_ids, lookup, model, threshold, pred_map):
    if not source1_ids:
        return 0, 0

    s1_rows   = lookup.reindex(source1_ids)
    cand_rows = lookup.reindex(candidate_ids)

    names1    = s1_rows["business_name"].fillna("").to_numpy(dtype=object)
    names2    = cand_rows["business_name"].fillna("").to_numpy(dtype=object)
    addrs1    = s1_rows["business_address"].fillna("").to_numpy(dtype=object)
    addrs2    = cand_rows["business_address"].fillna("").to_numpy(dtype=object)
    countries1 = s1_rows["country"].fillna("").to_numpy(dtype=object)
    countries2 = cand_rows["country"].fillna("").to_numpy(dtype=object)

    feat_dict = compute_features_batch(names1, names2, addrs1, addrs2, countries1, countries2)

    X = pd.DataFrame({col: feat_dict[col] for col in FEATURE_COLS})
    probabilities = model.predict_proba(X.values)[:, 1]
    matched_indices = np.flatnonzero(probabilities >= threshold)

    for idx in matched_indices:
        s1_id = source1_ids[idx]
        cand_id = candidate_ids[idx]
        pred_map.setdefault(s1_id, set()).add(cand_id)

    return len(source1_ids), len(matched_indices)


# ---------------------------------------------------------------------
# Process candidate file in chunks
# ---------------------------------------------------------------------

def process_test_candidates(candidate_path, lookup, model, threshold, chunk_size=5000):
    print("[score_and_submit.py] processing test candidates in chunks...")
    pred_map = {}
    total_pairs = 0
    total_matches = 0
    chunk_number = 0

    reader = pd.read_csv(candidate_path, sep="\t", dtype=str, chunksize=chunk_size)

    for chunk in reader:
        chunk = chunk.fillna("")
        batch_s1 = []
        batch_cand = []

        for row in chunk.itertuples(index=False):
            s1_id = row.source1_entity_id
            cand_string = row.candidate_entity_ids
            if not cand_string:
                continue
            for cid in cand_string.split(","):
                cid = cid.strip()
                if cid:
                    batch_s1.append(s1_id)
                    batch_cand.append(cid)

        if not batch_s1:
            continue

        processed, matches = score_batch(
            batch_s1, batch_cand, lookup, model, threshold, pred_map
        )
        total_pairs   += processed
        total_matches += matches
        chunk_number  += 1

        if chunk_number % 200 == 0:
            print(
                f"[score_and_submit.py] chunk {chunk_number}: "
                f"{total_pairs:,} pairs processed | {total_matches:,} matches",
                flush=True,
            )

    print(f"[score_and_submit.py] finished: {total_pairs:,} pairs → {total_matches:,} matches")
    return pred_map


# ---------------------------------------------------------------------
# Write final submission
# ---------------------------------------------------------------------

def write_matching_results(all_s1_ids, pred_map, output_path):
    print(f"[score_and_submit.py] writing {len(all_s1_ids):,} rows → {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matched = sorted(pred_map.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(matched)}\n")
    print("[score_and_submit.py] done writing submission")


# ---------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------

def tune_threshold_f05(model, val_features_path):
    from sklearn.metrics import fbeta_score
    print(f"[score_and_submit.py] tuning threshold from {val_features_path} ...")
    val = pd.read_csv(val_features_path, sep="\t")
    missing = [c for c in FEATURE_COLS if c not in val.columns]
    if missing:
        print(f"[score_and_submit.py] WARNING: missing cols {missing}, using 0.5")
        return 0.5

    X_val = val[FEATURE_COLS].astype(float).values
    y_val = val["label"].astype(int).values
    proba = model.predict_proba(X_val)[:, 1]

    best_thresh, best_f = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        preds = (proba >= t).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
        if f > best_f:
            best_f, best_thresh = f, float(t)

    print(f"[score_and_submit.py] best val F0.5={best_f:.4f} at threshold={best_thresh:.2f}")
    return best_thresh


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="output/matcher_model.joblib")
    parser.add_argument("--test-candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-source1",    default="dataset/test/test_source1.tsv")
    parser.add_argument("--test-source2",    default="dataset/test/test_source2.tsv")
    parser.add_argument("--test-source3",    default="dataset/test/test_source3.tsv")
    parser.add_argument("--out",             default="output/matching_results.tsv")
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--val-features",    default="output/val_features.tsv")
    parser.add_argument("--chunk-size",      type=int,   default=5000)
    args = parser.parse_args()

    print("[score_and_submit.py] loading model...")
    model = joblib.load(args.model)

    threshold = args.threshold
    if args.val_features and os.path.exists(args.val_features):
        threshold = tune_threshold_f05(model, args.val_features)
    else:
        print(f"[score_and_submit.py] using fixed threshold={threshold}")

    lookup, all_s1_ids = load_entity_lookup(
        args.test_source1, args.test_source2, args.test_source3
    )

    pred_map = process_test_candidates(
        args.test_candidates, lookup, model, threshold,
        chunk_size=args.chunk_size,
    )

    write_matching_results(all_s1_ids, pred_map, args.out)

    print()
    print("=" * 40)
    print("SUBMISSION CREATED")
    print("=" * 40)
    print(f"File:      {args.out}")
    print(f"Threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
