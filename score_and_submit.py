"""
score_and_submit.py — Person 2
 
1. Implements the challenge's exact macro-averaged F0.5 metric (singleton with
   correctly-predicted empty match list scores 1.0).
2. Sweeps decision thresholds 0.1-0.9 on the held-out validation split from
   train_model.py and reports F0.5 at each.
3. Loads Person 1's test-side output/candidate_pairs.tsv, scores every
   candidate pair with the trained model, applies the chosen threshold, and
   writes output/matching_results.tsv in the exact required format.
 
Usage:
 
    python3 src/score_and_submit.py \
        --model output/matcher_model.joblib \
        --val-features output/val_features.tsv \
        --train-ground-truth dataset/train/train_ground_truth.tsv \
        --test-candidates output/candidate_pairs.tsv \
        --test-source1 dataset/test/test_source1.tsv \
        --test-source2 dataset/test/test_source2.tsv \
        --test-source3 dataset/test/test_source3.tsv \
        --out output/matching_results.tsv
 
Then validate before uploading:
 
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
 
import joblib
import numpy as np
import pandas as pd
 
from features import compute_features
from train_model import FEATURE_COLS
 
 
# ---------------------------------------------------------------------------
# F_0.5 macro-average, exactly as defined in the problem statement.
# ---------------------------------------------------------------------------
 
def f_beta_score_per_entity(true_set, pred_set, beta=0.5):
    """F_beta for one Source 1 entity.
 
    - Singleton correctly predicted empty (true empty, pred empty) -> 1.0
    - Any match predicted for a true singleton (true empty, pred non-empty) -> 0.0
    - True matches exist but nothing predicted -> 0.0 (recall = 0)
    - Otherwise standard precision/recall F_beta, with 0.0 when there's no overlap.
    """
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
 
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
 
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    beta2 = beta ** 2
    denom = (beta2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom
 
 
def macro_f_beta(all_s1_ids, true_map, pred_map, beta=0.5):
    """Average of per-entity F_beta over every S1 entity in the eval set
    (singletons included, per the problem statement)."""
    scores = [
        f_beta_score_per_entity(true_map.get(s1_id, set()), pred_map.get(s1_id, set()), beta=beta)
        for s1_id in all_s1_ids
    ]
    return float(np.mean(scores)) if scores else 0.0
 
 
# ---------------------------------------------------------------------------
# Threshold sweep on the held-out validation split.
# ---------------------------------------------------------------------------
 
def load_ground_truth_map(ground_truth_path):
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    out = {}
    for _, row in gt.iterrows():
        ids = row["matched_entity_ids"]
        out[row["source1_entity_id"]] = (
            set(i.strip() for i in ids.split(",") if i.strip()) if ids else set()
        )
    return out
 
 
def predict_probs(model, df):
    X = df[FEATURE_COLS].astype(float)
    return model.predict_proba(X)[:, 1]
 
 
def sweep_thresholds(val_df, model, ground_truth_path, thresholds=None):
    """Score F0.5 at each threshold; returns (best_threshold, best_score, all_results).
 
    IMPORTANT caveat: val_df only contains pairs that survived blocking, so this
    is F0.5 conditional on Person 1's candidate recall for the validation split
    (the same recall ceiling that applies to the final submission).
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]
 
    val_df = val_df.copy()
    val_df["prob"] = predict_probs(model, val_df)
 
    all_s1_ids = sorted(val_df["source1_entity_id"].unique())
    gt_map = load_ground_truth_map(ground_truth_path)
    true_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in all_s1_ids}
 
    results = []
    for t in thresholds:
        matched = val_df[val_df["prob"] >= t]
        pred_map = {
            s1_id: set(group["candidate_entity_id"])
            for s1_id, group in matched.groupby("source1_entity_id")
        }
        score = macro_f_beta(all_s1_ids, true_map, pred_map, beta=0.5)
        results.append((t, score))
        print(f"[score_and_submit.py] threshold={t:.2f}  F0.5={score:.4f}")
 
    best_t, best_score = max(results, key=lambda x: x[1])
    print(f"[score_and_submit.py] best threshold={best_t:.2f}  F0.5={best_score:.4f}")
    return best_t, best_score, results
 
 
# ---------------------------------------------------------------------------
# Test-set inference and submission file.
# ---------------------------------------------------------------------------
 
def load_candidate_pairs(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows = []
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        ids = row["candidate_entity_ids"]
        for cid in (ids.split(",") if ids else []):
            cid = cid.strip()
            if cid:
                rows.append((s1_id, cid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
 
 
def build_features_for_pairs(pairs_df, source1_path, source2_path, source3_path):
    s1 = pd.read_csv(source1_path, sep="\t", dtype=str)
    s2 = pd.read_csv(source2_path, sep="\t", dtype=str)
    s3 = pd.read_csv(source3_path, sep="\t", dtype=str)
    lookup = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
 
    feat_rows = []
    for _, row in pairs_df.iterrows():
        s1_id, c_id = row["source1_entity_id"], row["candidate_entity_id"]
        if s1_id not in lookup.index or c_id not in lookup.index:
            continue
        name1, addr1 = lookup.at[s1_id, "business_name"], lookup.at[s1_id, "business_address"]
        name2, addr2 = lookup.at[c_id, "business_name"], lookup.at[c_id, "business_address"]
        feats = compute_features(name1, addr1, name2, addr2)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = c_id
        feat_rows.append(feats)
    return pd.DataFrame(feat_rows)
 
 
def write_matching_results(all_s1_ids, pred_map, out_path):
    lines = ["source1_entity_id\tmatched_entity_ids"]
    for s1_id in all_s1_ids:
        ids = sorted(pred_map.get(s1_id, set()))
        lines.append(f"{s1_id}\t{','.join(ids)}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[score_and_submit.py] wrote {len(all_s1_ids)} rows to {out_path}")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Sweep threshold on validation split, score test candidates, write submission."
    )
    parser.add_argument("--model", default="output/matcher_model.joblib")
    parser.add_argument("--val-features", default="output/val_features.tsv")
    parser.add_argument("--train-ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--test-candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-source1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--test-source2", default="dataset/test/test_source2.tsv")
    parser.add_argument("--test-source3", default="dataset/test/test_source3.tsv")
    parser.add_argument("--out", default="output/matching_results.tsv")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Skip the sweep and force this threshold instead.")
    args = parser.parse_args()
 
    model = joblib.load(args.model)
    val_df = pd.read_csv(args.val_features, sep="\t")
 
    if args.threshold is not None:
        best_t = args.threshold
        print(f"[score_and_submit.py] using forced threshold={best_t}")
    else:
        best_t, _, _ = sweep_thresholds(val_df, model, args.train_ground_truth)
 
    test_pairs = load_candidate_pairs(args.test_candidates)
    test_features = build_features_for_pairs(
        test_pairs, args.test_source1, args.test_source2, args.test_source3
    )
 
    all_s1_ids = pd.read_csv(args.test_source1, sep="\t", dtype=str)["entity_id"].tolist()
 
    pred_map = {}
    if len(test_features):
        test_features["prob"] = predict_probs(model, test_features)
        matched = test_features[test_features["prob"] >= best_t]
        pred_map = {
            s1_id: set(group["candidate_entity_id"])
            for s1_id, group in matched.groupby("source1_entity_id")
        }
 
    write_matching_results(all_s1_ids, pred_map, args.out)
 
    print("\n[score_and_submit.py] next step — validate before uploading:")
    print("  python3 utils/validate_submission.py \\")
    print(f"      --matching {args.out} \\")
    print(f"      --candidate {args.test_candidates} \\")
    print("      --test-dir dataset/test")
 
 
if __name__ == "__main__":
    main()
 