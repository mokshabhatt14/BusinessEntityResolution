"""
features.py — Person 2
 
Pairwise similarity features for a (Source 1 record, Source 2/3 record) pair,
plus a helper that turns Person 1's candidate_pairs.tsv + train_ground_truth.tsv
into a labeled training table.
 
Usage as a script (build the labeled training feature table):
 
    python3 src/features.py \
        --candidate  output/train_candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --out output/train_features.tsv
 
Note: --candidate here is Person 1's *train-side* candidate_pairs.tsv (blocking
run on the training data), not the final output/candidate_pairs.tsv, which is
reserved for the test-set submission per the problem statement's file layout.
"""
import argparse
import re
 
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
 
# Generic postal-code pattern: 4-6 digit runs (covers US 5-digit ZIP,
# India 6-digit PIN, and is a reasonable catch-all for other countries
# like France's 5-digit codes postaux, without hard-coding country logic).
_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")
 
 
def _extract_postal_code(address):
    """Return the last 4-6 digit run found in an address string, or None."""
    if not isinstance(address, str) or not address.strip():
        return None
    matches = _POSTAL_RE.findall(address)
    return matches[-1] if matches else None
 
 
def compute_features(name1, addr1, name2, addr2):
    """Compute similarity features for one candidate pair.
 
    Returns a dict with:
      - name_token_sort_ratio: rapidfuzz.fuzz.token_sort_ratio on names (0-100)
      - addr_token_sort_ratio: rapidfuzz.fuzz.token_sort_ratio on addresses (0-100)
      - name_jaro_winkler: rapidfuzz JaroWinkler similarity on names, scaled to 0-100
      - postal_exact_match: bool, True iff both addresses yield a postal code
        and the codes match exactly
    """
    name1 = name1 if isinstance(name1, str) else ""
    name2 = name2 if isinstance(name2, str) else ""
    addr1 = addr1 if isinstance(addr1, str) else ""
    addr2 = addr2 if isinstance(addr2, str) else ""
 
    name_token_sort = fuzz.token_sort_ratio(name1, name2)
    addr_token_sort = fuzz.token_sort_ratio(addr1, addr2)
    name_jaro_winkler = JaroWinkler.similarity(name1, name2) * 100.0
 
    pc1 = _extract_postal_code(addr1)
    pc2 = _extract_postal_code(addr2)
    postal_exact_match = bool(pc1 and pc2 and pc1 == pc2)
 
    return {
        "name_token_sort_ratio": name_token_sort,
        "addr_token_sort_ratio": addr_token_sort,
        "name_jaro_winkler": name_jaro_winkler,
        "postal_exact_match": postal_exact_match,
    }
 
 
def _load_lookup(source1_path, source2_path, source3_path):
    s1 = pd.read_csv(source1_path, sep="\t", dtype=str)
    s2 = pd.read_csv(source2_path, sep="\t", dtype=str)
    s3 = pd.read_csv(source3_path, sep="\t", dtype=str)
    lookup = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
    return lookup
 
 
def _explode_candidate_pairs(candidate_pairs_path):
    df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str).fillna("")
    rows = []
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        ids = row["candidate_entity_ids"]
        if not ids:
            continue
        for cid in ids.split(","):
            cid = cid.strip()
            if cid:
                rows.append((s1_id, cid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
 
 
def load_ground_truth_map(ground_truth_path):
    """source1_entity_id -> set of matched_entity_ids (empty set for singletons)."""
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    out = {}
    for _, row in gt.iterrows():
        ids = row["matched_entity_ids"]
        out[row["source1_entity_id"]] = (
            set(i.strip() for i in ids.split(",") if i.strip()) if ids else set()
        )
    return out
 
 
def build_training_table(candidate_pairs_path, ground_truth_path,
                          source1_path, source2_path, source3_path):
    """Join candidate pairs with ground truth, label 1/0, compute features row-wise."""
    lookup = _load_lookup(source1_path, source2_path, source3_path)
    pairs = _explode_candidate_pairs(candidate_pairs_path)
    gt_map = load_ground_truth_map(ground_truth_path)
 
    feature_rows = []
    skipped = 0
    for _, row in pairs.iterrows():
        s1_id, c_id = row["source1_entity_id"], row["candidate_entity_id"]
        if s1_id not in lookup.index or c_id not in lookup.index:
            skipped += 1
            continue
        name1, addr1 = lookup.at[s1_id, "business_name"], lookup.at[s1_id, "business_address"]
        name2, addr2 = lookup.at[c_id, "business_name"], lookup.at[c_id, "business_address"]
        feats = compute_features(name1, addr1, name2, addr2)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = c_id
        feats["label"] = int(c_id in gt_map.get(s1_id, set()))
        feature_rows.append(feats)
 
    if skipped:
        print(f"[features.py] warning: skipped {skipped} pairs referencing unknown entity_ids")
 
    cols = ["source1_entity_id", "candidate_entity_id",
            "name_token_sort_ratio", "addr_token_sort_ratio",
            "name_jaro_winkler", "postal_exact_match", "label"]
    return pd.DataFrame(feature_rows, columns=cols)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Build a labeled training feature table from candidate pairs + ground truth."
    )
    parser.add_argument("--candidate", default="output/train_candidate_pairs.tsv")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--source1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--source2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--source3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--out", default="output/train_features.tsv")
    args = parser.parse_args()
 
    table = build_training_table(
        args.candidate, args.ground_truth, args.source1, args.source2, args.source3
    )
    table.to_csv(args.out, sep="\t", index=False)
    n_pos = int(table["label"].sum())
    print(f"[features.py] wrote {len(table)} labeled pairs ({n_pos} positive, "
          f"{len(table) - n_pos} negative) to {args.out}")
 
 
if __name__ == "__main__":
    main()
 