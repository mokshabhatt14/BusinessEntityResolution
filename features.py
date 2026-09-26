"""
features.py — Pairwise similarity features for business entity resolution.

Features (14 total):
  1.  name_token_sort_ratio     – token-sort fuzzy ratio
  2.  name_token_set_ratio      – token-set fuzzy ratio (subset-robust)
  3.  name_partial_ratio        – partial fuzzy ratio (substring match)
  4.  name_jaro_winkler         – Jaro-Winkler * 100
  5.  name_exact_match          – 1 if normalized names are identical
  6.  name_len_diff             – absolute character-length difference
  7.  addr_token_sort_ratio     – token-sort ratio on addresses
  8.  addr_token_set_ratio      – token-set ratio on addresses
  9.  addr_partial_ratio        – partial ratio on addresses
  10. postal_exact_match        – 1 if postal codes match
  11. country_exact_match       – 1 if country codes match
  12. name_num_common_tokens    – count of shared content tokens
  13. name_jaccard              – Jaccard on name token sets
  14. addr_jaccard              – Jaccard on address token sets

Usage:
    python features.py \\
        --candidate output/train_candidate_pairs.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --source1 dataset/train/train_source1.tsv \\
        --source2 dataset/train/train_source2.tsv \\
        --source3 dataset/train/train_source3.tsv \\
        --out output/train_features.tsv
"""

import argparse
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import ascii_normalize, name_content_tokens, addr_content_tokens

# ---------------------------------------------------------
# POSTAL CODE PATTERN
# ---------------------------------------------------------
_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")


def _extract_postal_code(address):
    if not isinstance(address, str) or not address.strip():
        return None
    matches = _POSTAL_RE.findall(address)
    return matches[-1] if matches else None


# ---------------------------------------------------------
# VECTORIZED FEATURE COMPUTATION
# (operates on numpy arrays of strings for speed)
# ---------------------------------------------------------

def compute_features_batch(names1, names2, addrs1, addrs2, countries1, countries2):
    """
    Compute features for arrays of pairs.  Returns a dict of numpy arrays.

    All inputs are numpy arrays (or lists) of strings.
    """
    n = len(names1)

    name_tok_sort   = np.empty(n, dtype=np.float32)
    name_tok_set    = np.empty(n, dtype=np.float32)
    name_partial    = np.empty(n, dtype=np.float32)
    name_jw         = np.empty(n, dtype=np.float32)
    name_exact      = np.zeros(n, dtype=np.float32)
    name_len_diff   = np.empty(n, dtype=np.float32)
    addr_tok_sort   = np.empty(n, dtype=np.float32)
    addr_tok_set    = np.empty(n, dtype=np.float32)
    addr_partial    = np.empty(n, dtype=np.float32)
    postal_match    = np.zeros(n, dtype=np.float32)
    country_match   = np.zeros(n, dtype=np.float32)
    common_tokens   = np.zeros(n, dtype=np.float32)
    name_jaccard    = np.zeros(n, dtype=np.float32)
    addr_jaccard    = np.zeros(n, dtype=np.float32)

    for i in range(n):
        n1  = names1[i]    if isinstance(names1[i],    str) else ""
        n2  = names2[i]    if isinstance(names2[i],    str) else ""
        a1  = addrs1[i]    if isinstance(addrs1[i],    str) else ""
        a2  = addrs2[i]    if isinstance(addrs2[i],    str) else ""
        c1  = countries1[i] if isinstance(countries1[i], str) else ""
        c2  = countries2[i] if isinstance(countries2[i], str) else ""

        # Name features
        name_tok_sort[i]  = fuzz.token_sort_ratio(n1, n2)
        name_tok_set[i]   = fuzz.token_set_ratio(n1, n2)
        name_partial[i]   = fuzz.partial_ratio(n1, n2)
        name_jw[i]        = JaroWinkler.similarity(n1, n2) * 100.0
        name_len_diff[i]  = abs(len(n1) - len(n2))

        norm1 = ascii_normalize(n1)
        norm2 = ascii_normalize(n2)
        name_exact[i] = float(norm1 == norm2 and norm1 != "")

        # Name token jaccard
        toks1 = set(name_content_tokens(n1))
        toks2 = set(name_content_tokens(n2))
        if toks1 or toks2:
            inter = len(toks1 & toks2)
            union = len(toks1 | toks2)
            common_tokens[i] = float(inter)
            name_jaccard[i]  = inter / union if union > 0 else 0.0

        # Address features
        addr_tok_sort[i] = fuzz.token_sort_ratio(a1, a2)
        addr_tok_set[i]  = fuzz.token_set_ratio(a1, a2)
        addr_partial[i]  = fuzz.partial_ratio(a1, a2)

        # Address token jaccard
        atoks1 = set(addr_content_tokens(a1))
        atoks2 = set(addr_content_tokens(a2))
        if atoks1 or atoks2:
            inter_a = len(atoks1 & atoks2)
            union_a = len(atoks1 | atoks2)
            addr_jaccard[i] = inter_a / union_a if union_a > 0 else 0.0

        # Postal code
        pc1 = _extract_postal_code(a1)
        pc2 = _extract_postal_code(a2)
        postal_match[i] = float(bool(pc1 and pc2 and pc1 == pc2))

        # Country
        country_match[i] = float(
            c1.strip() and c2.strip() and
            c1.strip().lower() == c2.strip().lower()
        )

    return {
        "name_token_sort_ratio":  name_tok_sort,
        "name_token_set_ratio":   name_tok_set,
        "name_partial_ratio":     name_partial,
        "name_jaro_winkler":      name_jw,
        "name_exact_match":       name_exact,
        "name_len_diff":          name_len_diff,
        "addr_token_sort_ratio":  addr_tok_sort,
        "addr_token_set_ratio":   addr_tok_set,
        "addr_partial_ratio":     addr_partial,
        "postal_exact_match":     postal_match,
        "country_exact_match":    country_match,
        "name_num_common_tokens": common_tokens,
        "name_jaccard":           name_jaccard,
        "addr_jaccard":           addr_jaccard,
    }


# Keep scalar version for backward-compat
def compute_features(name1, addr1, name2, addr2, country1="", country2=""):
    result = compute_features_batch(
        [name1], [name2], [addr1], [addr2], [country1], [country2]
    )
    return {k: float(v[0]) for k, v in result.items()}


# ---------------------------------------------------------
# FEATURE COLUMN LIST (shared with train_model.py / score_and_submit.py)
# ---------------------------------------------------------

FEATURE_COLS = [
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaro_winkler",
    "name_exact_match",
    "name_len_diff",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "postal_exact_match",
    "country_exact_match",
    "name_num_common_tokens",
    "name_jaccard",
    "addr_jaccard",
]


# ---------------------------------------------------------
# LOAD SOURCE DATA
# ---------------------------------------------------------

def _load_lookup(source1_path, source2_path, source3_path):
    s1 = pd.read_csv(source1_path, sep="\t", dtype=str)
    s2 = pd.read_csv(source2_path, sep="\t", dtype=str)
    s3 = pd.read_csv(source3_path, sep="\t", dtype=str)
    lookup = pd.concat([s1, s2, s3], ignore_index=True).set_index("entity_id")
    return lookup


# ---------------------------------------------------------
# EXPLODE CANDIDATE PAIRS
# ---------------------------------------------------------

def _explode_candidate_pairs(candidate_pairs_path, chunksize=50_000):
    """Stream candidate pairs without loading full file — avoids OOM on explode."""
    rows = []
    for chunk in pd.read_csv(candidate_pairs_path, sep="\t", dtype=str,
                             chunksize=chunksize):
        chunk = chunk.fillna("")
        chunk = chunk[chunk["candidate_entity_ids"] != ""]
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            for cid in row["candidate_entity_ids"].split(","):
                cid = cid.strip()
                if cid:
                    rows.append((s1_id, cid))
        # flush to DataFrame periodically to avoid growing rows list too large
        if len(rows) >= 5_000_000:
            yield from rows
            rows = []
    yield from rows

def _collect_candidate_pairs(candidate_pairs_path):
    """Collect all pairs into a DataFrame, chunked to avoid OOM."""
    all_rows = []
    for s1_id, cid in _explode_candidate_pairs(candidate_pairs_path):
        all_rows.append((s1_id, cid))
    return pd.DataFrame(all_rows, columns=["source1_entity_id", "candidate_entity_id"])


# ---------------------------------------------------------
# LOAD GROUND TRUTH
# ---------------------------------------------------------

def load_ground_truth_map(ground_truth_path):
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")

    def _split(ids):
        if not ids:
            return set()
        return set(i.strip() for i in ids.split(",") if i.strip())

    return dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(_split)))


# ---------------------------------------------------------
# BUILD TRAINING TABLE
# ---------------------------------------------------------

def build_training_table(
    candidate_pairs_path,
    ground_truth_path,
    source1_path,
    source2_path,
    source3_path,
    out_path="output/train_features.tsv",
    max_pairs=None,
    batch_size=20_000,
):
    print("[features.py] loading source data...")
    lookup = _load_lookup(source1_path, source2_path, source3_path)

    print("[features.py] loading ground truth...")
    gt_map = load_ground_truth_map(ground_truth_path)

    print("[features.py] streaming candidate pairs + computing features in batches...")
    return _stream_features(
        candidate_pairs_path, lookup, gt_map,
        batch_size=batch_size, max_pairs=max_pairs, out_path=out_path
    )


def _stream_features(candidate_pairs_path, lookup, gt_map,
                     batch_size=20_000, max_pairs=None, out_path="output/train_features.tsv"):
    """
    Streams candidate pairs in batches, writes features directly to disk.
    Never holds more than batch_size pairs in RAM at once.
    Returns the output path.
    """
    total_pairs = 0
    total_pos   = 0
    buf_s1, buf_cand = [], []
    cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLS + ["label"]
    first_write = True

    def flush(buf_s1, buf_cand):
        nonlocal total_pos, first_write
        chunk_df = _compute_chunk(buf_s1, buf_cand, lookup, gt_map)
        total_pos += int(chunk_df["label"].sum())
        chunk_df[cols].to_csv(
            out_path, sep="\t", index=False,
            mode="w" if first_write else "a",
            header=first_write
        )
        first_write = False

    for s1_id, cid in _explode_candidate_pairs(candidate_pairs_path):
        if max_pairs and total_pairs >= max_pairs:
            break
        if s1_id not in lookup.index or cid not in lookup.index:
            continue
        buf_s1.append(s1_id)
        buf_cand.append(cid)
        total_pairs += 1
        if len(buf_s1) >= batch_size:
            flush(buf_s1, buf_cand)
            if total_pairs % 500_000 == 0:
                print(f"  [{total_pairs:,} pairs, {total_pos:,} pos]")
            buf_s1, buf_cand = [], []

    if buf_s1:
        flush(buf_s1, buf_cand)

    print(f"  [{total_pairs:,} pairs total, {total_pos:,} positive]")
    return out_path


def _compute_chunk(s1_ids, cand_ids, lookup, gt_map):
    s1_info   = lookup.reindex(s1_ids)[["business_name","business_address","country"]].fillna("").reset_index(drop=True)
    cand_info = lookup.reindex(cand_ids)[["business_name","business_address","country"]].fillna("").reset_index(drop=True)
    feats = compute_features_batch(
        s1_info["business_name"].to_numpy(dtype=object),
        cand_info["business_name"].to_numpy(dtype=object),
        s1_info["business_address"].to_numpy(dtype=object),
        cand_info["business_address"].to_numpy(dtype=object),
        s1_info["country"].to_numpy(dtype=object),
        cand_info["country"].to_numpy(dtype=object),
    )
    df = pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_id": cand_ids, **feats})
    df["label"] = [int(c in gt_map.get(s, set())) for s, c in zip(s1_ids, cand_ids)]
    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate",     default="output/train_candidate_pairs.tsv")
    parser.add_argument("--ground-truth",  default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--source1",       default="dataset/train/train_source1.tsv")
    parser.add_argument("--source2",       default="dataset/train/train_source2.tsv")
    parser.add_argument("--source3",       default="dataset/train/train_source3.tsv")
    parser.add_argument("--out",           default="output/train_features.tsv")
    parser.add_argument("--max-pairs",     type=int, default=None)
    args = parser.parse_args()

    build_training_table(
        args.candidate, args.ground_truth,
        args.source1, args.source2, args.source3,
        out_path=args.out,
        max_pairs=args.max_pairs,
    )

    # Read back just label column for summary stats (lightweight)
    labels = pd.read_csv(args.out, sep="\t", usecols=["label"])["label"]
    n_pos   = int(labels.sum())
    n_total = len(labels)
    print(f"\n{'='*40}")
    print(f"[features.py] wrote {n_total:,} labeled pairs")
    print(f"  positive: {n_pos:,}  ({n_pos/n_total*100:.2f}%)")
    print(f"  negative: {n_total-n_pos:,}")
    print(f"  output:   {args.out}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
