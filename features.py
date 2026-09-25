"""
features.py — Person 2

Pairwise similarity features for a (Source 1 record, Source 2/3 record) pair,
plus a helper that turns Person 1's candidate_pairs.tsv + train_ground_truth.tsv
into a labeled training table.

This version supports optional sampling of candidate pairs using --max-pairs.

Example:

    python features.py \
        --candidate output/train_candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --source1 dataset/train/train_source1.tsv \
        --source2 dataset/train/train_source2.tsv \
        --source3 dataset/train/train_source3.tsv \
        --out output/train_features.tsv \
        --max-pairs 10000000
"""

import argparse
import re

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# ---------------------------------------------------------
# POSTAL CODE PATTERN
# ---------------------------------------------------------

# Generic postal-code pattern:
# 4-6 digit runs.
# Covers:
# - US ZIP codes
# - India PIN codes
# - France postal codes
# - Other common formats
_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")


def _extract_postal_code(address):
    """
    Return the last 4-6 digit run found in an address string.

    Returns:
        postal code as string, or None if no postal code is found.
    """

    if not isinstance(address, str) or not address.strip():
        return None

    matches = _POSTAL_RE.findall(address)

    return matches[-1] if matches else None


# ---------------------------------------------------------
# FEATURE CALCULATION
# ---------------------------------------------------------

def compute_features(name1, addr1, name2, addr2):
    """
    Compute similarity features for one candidate pair.

    Features:
      - name_token_sort_ratio
      - addr_token_sort_ratio
      - name_jaro_winkler
      - postal_exact_match
    """

    # Convert missing/non-string values into empty strings.
    name1 = name1 if isinstance(name1, str) else ""
    name2 = name2 if isinstance(name2, str) else ""

    addr1 = addr1 if isinstance(addr1, str) else ""
    addr2 = addr2 if isinstance(addr2, str) else ""

    # -----------------------------------------------------
    # NAME SIMILARITY
    # -----------------------------------------------------

    name_token_sort = fuzz.token_sort_ratio(
        name1,
        name2
    )

    # -----------------------------------------------------
    # ADDRESS SIMILARITY
    # -----------------------------------------------------

    addr_token_sort = fuzz.token_sort_ratio(
        addr1,
        addr2
    )

    # -----------------------------------------------------
    # JARO-WINKLER NAME SIMILARITY
    # -----------------------------------------------------

    name_jaro_winkler = (
        JaroWinkler.similarity(
            name1,
            name2
        ) * 100.0
    )

    # -----------------------------------------------------
    # POSTAL CODE MATCH
    # -----------------------------------------------------

    pc1 = _extract_postal_code(addr1)
    pc2 = _extract_postal_code(addr2)

    postal_exact_match = bool(
        pc1
        and pc2
        and pc1 == pc2
    )

    # -----------------------------------------------------
    # RETURN FEATURES
    # -----------------------------------------------------

    return {
        "name_token_sort_ratio": name_token_sort,
        "addr_token_sort_ratio": addr_token_sort,
        "name_jaro_winkler": name_jaro_winkler,
        "postal_exact_match": postal_exact_match,
    }


# ---------------------------------------------------------
# LOAD SOURCE DATA
# ---------------------------------------------------------

def _load_lookup(source1_path, source2_path, source3_path):
    """
    Load Source 1, Source 2 and Source 3.

    Combines them into one lookup table indexed by entity_id.
    """

    s1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str
    )

    s2 = pd.read_csv(
        source2_path,
        sep="\t",
        dtype=str
    )

    s3 = pd.read_csv(
        source3_path,
        sep="\t",
        dtype=str
    )

    lookup = pd.concat(
        [
            s1,
            s2,
            s3
        ],
        ignore_index=True
    ).set_index("entity_id")

    return lookup


# ---------------------------------------------------------
# EXPLODE CANDIDATE PAIRS
# ---------------------------------------------------------

def _explode_candidate_pairs(candidate_pairs_path):
    """
    Convert candidate_entity_ids containing comma-separated IDs
    into one row per candidate pair.

    Example:

        source1_entity_id | candidate_entity_ids
        A001              | B001,B002,B003

    becomes:

        A001 | B001
        A001 | B002
        A001 | B003
    """

    df = pd.read_csv(
        candidate_pairs_path,
        sep="\t",
        dtype=str
    ).fillna("")

    # Remove rows where there are no candidates.
    df = df[
        df["candidate_entity_ids"] != ""
    ]

    # Split comma-separated candidate IDs.
    df = df.assign(
        candidate_entity_id=
        df["candidate_entity_ids"].str.split(",")
    )

    # Turn list values into separate rows.
    df = df.explode(
        "candidate_entity_id"
    )

    # Remove extra whitespace.
    df["candidate_entity_id"] = (
        df["candidate_entity_id"]
        .str.strip()
    )

    # Remove empty candidate IDs.
    df = df[
        df["candidate_entity_id"] != ""
    ]

    return df[
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ].reset_index(drop=True)


# ---------------------------------------------------------
# LOAD GROUND TRUTH
# ---------------------------------------------------------

def load_ground_truth_map(ground_truth_path):
    """
    Create a mapping:

        source1_entity_id -> set of matched_entity_ids

    Example:

        A001 -> {B001, B007}
        A002 -> {B004}

    A source with no matches gets an empty set.
    """

    gt = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str
    ).fillna("")

    def _split(ids):
        if not ids:
            return set()

        return set(
            i.strip()
            for i in ids.split(",")
            if i.strip()
        )

    matched = gt[
        "matched_entity_ids"
    ].map(_split)

    return dict(
        zip(
            gt["source1_entity_id"],
            matched
        )
    )


# ---------------------------------------------------------
# BUILD TRAINING TABLE
# ---------------------------------------------------------

def build_training_table(
    candidate_pairs_path,
    ground_truth_path,
    source1_path,
    source2_path,
    source3_path,
    max_pairs=None
):
    """
    Join candidate pairs with ground truth,
    create labels, and compute similarity features.

    max_pairs:
        If specified, randomly samples this many candidate
        pairs before feature computation.

        Example:
            max_pairs=10000000

        processes 10 million pairs instead of all
        76+ million pairs.
    """

    # -----------------------------------------------------
    # LOAD ALL SOURCE DATA
    # -----------------------------------------------------

    print("[features.py] loading source data...")

    lookup = _load_lookup(
        source1_path,
        source2_path,
        source3_path
    )

    # -----------------------------------------------------
    # LOAD CANDIDATE PAIRS
    # -----------------------------------------------------

    print("[features.py] loading candidate pairs...")

    pairs = _explode_candidate_pairs(
        candidate_pairs_path
    )

    # -----------------------------------------------------
    # LOAD GROUND TRUTH
    # -----------------------------------------------------

    print("[features.py] loading ground truth...")

    gt_map = load_ground_truth_map(
        ground_truth_path
    )

    print(
        f"[features.py] {len(pairs):,} "
        "candidate pairs found..."
    )

    # -----------------------------------------------------
    # OPTIONAL SAMPLING
    # -----------------------------------------------------

    if max_pairs is not None:

        if max_pairs <= 0:
            raise ValueError(
                "--max-pairs must be greater than 0."
            )

        if len(pairs) > max_pairs:

            print(
                f"[features.py] sampling "
                f"{max_pairs:,} pairs from "
                f"{len(pairs):,} total pairs..."
            )

            # Fixed random_state means:
            # every run gives the same sample.
            pairs = pairs.sample(
                n=max_pairs,
                random_state=42
            ).reset_index(drop=True)

            print(
                f"[features.py] "
                f"{len(pairs):,} pairs selected "
                "for processing."
            )

        else:

            print(
                "[features.py] total candidate pairs "
                "are already below --max-pairs; "
                "no sampling needed."
            )

    # -----------------------------------------------------
    # CHECK ENTITY IDS
    # -----------------------------------------------------

    before = len(pairs)

    pairs = pairs[
        pairs["source1_entity_id"].isin(
            lookup.index
        )
        &
        pairs["candidate_entity_id"].isin(
            lookup.index
        )
    ]

    skipped = before - len(pairs)

    if skipped:

        print(
            f"[features.py] warning: skipped "
            f"{skipped:,} pairs referencing "
            "unknown entity_ids"
        )

    pairs = pairs.reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # GET SOURCE 1 INFORMATION
    # -----------------------------------------------------

    print(
        "[features.py] looking up Source 1 "
        "business information..."
    )

    s1_info = lookup.loc[
        pairs["source1_entity_id"],
        [
            "business_name",
            "business_address"
        ]
    ].reset_index(
        drop=True
    )

    s1_info.columns = [
        "name1",
        "addr1"
    ]

    # -----------------------------------------------------
    # GET CANDIDATE INFORMATION
    # -----------------------------------------------------

    print(
        "[features.py] looking up candidate "
        "business information..."
    )

    c_info = lookup.loc[
        pairs["candidate_entity_id"],
        [
            "business_name",
            "business_address"
        ]
    ].reset_index(
        drop=True
    )

    c_info.columns = [
        "name2",
        "addr2"
    ]

    # -----------------------------------------------------
    # COMBINE DATA
    # -----------------------------------------------------

    merged = pd.concat(
        [
            pairs,
            s1_info,
            c_info
        ],
        axis=1
    )

    # -----------------------------------------------------
    # COMPUTE FEATURES
    # -----------------------------------------------------

    print(
        "[features.py] computing features "
        "(this is the slow-ish part, "
        "be patient)..."
    )

    feats = merged.apply(
        lambda r: compute_features(
            r["name1"],
            r["addr1"],
            r["name2"],
            r["addr2"]
        ),
        axis=1,
        result_type="expand"
    )

    # -----------------------------------------------------
    # COMBINE PAIRS + FEATURES
    # -----------------------------------------------------

    result = pd.concat(
        [
            merged[
                [
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            ],
            feats
        ],
        axis=1
    )

    # -----------------------------------------------------
    # LABEL PAIRS
    # -----------------------------------------------------

    print(
        "[features.py] labeling pairs "
        "against ground truth..."
    )

    result["label"] = [
        int(
            cand in gt_map.get(
                src,
                set()
            )
        )
        for src, cand in zip(
            result["source1_entity_id"],
            result["candidate_entity_id"]
        )
    ]

    # -----------------------------------------------------
    # FINAL COLUMN ORDER
    # -----------------------------------------------------

    cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "name_token_sort_ratio",
        "addr_token_sort_ratio",
        "name_jaro_winkler",
        "postal_exact_match",
        "label"
    ]

    return result[
        cols
    ]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a labeled training feature table "
            "from candidate pairs + ground truth."
        )
    )

    # Candidate pairs
    parser.add_argument(
        "--candidate",
        default="output/train_candidate_pairs.tsv"
    )

    # Ground truth
    parser.add_argument(
        "--ground-truth",
        default="dataset/train/train_ground_truth.tsv"
    )

    # Source files
    parser.add_argument(
        "--source1",
        default="dataset/train/train_source1.tsv"
    )

    parser.add_argument(
        "--source2",
        default="dataset/train/train_source2.tsv"
    )

    parser.add_argument(
        "--source3",
        default="dataset/train/train_source3.tsv"
    )

    # Output
    parser.add_argument(
        "--out",
        default="output/train_features.tsv"
    )

    # Maximum number of pairs
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Maximum number of candidate pairs "
            "to process. Example: 10000000"
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # BUILD TABLE
    # -----------------------------------------------------

    table = build_training_table(
        args.candidate,
        args.ground_truth,
        args.source1,
        args.source2,
        args.source3,
        max_pairs=args.max_pairs
    )

    # -----------------------------------------------------
    # SAVE OUTPUT
    # -----------------------------------------------------

    table.to_csv(
        args.out,
        sep="\t",
        index=False
    )

    # -----------------------------------------------------
    # PRINT SUMMARY
    # -----------------------------------------------------

    n_pos = int(
        table["label"].sum()
    )

    n_total = len(table)

    n_negative = n_total - n_pos

    print()
    print(
        "========================================"
    )
    print(
        "[features.py] DONE"
    )
    print(
        "========================================"
    )

    print(
        f"[features.py] wrote "
        f"{n_total:,} labeled pairs"
    )

    print(
        f"[features.py] positive pairs: "
        f"{n_pos:,}"
    )

    print(
        f"[features.py] negative pairs: "
        f"{n_negative:,}"
    )

    print(
        f"[features.py] output: "
        f"{args.out}"
    )

    print(
        "========================================"
    )


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()