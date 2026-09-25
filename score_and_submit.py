# score_and_submit.py
#
# Final test scoring and submission generation for Business Entity Resolution.
#
# IMPORTANT:
# - Uses the EXACT same four features as features.py
# - Uses the already-trained matcher_model.joblib
# - Processes test candidates in batches to avoid creating a 59M-row DataFrame
# - Does NOT change the matching methodology
# - Threshold can be supplied with --threshold

import argparse
import re
import sys

import joblib
import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# ---------------------------------------------------------------------
# Exact postal-code logic from features.py
# ---------------------------------------------------------------------

_POSTAL_RE = re.compile(r"\b(\d{4,6})\b")


def _extract_postal_code(address):
    """
    EXACTLY matches the logic used in features.py.

    Finds all 4-6 digit sequences and returns the LAST one.
    """
    if not isinstance(address, str) or not address.strip():
        return None

    matches = _POSTAL_RE.findall(address)
    return matches[-1] if matches else None


# ---------------------------------------------------------------------
# Exact feature calculation from features.py
# ---------------------------------------------------------------------

def compute_features(name1, addr1, name2, addr2):
    """
    EXACT same feature definitions used during training.
    """

    name1 = name1 if isinstance(name1, str) else ""
    name2 = name2 if isinstance(name2, str) else ""

    addr1 = addr1 if isinstance(addr1, str) else ""
    addr2 = addr2 if isinstance(addr2, str) else ""

    # Feature 1
    name_token_sort = fuzz.token_sort_ratio(name1, name2)

    # Feature 2
    addr_token_sort = fuzz.token_sort_ratio(addr1, addr2)

    # Feature 3
    name_jaro_winkler = JaroWinkler.similarity(name1, name2) * 100.0

    # Feature 4
    pc1 = _extract_postal_code(addr1)
    pc2 = _extract_postal_code(addr2)

    postal_exact_match = bool(
        pc1 and pc2 and pc1 == pc2
    )

    return {
        "name_token_sort_ratio": name_token_sort,
        "addr_token_sort_ratio": addr_token_sort,
        "name_jaro_winkler": name_jaro_winkler,
        "postal_exact_match": postal_exact_match,
    }


# ---------------------------------------------------------------------
# Load test entity lookup
# ---------------------------------------------------------------------

def load_entity_lookup(source1_path, source2_path, source3_path):

    print("[score_and_submit.py] loading test source files...")

    s1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str
    ).fillna("")

    s2 = pd.read_csv(
        source2_path,
        sep="\t",
        dtype=str
    ).fillna("")

    s3 = pd.read_csv(
        source3_path,
        sep="\t",
        dtype=str
    ).fillna("")

    lookup = pd.concat(
        [s1, s2, s3],
        ignore_index=True
    )

    lookup = lookup[
        [
            "entity_id",
            "business_name",
            "business_address"
        ]
    ]

    lookup = lookup.set_index("entity_id")

    print(
        f"[score_and_submit.py] "
        f"loaded {len(lookup):,} unique entities"
    )

    return lookup, s1["entity_id"].tolist()


# ---------------------------------------------------------------------
# Score one batch
# ---------------------------------------------------------------------

def score_batch(
    source1_ids,
    candidate_ids,
    lookup,
    model,
    threshold,
    pred_map
):
    """
    Scores one batch of candidate pairs.

    Uses RapidFuzz batch operations while preserving the exact
    feature definitions from features.py.
    """

    if len(source1_ids) == 0:
        return 0, 0

    # -------------------------------------------------------------
    # Lookup entity information
    # -------------------------------------------------------------

    s1_rows = lookup.reindex(source1_ids)
    s2_rows = lookup.reindex(candidate_ids)

    names1 = s1_rows["business_name"].fillna("").to_numpy(dtype=object)
    names2 = s2_rows["business_name"].fillna("").to_numpy(dtype=object)

    addrs1 = s1_rows["business_address"].fillna("").to_numpy(dtype=object)
    addrs2 = s2_rows["business_address"].fillna("").to_numpy(dtype=object)

    # -------------------------------------------------------------
    # Exact Feature 1: token-sort ratio
    # -------------------------------------------------------------

    name_token_sort = np.asarray(
        [
            fuzz.token_sort_ratio(a, b)
            for a, b in zip(names1, names2)
        ],
        dtype=np.float64
    )

    # -------------------------------------------------------------
    # Exact Feature 2: address token-sort ratio
    # -------------------------------------------------------------

    addr_token_sort = np.asarray(
        [
            fuzz.token_sort_ratio(a, b)
            for a, b in zip(addrs1, addrs2)
        ],
        dtype=np.float64
    )

    # -------------------------------------------------------------
    # Exact Feature 3: Jaro-Winkler * 100
    #
    # This deliberately uses the SAME operation as features.py.
    # -------------------------------------------------------------

    name_jaro_winkler = np.asarray(
        [
            JaroWinkler.similarity(a, b) * 100.0
            for a, b in zip(names1, names2)
        ],
        dtype=np.float64
    )

    # -------------------------------------------------------------
    # Exact Feature 4: postal code equality
    #
    # features.py uses the LAST 4-6 digit sequence.
    # -------------------------------------------------------------

    postal_exact_match = np.zeros(
        len(addrs1),
        dtype=np.float64
    )

    for i, (a1, a2) in enumerate(zip(addrs1, addrs2)):

        pc1 = _extract_postal_code(a1)
        pc2 = _extract_postal_code(a2)

        if pc1 and pc2 and pc1 == pc2:
            postal_exact_match[i] = 1.0

    # -------------------------------------------------------------
    # Build EXACT same feature order used by train_model.py
    # -------------------------------------------------------------

    X = pd.DataFrame(
        {
            "name_token_sort_ratio": name_token_sort,
            "addr_token_sort_ratio": addr_token_sort,
            "name_jaro_winkler": name_jaro_winkler,
            "postal_exact_match": postal_exact_match,
        }
    )

    # -------------------------------------------------------------
    # Model prediction
    # -------------------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    matched_indices = np.flatnonzero(
        probabilities >= threshold
    )

    # -------------------------------------------------------------
    # Store predicted matches
    # -------------------------------------------------------------

    for idx in matched_indices:

        s1_id = source1_ids[idx]
        candidate_id = candidate_ids[idx]

        if s1_id not in pred_map:
            pred_map[s1_id] = set()

        pred_map[s1_id].add(candidate_id)

    return len(source1_ids), len(matched_indices)


# ---------------------------------------------------------------------
# Process candidate file in chunks
# ---------------------------------------------------------------------

def process_test_candidates(
    candidate_path,
    lookup,
    model,
    threshold,
    chunk_size=5000
):

    print(
        "[score_and_submit.py] "
        "processing test candidates in chunks..."
    )

    pred_map = {}

    total_pairs = 0
    total_matches = 0
    chunk_number = 0

    # Read the ORIGINAL candidate rows in chunks.
    #
    # We intentionally do NOT load the entire candidate file into
    # memory because it expands to approximately 59M pairs.
    reader = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size
    )

    for chunk in reader:

        chunk = chunk.fillna("")

        batch_s1 = []
        batch_candidate = []

        # ---------------------------------------------------------
        # Explode candidate_entity_ids
        # ---------------------------------------------------------

        for row in chunk.itertuples(index=False):

            s1_id = row.source1_entity_id
            candidate_string = row.candidate_entity_ids

            if not candidate_string:
                continue

            candidate_ids = candidate_string.split(",")

            for candidate_id in candidate_ids:

                candidate_id = candidate_id.strip()

                if not candidate_id:
                    continue

                batch_s1.append(s1_id)
                batch_candidate.append(candidate_id)

        # ---------------------------------------------------------
        # Nothing in this chunk
        # ---------------------------------------------------------

        if not batch_s1:
            continue

        # ---------------------------------------------------------
        # Score this batch
        # ---------------------------------------------------------

        processed, matches = score_batch(
            batch_s1,
            batch_candidate,
            lookup,
            model,
            threshold,
            pred_map
        )

        total_pairs += processed
        total_matches += matches

        chunk_number += 1

        print(
            f"[score_and_submit.py] "
            f"chunk {chunk_number}: "
            f"processed {total_pairs:,} candidate pairs | "
            f"matches {total_matches:,}",
            flush=True
        )

    print(
        f"[score_and_submit.py] "
        f"finished processing {total_pairs:,} candidate pairs"
    )

    print(
        f"[score_and_submit.py] "
        f"predicted {total_matches:,} matches"
    )

    return pred_map


# ---------------------------------------------------------------------
# Write final submission
# ---------------------------------------------------------------------

def write_matching_results(
    all_s1_ids,
    pred_map,
    output_path
):

    print(
        "[score_and_submit.py] "
        "writing final submission..."
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        f.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        for s1_id in all_s1_ids:

            matched_ids = sorted(
                pred_map.get(s1_id, set())
            )

            f.write(
                f"{s1_id}\t{','.join(matched_ids)}\n"
            )

    print(
        f"[score_and_submit.py] "
        f"wrote {len(all_s1_ids):,} rows to "
        f"{output_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Score test candidates and create submission."
    )

    parser.add_argument(
        "--model",
        default="output/matcher_model.joblib"
    )

    parser.add_argument(
        "--test-candidates",
        default="output/candidate_pairs.tsv"
    )

    parser.add_argument(
        "--test-source1",
        default="dataset/test/test_source1.tsv"
    )

    parser.add_argument(
        "--test-source2",
        default="dataset/test/test_source2.tsv"
    )

    parser.add_argument(
        "--test-source3",
        default="dataset/test/test_source3.tsv"
    )

    parser.add_argument(
        "--out",
        default="output/matching_results.tsv"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Load trained model
    # -------------------------------------------------------------

    print(
        "[score_and_submit.py] "
        "loading trained model..."
    )

    model = joblib.load(args.model)

    print(
        f"[score_and_submit.py] "
        f"using threshold={args.threshold}"
    )

    # -------------------------------------------------------------
    # Load test entities
    # -------------------------------------------------------------

    lookup, all_s1_ids = load_entity_lookup(
        args.test_source1,
        args.test_source2,
        args.test_source3
    )

    # -------------------------------------------------------------
    # Process all test candidates
    # -------------------------------------------------------------

    pred_map = process_test_candidates(
        args.test_candidates,
        lookup,
        model,
        args.threshold,
        chunk_size=args.chunk_size
    )

    # -------------------------------------------------------------
    # Write submission
    # -------------------------------------------------------------

    write_matching_results(
        all_s1_ids,
        pred_map,
        args.out
    )

    print()
    print("=" * 40)
    print("SUBMISSION CREATED")
    print("=" * 40)
    print(f"File: {args.out}")
    print()
    print("Validate with:")
    print(
        "python utils/validate_submission.py "
        "--matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv "
        "--test-dir dataset/test"
    )


if __name__ == "__main__":
    main()