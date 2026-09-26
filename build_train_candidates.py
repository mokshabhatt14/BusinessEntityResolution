"""
build_train_candidates.py  — fast vectorized multi-key blocking
Output: output/train_candidate_pairs.tsv

Strategy: build keys for pool AND s1, then join via dict — O(total keys),
no per-row Python loop over S1.
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 200   # max pool entities per key bucket
MAX_CANDS   = 500   # max candidates per S1 entity

def build_key_lists(df):
    """Return list-of-lists: for each row, the list of blocking keys."""
    result = []
    for row in df.itertuples(index=False):
        result.append(list(build_block_keys(row.country, row.business_name, row.business_address)))
    return result

print("Loading train S1...")
s1 = pd.read_csv(
    "dataset/train/train_source1.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
print(f"S1: {len(s1):,} rows")

print("Loading train S2 + S3...")
s2 = pd.read_csv(
    "dataset/train/train_source2.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
s3 = pd.read_csv(
    "dataset/train/train_source3.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
print(f"Pool: {len(pool):,} rows")

# ── Step 1: build pool index (key → [eid, ...]) ──────────────────────────
print("Building pool key index...")
key_to_pool_ids: dict = {}
for row in pool.itertuples(index=False):
    eid = row.entity_id
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        bucket = key_to_pool_ids.setdefault(k, [])
        if len(bucket) < CAP_PER_KEY:
            bucket.append(eid)
del pool
gc.collect()
print(f"Pool index: {len(key_to_pool_ids):,} keys")

# ── Step 2: for each S1, collect union of candidates via keys ─────────────
# Inverted approach: build s1_key→[s1_id] index, then sweep pool keys once.
# This avoids a Python loop over 2.2M S1 rows for the join.
print("Building S1 key index...")
key_to_s1_ids: dict = {}
s1_ids_list = s1["entity_id"].tolist()
s1_key_lists = []
for row in s1.itertuples(index=False):
    ks = list(build_block_keys(row.country, row.business_name, row.business_address))
    s1_key_lists.append(ks)
    for k in ks:
        key_to_s1_ids.setdefault(k, []).append(row.entity_id)

print(f"S1 key index: {len(key_to_s1_ids):,} keys")

# ── Step 3: join — for each pool key, propagate pool ids to s1 candidates ─
print("Joining pool -> S1 candidates...")
s1_candidates: dict = collections.defaultdict(set)  # s1_id -> set of pool ids

common_keys = set(key_to_pool_ids) & set(key_to_s1_ids)
print(f"Common keys (join size): {len(common_keys):,}")

for k in common_keys:
    pool_ids = key_to_pool_ids[k]
    s1_ids   = key_to_s1_ids[k]
    for sid in s1_ids:
        s1_candidates[sid].update(pool_ids)

del key_to_pool_ids, key_to_s1_ids
gc.collect()

# ── Step 4: write output ───────────────────────────────────────────────────
print("Writing output/train_candidate_pairs.tsv ...")
with open("output/train_candidate_pairs.tsv", "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids_list:
        cands = s1_candidates.get(sid, set())
        cand_list = list(cands)[:MAX_CANDS]
        f.write(f"{sid}\t{','.join(cand_list)}\n")

total_cands = sum(min(len(v), MAX_CANDS) for v in s1_candidates.values())
print(f"Done. {len(s1_ids_list):,} S1 rows, avg {total_cands/len(s1_ids_list):.1f} candidates/S1")  # noqa

# ── Quick recall check ────────────────────────────────────────────────────
print("\nRecall check (first 5000 GT rows)...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
hit, total = 0, 0
for _, row in gt.head(5000).iterrows():
    sid = row["source1_entity_id"]
    true_ids = [i.strip() for i in row["matched_entity_ids"].split(",") if i.strip()]
    cands = s1_candidates.get(sid, set())
    for t in true_ids:
        total += 1
        if t in cands:
            hit += 1
if total:
    print(f"Recall: {hit}/{total} = {hit/total:.3f}")
print("DONE")
