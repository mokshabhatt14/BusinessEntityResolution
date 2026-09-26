"""
build_submission.py  — fast vectorized multi-key blocking
Output: output/candidate_pairs.tsv

Same O(total-keys) join strategy as build_train_candidates.py.
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 200
MAX_CANDS   = 500

print("Loading test S1...")
s1 = pd.read_csv(
    "dataset/test/test_source1.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
print(f"S1: {len(s1):,} rows")

print("Loading test S2 + S3...")
s2 = pd.read_csv(
    "dataset/test/test_source2.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
s3 = pd.read_csv(
    "dataset/test/test_source3.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()
print(f"Pool: {len(pool):,} rows")

# ── Step 1: pool key index ─────────────────────────────────────────────────
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

# ── Step 2: S1 key index ───────────────────────────────────────────────────
print("Building S1 key index...")
key_to_s1_ids: dict = {}
s1_ids_list = s1["entity_id"].tolist()
for row in s1.itertuples(index=False):
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        key_to_s1_ids.setdefault(k, []).append(row.entity_id)
print(f"S1 key index: {len(key_to_s1_ids):,} keys")

# ── Step 3: join ───────────────────────────────────────────────────────────
print("Joining pool → S1 candidates...")
s1_candidates: dict = collections.defaultdict(set)
common_keys = set(key_to_pool_ids) & set(key_to_s1_ids)
print(f"Common keys: {len(common_keys):,}")
for k in common_keys:
    pool_ids = key_to_pool_ids[k]
    s1_ids   = key_to_s1_ids[k]
    for sid in s1_ids:
        s1_candidates[sid].update(pool_ids)

del key_to_pool_ids, key_to_s1_ids
gc.collect()

# ── Step 4: write output ───────────────────────────────────────────────────
print("Writing output/candidate_pairs.tsv ...")
with open("output/candidate_pairs.tsv", "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids_list:
        cands = s1_candidates.get(sid, set())
        cand_list = list(cands)[:MAX_CANDS]
        f.write(f"{sid}\t{','.join(cand_list)}\n")

total_cands = sum(min(len(v), MAX_CANDS) for v in s1_candidates.values())
print(f"Done. {len(s1_ids_list):,} S1 rows, avg {total_cands/len(s1_ids_list):.1f} cands/S1")
print("DONE")
