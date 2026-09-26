"""
build_submission.py  — chunked multi-key blocking (low memory)
Output: output/candidate_pairs.tsv
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 200
MAX_CANDS   = 500
CHUNK_SIZE  = 200_000

print("Loading test S1...")
s1 = pd.read_csv(
    "dataset/test/test_source1.tsv", sep="\t", dtype=str,
    usecols=["entity_id", "business_name", "business_address", "country"]
).fillna("")
print(f"S1: {len(s1):,} rows")

# ── Step 1: pool key index in chunks ──────────────────────────────────────
print("Building pool key index from S2 + S3 in chunks...")
key_to_pool_ids: dict = {}

for src_path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
    print(f"  reading {src_path}...")
    for chunk in pd.read_csv(src_path, sep="\t", dtype=str,
                              usecols=["entity_id", "business_name", "business_address", "country"],
                              chunksize=CHUNK_SIZE):
        chunk = chunk.fillna("")
        for row in chunk.itertuples(index=False):
            for k in build_block_keys(row.country, row.business_name, row.business_address):
                bucket = key_to_pool_ids.setdefault(k, [])
                if len(bucket) < CAP_PER_KEY:
                    bucket.append(row.entity_id)
        gc.collect()

print(f"Pool index: {len(key_to_pool_ids):,} keys")

# ── Step 2: S1 key index ──────────────────────────────────────────────────
print("Building S1 key index...")
key_to_s1_ids: dict = {}
s1_ids_list = s1["entity_id"].tolist()
for row in s1.itertuples(index=False):
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        key_to_s1_ids.setdefault(k, []).append(row.entity_id)
del s1
gc.collect()
print(f"S1 key index: {len(key_to_s1_ids):,} keys")

# ── Step 3: join ──────────────────────────────────────────────────────────
print("Joining pool -> S1 candidates...")
s1_candidates: dict = collections.defaultdict(set)
common_keys = set(key_to_pool_ids) & set(key_to_s1_ids)
print(f"Common keys: {len(common_keys):,}")
for k in common_keys:
    pool_ids = key_to_pool_ids[k]
    for sid in key_to_s1_ids[k]:
        s1_candidates[sid].update(pool_ids)

del key_to_pool_ids, key_to_s1_ids
gc.collect()

# ── Step 4: write output ──────────────────────────────────────────────────
print("Writing output/candidate_pairs.tsv ...")
with open("output/candidate_pairs.tsv", "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids_list:
        cands = list(s1_candidates.get(sid, set()))[:MAX_CANDS]
        f.write(f"{sid}\t{','.join(cands)}\n")

n_with_cands = sum(1 for sid in s1_ids_list if sid in s1_candidates)
avg = sum(min(len(v), MAX_CANDS) for v in s1_candidates.values()) / max(len(s1_ids_list), 1)
print(f"Done. {len(s1_ids_list):,} S1 rows, {n_with_cands:,} with candidates, avg {avg:.1f} cands/S1")
print("DONE")
