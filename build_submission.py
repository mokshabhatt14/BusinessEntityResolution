"""
build_submission.py  — streaming, minimal RAM usage
Reads every file in chunks, never holds more than one chunk at a time.
Output: output/candidate_pairs.tsv
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 200
MAX_CANDS   = 500
CHUNK_SIZE  = 100_000

COLS = ["entity_id", "business_name", "business_address", "country"]

def iter_tsv(path):
    """Yield rows one at a time, skipping bad lines."""
    for chunk in pd.read_csv(path, sep="\t", dtype=str, usecols=COLS,
                             chunksize=CHUNK_SIZE, on_bad_lines="skip",
                             engine="python"):
        yield from chunk.fillna("").itertuples(index=False)

# ── Step 1: pool key index (S2 + S3), streaming ───────────────────────────
print("Building pool key index (S2 + S3) ...")
key_to_pool_ids: dict = {}
n_pool = 0
for path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
    print(f"  {path}")
    for row in iter_tsv(path):
        for k in build_block_keys(row.country, row.business_name, row.business_address):
            b = key_to_pool_ids.setdefault(k, [])
            if len(b) < CAP_PER_KEY:
                b.append(row.entity_id)
        n_pool += 1
        if n_pool % 500_000 == 0:
            print(f"    {n_pool:,} pool rows processed")
gc.collect()
print(f"Pool index: {len(key_to_pool_ids):,} keys from {n_pool:,} rows")

# ── Step 2: S1 key index, streaming ──────────────────────────────────────
print("Building S1 key index ...")
key_to_s1_ids: dict = {}
s1_ids_order = []
n_s1 = 0
for row in iter_tsv("dataset/test/test_source1.tsv"):
    s1_ids_order.append(row.entity_id)
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        key_to_s1_ids.setdefault(k, []).append(row.entity_id)
    n_s1 += 1
    if n_s1 % 500_000 == 0:
        print(f"    {n_s1:,} S1 rows processed")
gc.collect()
print(f"S1 index: {len(key_to_s1_ids):,} keys from {n_s1:,} rows")

# ── Step 3: join ──────────────────────────────────────────────────────────
print("Joining ...")
s1_candidates: dict = collections.defaultdict(set)
common = set(key_to_pool_ids) & set(key_to_s1_ids)
print(f"Common keys: {len(common):,}")
for k in common:
    pool_ids = key_to_pool_ids[k]
    for sid in key_to_s1_ids[k]:
        s1_candidates[sid].update(pool_ids)
del key_to_pool_ids, key_to_s1_ids, common
gc.collect()

# ── Step 4: write output ──────────────────────────────────────────────────
print("Writing output/candidate_pairs.tsv ...")
with open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids_order:
        cands = list(s1_candidates.get(sid, set()))[:MAX_CANDS]
        f.write(f"{sid}\t{','.join(cands)}\n")

n_with = sum(1 for sid in s1_ids_order if sid in s1_candidates)
avg = sum(min(len(v), MAX_CANDS) for v in s1_candidates.values()) / max(len(s1_ids_order), 1)
print(f"Done: {len(s1_ids_order):,} S1 rows, {n_with:,} with candidates, avg {avg:.0f} cands/S1")
print("DONE")
