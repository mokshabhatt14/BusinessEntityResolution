"""
build_submission.py  — memory-efficient blocking
Uses integer entity ID indices instead of strings to cut RAM by ~5x.
Output: output/candidate_pairs.tsv
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 50
MAX_CANDS   = 300
CHUNK_SIZE  = 100_000

COLS = ["entity_id", "business_name", "business_address", "country"]

def iter_tsv(path):
    for chunk in pd.read_csv(path, sep="\t", dtype=str, usecols=COLS,
                             chunksize=CHUNK_SIZE, on_bad_lines="skip",
                             engine="python"):
        yield from chunk.fillna("").itertuples(index=False)

# ── Step 1: pool key index (S2+S3), integer-based ────────────────────────
print("Step 1: indexing pool (S2+S3) ...")
pool_id_list = []
key_to_pool_ints: dict = {}
n_pool = 0

for path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
    print(f"  {path}")
    for row in iter_tsv(path):
        idx = len(pool_id_list)
        pool_id_list.append(row.entity_id)
        for k in build_block_keys(row.country, row.business_name, row.business_address):
            b = key_to_pool_ints.setdefault(k, [])
            if len(b) < CAP_PER_KEY:
                b.append(idx)
        n_pool += 1
        if n_pool % 1_000_000 == 0:
            print(f"    {n_pool:,} rows")

gc.collect()
print(f"  pool: {n_pool:,} rows, {len(key_to_pool_ints):,} keys")

# ── Step 2: S1 key index, integer-based ──────────────────────────────────
print("Step 2: indexing S1 ...")
key_to_s1_ints: dict = {}
s1_id_list = []
n_s1 = 0

for row in iter_tsv("dataset/test/test_source1.tsv"):
    idx = len(s1_id_list)
    s1_id_list.append(row.entity_id)
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        key_to_s1_ints.setdefault(k, []).append(idx)
    n_s1 += 1
    if n_s1 % 1_000_000 == 0:
        print(f"    {n_s1:,} rows")

gc.collect()
print(f"  S1: {n_s1:,} rows, {len(key_to_s1_ints):,} keys")

# ── Step 3: join ──────────────────────────────────────────────────────────
print("Step 3: joining ...")
s1_cands: dict = collections.defaultdict(set)
common = set(key_to_pool_ints) & set(key_to_s1_ints)
print(f"  common keys: {len(common):,}")
for k in common:
    for s1_int in key_to_s1_ints[k]:
        s1_cands[s1_int].update(key_to_pool_ints[k])

del key_to_pool_ints, key_to_s1_ints, common
gc.collect()

# ── Step 4: write ─────────────────────────────────────────────────────────
print("Step 4: writing output/candidate_pairs.tsv ...")
with open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_int, s1_eid in enumerate(s1_id_list):
        pool_ints = list(s1_cands.get(s1_int, set()))[:MAX_CANDS]
        cand_eids = [pool_id_list[i] for i in pool_ints]
        f.write(f"{s1_eid}\t{','.join(cand_eids)}\n")

n_with = len(s1_cands)
avg = sum(min(len(v), MAX_CANDS) for v in s1_cands.values()) / max(n_s1, 1)
print(f"  {n_s1:,} S1 rows, {n_with:,} with candidates, avg {avg:.0f} cands/S1")
print("DONE")
