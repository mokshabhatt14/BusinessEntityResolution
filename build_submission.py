"""
build_submission.py  — fast chunked blocking
Output: output/candidate_pairs.tsv
"""
import gc, os, collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 30
MAX_CANDS   = 200
CHUNK_SIZE  = 200_000

COLS = ["entity_id", "business_name", "business_address", "country"]

def iter_chunks(path):
    """Yield pandas chunks, using C engine for speed, skipping bad lines."""
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, usecols=COLS,
        chunksize=CHUNK_SIZE, on_bad_lines="skip", engine="c",
        encoding_errors="replace"
    ):
        yield chunk.fillna("")

# ── Step 1: index pool (S2+S3) ────────────────────────────────────────────
print("Step 1: indexing pool (S2+S3) ...")
key_to_pool_ints: dict = {}
pool_tmp = "output/pool_ids_test.tmp"

with open(pool_tmp, "w", encoding="utf-8") as pid_f:
    n_pool = 0
    for path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        print(f"  {path}")
        for chunk in iter_chunks(path):
            for row in chunk.itertuples(index=False):
                idx = n_pool
                pid_f.write(row.entity_id + "\n")
                for k in build_block_keys(row.country, row.business_name, row.business_address):
                    b = key_to_pool_ints.setdefault(k, [])
                    if len(b) < CAP_PER_KEY:
                        b.append(idx)
                n_pool += 1
            if n_pool % 1_000_000 == 0 or True:
                pass  # progress printed below
        print(f"    {n_pool:,} rows so far")
    print(f"  pool: {n_pool:,} rows, {len(key_to_pool_ints):,} keys")

gc.collect()

# ── Step 2: index S1 ──────────────────────────────────────────────────────
print("Step 2: indexing S1 ...")
key_to_s1_ints: dict = {}
s1_tmp = "output/s1_ids_test.tmp"

with open(s1_tmp, "w", encoding="utf-8") as sid_f:
    n_s1 = 0
    for chunk in iter_chunks("dataset/test/test_source1.tsv"):
        for row in chunk.itertuples(index=False):
            sid_f.write(row.entity_id + "\n")
            for k in build_block_keys(row.country, row.business_name, row.business_address):
                key_to_s1_ints.setdefault(k, []).append(n_s1)
            n_s1 += 1
    print(f"  S1: {n_s1:,} rows, {len(key_to_s1_ints):,} keys")

gc.collect()

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
print(f"  {len(s1_cands):,} S1 entities have candidates")

# ── Step 4: write output ──────────────────────────────────────────────────
print("Step 4: writing output/candidate_pairs.tsv ...")
with open(pool_tmp, encoding="utf-8") as f:
    pool_id_list = [line.rstrip("\n") for line in f]

with open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as out_f:
    out_f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(s1_tmp, encoding="utf-8") as sid_f:
        for s1_int, s1_eid in enumerate(sid_f):
            s1_eid = s1_eid.rstrip("\n")
            pool_ints = list(s1_cands.get(s1_int, set()))[:MAX_CANDS]
            cand_eids = [pool_id_list[i] for i in pool_ints]
            out_f.write(f"{s1_eid}\t{','.join(cand_eids)}\n")

os.remove(pool_tmp)
os.remove(s1_tmp)
avg = sum(min(len(v), MAX_CANDS) for v in s1_cands.values()) / max(n_s1, 1)
print(f"  Done: {n_s1:,} S1 rows, avg {avg:.0f} cands/S1")
print("DONE")
