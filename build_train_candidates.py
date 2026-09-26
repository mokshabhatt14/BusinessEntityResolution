"""
build_train_candidates.py  — memory-efficient blocking
Uses integer entity ID indices instead of strings to cut RAM by ~5x.
Output: output/train_candidate_pairs.tsv
"""
import gc
import collections
import pandas as pd
from normalize import build_block_keys

CAP_PER_KEY = 50    # reduced from 200 — enough for high recall, saves RAM
MAX_CANDS   = 300   # reduced from 500
CHUNK_SIZE  = 100_000

COLS = ["entity_id", "business_name", "business_address", "country"]
MAX_LINE_BYTES = 10_000   # skip any line longer than this (corrupt data)

def iter_tsv(path):
    """Stream rows as namedtuples, skipping corrupt/huge lines."""
    import io
    col_idx = None
    Row = None
    with open(path, "rb") as raw:
        for raw_line in raw:
            if len(raw_line) > MAX_LINE_BYTES:
                continue   # skip giant corrupt rows
            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                continue
            parts = line.split("\t")
            if col_idx is None:
                # parse header
                col_idx = {c: i for i, c in enumerate(parts)}
                missing = [c for c in COLS if c not in col_idx]
                if missing:
                    raise ValueError(f"Missing columns {missing} in {path}")
                from collections import namedtuple
                Row = namedtuple("Row", COLS)
                continue
            try:
                yield Row(
                    entity_id       = parts[col_idx["entity_id"]].strip(),
                    business_name   = parts[col_idx["business_name"]]   if col_idx["business_name"]    < len(parts) else "",
                    business_address= parts[col_idx["business_address"]] if col_idx["business_address"] < len(parts) else "",
                    country         = parts[col_idx["country"]]          if col_idx["country"]          < len(parts) else "",
                )
            except Exception:
                continue

# ── Step 1: pool — assign integer indices, build key index ───────────────
print("Step 1: indexing pool (S2+S3) ...")
pool_id_list = []           # pool_int_idx -> entity_id string
key_to_pool_ints: dict = {} # key -> [int, ...]
n_pool = 0

for path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
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

# ── Step 2: S1 — stream and build key index ───────────────────────────────
print("Step 2: indexing S1 ...")
key_to_s1_ints: dict = {}
s1_id_list = []
n_s1 = 0

for row in iter_tsv("dataset/train/train_source1.tsv"):
    idx = len(s1_id_list)
    s1_id_list.append(row.entity_id)
    for k in build_block_keys(row.country, row.business_name, row.business_address):
        key_to_s1_ints.setdefault(k, []).append(idx)
    n_s1 += 1
    if n_s1 % 1_000_000 == 0:
        print(f"    {n_s1:,} rows")

gc.collect()
print(f"  S1: {n_s1:,} rows, {len(key_to_s1_ints):,} keys")

# ── Step 3: join (all integers — very cheap) ──────────────────────────────
print("Step 3: joining ...")
s1_cands: dict = collections.defaultdict(set)  # s1_int -> set of pool_ints
common = set(key_to_pool_ints) & set(key_to_s1_ints)
print(f"  common keys: {len(common):,}")
for k in common:
    pool_ints = key_to_pool_ints[k]
    for s1_int in key_to_s1_ints[k]:
        s1_cands[s1_int].update(pool_ints)

del key_to_pool_ints, key_to_s1_ints, common
gc.collect()

# ── Step 4: write (convert ints back to strings only at write time) ───────
print("Step 4: writing output/train_candidate_pairs.tsv ...")
with open("output/train_candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_int, s1_eid in enumerate(s1_id_list):
        pool_ints = list(s1_cands.get(s1_int, set()))[:MAX_CANDS]
        cand_eids = [pool_id_list[i] for i in pool_ints]
        f.write(f"{s1_eid}\t{','.join(cand_eids)}\n")

n_with = len(s1_cands)
avg = sum(min(len(v), MAX_CANDS) for v in s1_cands.values()) / max(n_s1, 1)
print(f"  {n_s1:,} S1 rows, {n_with:,} with candidates, avg {avg:.0f} cands/S1")

# ── Step 5: recall check ─────────────────────────────────────────────────
print("Step 5: recall check ...")
s1_eid_to_int   = {eid: i for i, eid in enumerate(s1_id_list)}
pool_eid_to_int = {eid: i for i, eid in enumerate(pool_id_list)}
hit = total = 0

# Use raw line reader for GT (different columns)
with open("dataset/train/train_ground_truth.tsv", "rb") as f:
    header = f.readline().decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
    ci = {c: i for i, c in enumerate(header)}
    for raw_line in f:
        if total >= 20000:
            break
        if len(raw_line) > MAX_LINE_BYTES:
            continue
        parts = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
        try:
            s1_eid   = parts[ci["source1_entity_id"]].strip()
            match_str = parts[ci["matched_entity_ids"]] if ci["matched_entity_ids"] < len(parts) else ""
        except Exception:
            continue
        s1_int = s1_eid_to_int.get(s1_eid)
        if s1_int is None:
            continue
        cand_set = s1_cands.get(s1_int, set())
        for t in match_str.split(","):
            t = t.strip()
            if not t:
                continue
            total += 1
            if pool_eid_to_int.get(t) in cand_set:
                hit += 1

if total:
    print(f"  Recall: {hit}/{total} = {hit/total:.3f}")
print("DONE")
