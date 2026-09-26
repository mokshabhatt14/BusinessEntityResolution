"""
build_submission.py  — ultra-low-memory blocking
Writes pool IDs to a temp file so they never live in RAM.
Output: output/candidate_pairs.tsv
"""
import gc, os, collections
from normalize import build_block_keys

CAP_PER_KEY = 30
MAX_CANDS   = 200
MAX_LINE_BYTES = 10_000

COLS = ["entity_id", "business_name", "business_address", "country"]

def iter_tsv(path):
    from collections import namedtuple
    col_idx = None
    Row = None
    with open(path, "rb") as f:
        for raw in f:
            if len(raw) > MAX_LINE_BYTES:
                continue
            try:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                continue
            parts = line.split("\t")
            if col_idx is None:
                col_idx = {c: i for i, c in enumerate(parts)}
                if any(c not in col_idx for c in COLS):
                    continue
                Row = namedtuple("Row", COLS)
                continue
            if Row is None:
                continue
            try:
                yield Row(
                    entity_id        = parts[col_idx["entity_id"]].strip(),
                    business_name    = parts[col_idx["business_name"]]    if col_idx["business_name"]    < len(parts) else "",
                    business_address = parts[col_idx["business_address"]] if col_idx["business_address"] < len(parts) else "",
                    country          = parts[col_idx["country"]]          if col_idx["country"]          < len(parts) else "",
                )
            except Exception:
                continue

# ── Step 1: index pool, write IDs to temp file ───────────────────────────
print("Step 1: indexing pool (S2+S3) ...")
key_to_pool_ints: dict = {}
pool_tmp = "output/pool_ids_test.tmp"

with open(pool_tmp, "w", encoding="utf-8") as pid_f:
    n_pool = 0
    for path in ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"]:
        print(f"  {path}")
        for row in iter_tsv(path):
            idx = n_pool
            pid_f.write(row.entity_id + "\n")
            for k in build_block_keys(row.country, row.business_name, row.business_address):
                b = key_to_pool_ints.setdefault(k, [])
                if len(b) < CAP_PER_KEY:
                    b.append(idx)
            n_pool += 1
            if n_pool % 1_000_000 == 0:
                print(f"    {n_pool:,} rows")

gc.collect()
print(f"  pool: {n_pool:,} rows, {len(key_to_pool_ints):,} keys")

# ── Step 2: index S1, write IDs to temp file ─────────────────────────────
print("Step 2: indexing S1 ...")
key_to_s1_ints: dict = {}
s1_tmp = "output/s1_ids_test.tmp"

with open(s1_tmp, "w", encoding="utf-8") as sid_f:
    n_s1 = 0
    for row in iter_tsv("dataset/test/test_source1.tsv"):
        idx = n_s1
        sid_f.write(row.entity_id + "\n")
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
