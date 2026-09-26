import pandas as pd
import gc
from normalize import make_block_keys_series

def load_and_normalize(path, nrows=None):
    df = pd.read_csv(path, sep="\t", nrows=nrows, usecols=["entity_id", "business_name", "business_address", "country"])
    df["entity_id"] = df["entity_id"].str.strip()
    df["block_key"] = make_block_keys_series(df["country"], df["business_name"])
    return df[["entity_id", "block_key"]]  # drop unneeded columns to save memory

print("Loading S1 (full)...")
s1 = load_and_normalize("dataset/train/train_source1.tsv")
print("Loaded S1:", len(s1))

print("Loading S2 (full file)...")
s2 = load_and_normalize("dataset/train/train_source2.tsv")
print("Loaded S2:", len(s2))

print("Loading S3 (full file)...")
s3 = load_and_normalize("dataset/train/train_source3.tsv")
print("Loaded S3:", len(s3))

print("Building pool...")
pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3
gc.collect()

print("Building lookup dictionary (one pass)...")
key_to_ids = pool.groupby("block_key")["entity_id"].agg(list).to_dict()
del pool
gc.collect()

print("Matching S1 entities...")
rows = []
for i, (_, s1_row) in enumerate(s1.iterrows()):
    if i % 100 == 0:
        print(f"  processed {i} / {len(s1)}")
    key = s1_row["block_key"]
    candidates = key_to_ids.get(key, [])
    rows.append({"source1_entity_id": s1_row["entity_id"], "candidate_entity_ids": ",".join(candidates)})

print("Writing output...")
candidates_df = pd.DataFrame(rows)
candidates_df.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
print("Done!")

print("Checking recall against ground truth...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
gt["source1_entity_id"] = gt["source1_entity_id"].str.strip()

merged = gt.merge(candidates_df, on="source1_entity_id", how="inner")
hit_count, total_count = 0, 0
for _, row in merged.iterrows():
    true_ids = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) and row["matched_entity_ids"] else []
    cand_ids = str(row["candidate_entity_ids"]).split(",") if pd.notna(row["candidate_entity_ids"]) else []
    for t in true_ids:
        total_count += 1
        if t in cand_ids:
            hit_count += 1

if total_count > 0:
    print(f"Recall: {hit_count}/{total_count} = {hit_count/total_count:.3f}")
else:
    print("No true matches in this sample to check.")