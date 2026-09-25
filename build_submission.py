import pandas as pd
from normalize import normalize_name

def load(path, nrows=None):
    df = pd.read_csv(path, sep="\t", nrows=nrows,
                      usecols=["entity_id", "business_name", "country"])
    df["entity_id"] = df["entity_id"].str.strip()
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["block_key"] = df["country"].astype(str) + "_" + df["norm_name"].str[:8]
    return df

print("Loading test S1...")
s1 = load("dataset/test/test_source1.tsv")
print("Loading test S2...")
s2 = load("dataset/test/test_source2.tsv")
print("Loading test S3...")
s3 = load("dataset/test/test_source3.tsv")
pool = pd.concat([s2, s3], ignore_index=True)
del s2, s3

print("Building candidate lookup (capped at 50)...")
grouped = pool.groupby("block_key")["entity_id"].agg(list)
key_to_ids = {k: ",".join(v[:50]) for k, v in grouped.items()}

print("Building exact-name-match lookup for matches...")
name_to_ids = pool.groupby(["block_key", "norm_name"])["entity_id"].agg(list).to_dict()
del pool

s1["candidate_entity_ids"] = s1["block_key"].map(key_to_ids).fillna("")

def exact_matches(row):
    ids = name_to_ids.get((row["block_key"], row["norm_name"]), [])
    candidate_set = set(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"] else set()
    valid_ids = [i for i in ids if i in candidate_set]
    return ",".join(valid_ids[:20])

s1["matched_entity_ids"] = s1.apply(exact_matches, axis=1)

print("Writing candidate_pairs.tsv...")
cand_out = s1[["entity_id", "candidate_entity_ids"]].rename(columns={"entity_id": "source1_entity_id"})
cand_out.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)

print("Writing matching_results.tsv...")
match_out = s1[["entity_id", "matched_entity_ids"]].rename(columns={"entity_id": "source1_entity_id"})
match_out.to_csv("output/matching_results.tsv", sep="\t", index=False)

print("DONE")