# BusinessEntityResolution

## Quick Start (for teammates)

### 1. Clone & setup
```bash
git clone https://github.com/mokshabhatt14/BusinessEntityResolution.git
cd BusinessEntityResolution
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the dataset
Place the competition dataset files like this (NOT committed to git):
```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

### 3. Run the full pipeline
```bash
python run_pipeline.py
```
This runs all 5 steps automatically (~50-60 min total):
1. **build_train_candidates.py** — multi-key blocking on train set (~8 min)
2. **features.py** — compute 14 similarity features (~20-30 min)
3. **train_model.py** — train HGBC classifier + tune threshold (~5 min)
4. **build_submission.py** — multi-key blocking on test set (~8 min)
5. **score_and_submit.py** — score test candidates → submission file (~10 min)

### 4. Submit
Upload `output/matching_results.tsv` (zip it first if required).

---

## Pipeline Files

| File | Role |
|------|------|
| `normalize.py` | Name/address normalization + multi-key blocking logic |
| `build_train_candidates.py` | Generate train candidate pairs |
| `build_submission.py` | Generate test candidate pairs |
| `features.py` | 14 pairwise similarity features + `FEATURE_COLS` list |
| `train_model.py` | Train HistGradientBoostingClassifier, tune F0.5 threshold |
| `score_and_submit.py` | Score test candidates, write matching_results.tsv |
| `run_pipeline.py` | Runs all steps above in order |

## Blocking Strategy (why it's better now)

Old approach: single sorted-token key → **~19% recall**

New approach — 5 key types per record:
- **NP**: country + sorted pair of name content tokens
- **NT**: country + individual long name token (≥7 chars)
- **ND**: country + first 12 chars of normalized name
- **Z**: country + postal code from address
- **AP**: country + sorted pair of address content tokens

Expected recall: **~95%** with ~300 candidates per S1 entity.

## Features (14 total)

| Feature | Description |
|---------|-------------|
| `name_token_sort_ratio` | Token-sort fuzzy ratio |
| `name_token_set_ratio` | Token-set fuzzy ratio (subset-robust) |
| `name_partial_ratio` | Partial string fuzzy ratio |
| `name_jaro_winkler` | Jaro-Winkler × 100 |
| `name_exact_match` | 1 if normalized names identical |
| `name_len_diff` | Absolute character length difference |
| `addr_token_sort_ratio` | Token-sort ratio on addresses |
| `addr_token_set_ratio` | Token-set ratio on addresses |
| `addr_partial_ratio` | Partial ratio on addresses |
| `postal_exact_match` | 1 if postal codes match |
| `country_exact_match` | 1 if country codes match |
| `name_num_common_tokens` | Count of shared name tokens |
| `name_jaccard` | Jaccard similarity on name token sets |
| `addr_jaccard` | Jaccard similarity on address token sets |

## Best Scores

| Submission | Score |
|------------|-------|
| #3 (old) | 0.359 |
| #2 (old) | 0.295 |
| #1 (old) | 0.354 |
| **new pipeline** | **TBD — run and submit!** |
