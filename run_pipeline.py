"""
run_pipeline.py — runs the complete pipeline end to end with timing.
Steps:
  1. build_train_candidates  (new blocking, ~8 min)
  2. features.py             (compute 14 features, ~15 min)
  3. train_model.py          (train HGBC, ~5 min)
  4. build_submission.py     (test blocking, ~8 min)
  5. score_and_submit.py     (score + write matching_results.tsv)
"""
import subprocess, sys, time, os

os.environ["PYTHONIOENCODING"] = "utf-8"
py = sys.executable

def run(label, args, timeout=None):
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    r = subprocess.run([py] + args, timeout=timeout)
    elapsed = time.time() - t0
    print(f"\n[{label}] finished in {elapsed/60:.1f} min, exit={r.returncode}", flush=True)
    if r.returncode != 0:
        print(f"ERROR in {label}, aborting.")
        sys.exit(1)

run("build_train_candidates", ["build_train_candidates.py"])
run("features",               ["features.py"])
run("train_model",            ["train_model.py"])
run("build_submission",       ["build_submission.py"])
run("score_and_submit",       ["score_and_submit.py"])

print("\n" + "="*60)
print("PIPELINE COMPLETE — submit output/matching_results.tsv")
print("="*60)
