"""Evaluate cached OAT generations with OAT's own FEA evaluation script (scripts/RunComplianceTest.py, run unchanged).

1. merge the per-GPU shards of one configuration into the list-of-lists format of GenerateSolutions.py
   (generated_samples[problem][sample] = density tensor);
2. run RunComplianceTest.py via runpy, with datasets.load_dataset("OpenTO/OpenTO", split="test") redirected to the
   locally downloaded test parquet (the only change; the FEA and the statistics code are untouched).
"""
import argparse, glob, os, pickle, runpy, sys

import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TEST_PARQUET = os.path.expanduser(
    "~/mech_apps/hf_cache/hub/datasets--OpenTO--OpenTO/snapshots/c553a645a10feb46d4906c899fd354c1f8f2db13/data/"
    "test-00000-of-00001.parquet")
OAT_ROOT = os.path.expanduser("~/mech_apps/OAT")


def merge(res_dir, method, delta, n_problems=5000):
    merged = {}
    for f in sorted(glob.glob(os.path.join(res_dir, f"{method}_d{delta:g}_shard*.pkl"))):
        with open(f, "rb") as fh:
            merged.update(pickle.load(fh))
    missing = [i for i in range(n_problems) if i not in merged]
    assert not missing, f"{len(missing)} problems missing, e.g. {missing[:5]}"
    out = os.path.join(res_dir, f"GeneratedSamples_{method}_d{delta:g}.pkl")
    with open(out, "wb") as fh:
        pickle.dump([merged[i] for i in range(n_problems)], fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default=os.path.expanduser("~/mech_apps/results/oat"))
    ap.add_argument("--method", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--workers_per_gpu", type=int, default=4)
    ap.add_argument("--few_steps", action="store_true")
    a = ap.parse_args()
    gen_path = merge(a.res_dir, a.method, a.delta)
    out_path = gen_path.replace("GeneratedSamples_", "Compliance_" + ("few_" if a.few_steps else ""))

    _orig = datasets.load_dataset

    def _local(path, *args, split=None, **kw):
        if path == "OpenTO/OpenTO":
            assert split == "test", split
            return _orig("parquet", data_files={"test": TEST_PARQUET}, split="test")
        return _orig(path, *args, split=split, **kw)

    datasets.load_dataset = _local
    sys.argv = ["RunComplianceTest.py", "--splits", "test", "--generations_path", gen_path, "--output_path", out_path,
                "--num_workers_per_gpu", str(a.workers_per_gpu)] + (["--few_steps"] if a.few_steps else [])
    sys.path.insert(0, OAT_ROOT)
    os.chdir(OAT_ROOT)
    runpy.run_path(os.path.join(OAT_ROOT, "scripts", "RunComplianceTest.py"), run_name="__main__")


if __name__ == "__main__":
    main()
