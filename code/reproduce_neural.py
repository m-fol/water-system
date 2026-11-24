import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REPRO = os.path.join(RES, "repro")
os.makedirs(REPRO, exist_ok=True)

SUITES = {
    "v5": ("jobs_v5.txt", "train_v5.py", "results_v5.txt"),
    "v6": ("jobs_v6.txt", "train_v6.py", "results_v6.txt"),
}


def read_jobs(path):
    jobs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                jobs.append((parts[0], int(parts[1])))
    return jobs


def run_one(script, config, seed):
    proc = subprocess.run([sys.executable, script, config, str(seed)], cwd=HERE,
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return line
    raise RuntimeError(f"{script} {config} {seed} produced no RESULT line:\n{proc.stderr[-800:]}")


def committed(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                out[(r["config"], r["seed"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=list(SUITES), default=None)
    args = ap.parse_args()
    for name, (jobs_file, script, committed_file) in SUITES.items():
        if args.only and args.only != name:
            continue
        jobs = read_jobs(os.path.join(HERE, jobs_file))
        print(f"[{name}] {len(jobs)} jobs", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            lines = list(ex.map(lambda j: run_one(script, *j), jobs))
        out_path = os.path.join(REPRO, f"results_{name}_repro.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        ref = committed(os.path.join(RES, committed_file))
        worst = 0.0
        mismatches = []
        for line in lines:
            r = json.loads(line[7:])
            key = (r["config"], r["seed"])
            if key not in ref:
                continue
            for a, b in zip(r["eval_sets"], ref[key]["eval_sets"]):
                d = abs(a["mse"] - b["mse"])
                worst = max(worst, d)
                if d > 1e-6:
                    mismatches.append((key, a["mse"], b["mse"]))
        print(f"[{name}] written {out_path}", flush=True)
        print(f"[{name}] largest |MSE difference| vs committed: {worst:.3e}; runs off by > 1e-6: {len(mismatches)}", flush=True)
        for m in mismatches[:10]:
            print("   mismatch", m, flush=True)


if __name__ == "__main__":
    main()
