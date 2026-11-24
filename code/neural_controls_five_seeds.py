import json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as tdist
from neural_controls import run

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [("gru", 64, 3e-3), ("lstm", 128, 3e-3), ("mlp", 256, 3e-3), ("graph", (32, 64), 3e-3)]

if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(k, hp, lr, s, rnd, True) for rnd in (False, True) for (k, hp, lr) in MODELS for s in range(5)]
    with Pool(nproc) as p:
        res = p.map(run, jobs)
    summ, per = {}, {}
    for rnd in (False, True):
        mode = "random" if rnd else "fixed"
        for k, _, _ in MODELS:
            rs = sorted([r for r in res if r["kind"] == k and r["random"] == rnd], key=lambda r: r["seed"])
            per[(k, mode)] = np.array([np.mean([t["mse"] for t in r["test"]]) for r in rs])
            summ[f"{k}_{mode}"] = {"mse_mean": float(per[(k, mode)].mean()), "mse_sd": float(per[(k, mode)].std(ddof=1)), "per_seed": per[(k, mode)].tolist(),
                                   "antisym": float(np.mean([np.mean([t["mse_antisym"] for t in r["test"]]) for r in rs]))}
    paired = {}
    for mode in ("fixed", "random"):
        for other in ("gru", "lstm", "mlp"):
            d = per[("graph", mode)] - per[(other, mode)]
            se = d.std(ddof=1) / np.sqrt(d.size); h = tdist.ppf(0.975, d.size - 1) * se
            paired[f"graph_minus_{other}_{mode}"] = {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)], "seeds_graph_lower": int((d < 0).sum()), "n": int(d.size)}
    json.dump({"summary": summ, "paired": paired, "runs": res}, open(os.path.join(HERE, "..", "results", "neural_controls_five_seeds.json"), "w"), indent=1)
    print(json.dumps({"summary": {k: (round(v["mse_mean"], 4), round(v["mse_sd"], 4)) for k, v in summ.items()}, "paired": paired}, indent=1))
