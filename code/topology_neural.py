import json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy.stats import t as tdist
from topology_experiments import neural_job, ridge_exp

SETTINGS = [("ring_opposite", "linear"), ("ring_adjacent", "linear"), ("paper", "nonlinear")]
GRID = [("gru", 64, 3e-3), ("gru", 128, 1e-3), ("gru", 128, 3e-3),
        ("graph", (16, 64), 3e-3), ("graph", (32, 64), 3e-3), ("graph", (32, 128), 3e-3), ("graph", (32, 64), 1e-3)]

if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = {}
    with Pool(nproc) as p:
        jobs1 = [(k, hp, lr, 0, tp, dy) for (tp, dy) in SETTINGS for (k, hp, lr) in GRID]
        res1 = p.map(neural_job, jobs1)
        sel = {}
        for (tp, dy) in SETTINGS:
            for kind in ("gru", "graph"):
                cands = [(r, j) for r, j in zip(res1, jobs1) if j[4] == tp and j[5] == dy and j[0] == kind]
                r, j = min(cands, key=lambda c: c[0]["val_mse"]); sel[(tp, dy, kind)] = (j[1], j[2], r["val_mse"])
        jobs2 = [(kind, sel[(tp, dy, kind)][0], sel[(tp, dy, kind)][1], s, tp, dy) for (tp, dy) in SETTINGS for kind in ("gru", "graph") for s in range(5)]
        res2 = p.map(neural_job, jobs2)
    for (tp, dy) in SETTINGS:
        key = f"{tp}_{dy}"; out[key] = {"selected": {k: {"hp": sel[(tp, dy, k)][0], "lr": sel[(tp, dy, k)][1], "val_mse": sel[(tp, dy, k)][2]} for k in ("gru", "graph")}}
        per = {}
        for kind in ("gru", "graph"):
            rs = sorted([r for r, j in zip(res2, jobs2) if j[4] == tp and j[5] == dy and j[0] == kind], key=lambda r: r["seed"])
            per[kind] = np.array([r["mse"] for r in rs])
            out[key][kind] = {"mse_mean": float(per[kind].mean()), "mse_sd": float(per[kind].std(ddof=1)), "per_seed": per[kind].tolist(),
                              "mse_null_dirs": float(np.nanmean([r["mse_null_dirs"] for r in rs]))}
        d = per["graph"] - per["gru"]; h = tdist.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        out[key]["graph_minus_gru"] = {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)], "seeds_graph_lower": int((d < 0).sum())}
        out[key]["ridge"] = ridge_exp(tp, dy)
        out[key]["phase1"] = [{"kind": j[0], "hp": j[1], "lr": j[2], "val_mse": r["val_mse"]} for r, j in zip(res1, jobs1) if j[4] == tp and j[5] == dy]
        print(key, json.dumps({k: out[key][k] for k in ("gru", "graph", "graph_minus_gru", "selected")}, default=str)[:700], flush=True)
    json.dump(out, open(os.path.join("..", "results", "topology_neural.json"), "w"), indent=1, default=str)
