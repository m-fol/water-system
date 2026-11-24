import json, os, sys
import numpy as np
import torch
from multiprocessing import Pool

import train_v6 as T6
from data import generate_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def one(seed):
    torch.set_num_threads(1)
    net = T6.train("prob", seed)
    rows = {k: {"n": 0, "cover": 0, "width": 0.0, "se": 0.0} for k in ("calm", "dropout", "pump_trip", "demand_spike")}
    anti = {"n": 0, "cover": 0, "sd": 0.0, "se": 0.0}
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        O, X, K = [], [], []
        for _ in range(1000):
            o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
            O.append(o); X.append(x)
            k = np.full(64, "calm", dtype=object)
            for (t0, t1, kind, node) in ev:
                k[t0:t1] = kind
            K.append(k)
        O = torch.tensor(np.stack(O), dtype=torch.float32); X = np.stack(X); K = np.stack(K)
        with torch.no_grad():
            mu, logsd = net(O)
        mu = mu.numpy(); sd = logsd.exp().numpy()
        E = mu - X
        z = np.abs(E) / sd
        for kind in rows:
            sel = K == kind
            rows[kind]["n"] += int(sel.sum() * 6)
            rows[kind]["cover"] += int((z[sel] <= 1.96).sum())
            rows[kind]["width"] += float((2 * 1.96 * sd[sel]).sum())
            rows[kind]["se"] += float((E[sel] ** 2).sum())
        ea = E @ V
        sda = np.sqrt((sd ** 2) @ (V ** 2))
        anti["n"] += ea.size
        anti["cover"] += int((np.abs(ea) <= 1.96 * sda).sum())
        anti["sd"] += float(sda.sum()); anti["se"] += float((ea ** 2).sum())
    res = {"seed": seed, "by_event": {k: {"coverage95": v["cover"] / v["n"], "mean_width": v["width"] / v["n"],
                                           "rmse": float(np.sqrt(v["se"] / v["n"])), "n_values": v["n"]}
                                       for k, v in rows.items()},
           "antisym": {"coverage95": anti["cover"] / anti["n"], "mean_pred_sd": anti["sd"] / anti["n"],
                       "realised_rmse": float(np.sqrt(anti["se"] / anti["n"]))}}
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with Pool(nproc) as p:
        runs = p.map(one, range(5))
    agg = {}
    for kind in ("calm", "dropout", "pump_trip", "demand_spike"):
        agg[kind] = {m: float(np.mean([r["by_event"][kind][m] for r in runs])) for m in ("coverage95", "mean_width", "rmse")}
        agg[kind]["coverage_sd_over_seeds"] = float(np.std([r["by_event"][kind]["coverage95"] for r in runs], ddof=1))
    agg["antisym"] = {m: float(np.mean([r["antisym"][m] for r in runs])) for m in ("coverage95", "mean_pred_sd", "realised_rmse")}
    with open(os.path.join(RES, "calib_v8.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "aggregate": agg}, f, indent=2)
    print("AGG", json.dumps(agg, indent=1))
