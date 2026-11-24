import json, os
import numpy as np
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

rows = []
with open(os.path.join(RES, "results_v4.txt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("RESULT "):
            rows.append(json.loads(line[len("RESULT "):]))

by = {}
for r in rows:
    by.setdefault(r["config"], {})[r["seed"]] = r


def arr(config, key, sub=None):
    seeds = sorted(by[config])
    if sub is None:
        return np.array([by[config][s][key] for s in seeds])
    return np.array([by[config][s][key][sub] for s in seeds])


def ms(a):
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


summary = {"n_runs": len(rows), "configs": {}, "paired": {}, "calibration": {}, "events": {}}
for c in by:
    a = arr(c, "mse")
    summary["configs"][c] = {
        "params": int(by[c][min(by[c])]["params"]),
        "mse": ms(a),
        "mse_tank": ms(arr(c, "mse_tank")),
        "mse_junction": ms(arr(c, "mse_junction")),
    }

pairs = [("pointnet_pm", "gnn"), ("pointnet", "gnn"), ("pointnetpp", "gnn"), ("gnndeep", "gnn"),
         ("randgraph", "gnn"), ("diag_hom", "gnn"), ("prob", "gnn"), ("diag_hom", "dense_het"),
         ("diag_het", "dense_het")]
for a_, b_ in pairs:
    x, y = arr(a_, "mse"), arr(b_, "mse")
    d = x - y
    t = ttest_rel(x, y)
    summary["paired"][f"{a_} - {b_}"] = {
        "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
        "negative_seeds": int((d < 0).sum()), "n": int(d.size), "p_value": float(t.pvalue),
    }

levels = ["0.5", "0.8", "0.9", "0.95", "0.99"]
summary["calibration"] = {l: ms(np.array([by["prob"][s]["coverage"][l] for s in sorted(by["prob"])])) for l in levels}
summary["calibration"]["tank_95"] = ms(arr("prob", "coverage95_tank"))
summary["calibration"]["junction_95"] = ms(arr("prob", "coverage95_junction"))

summary["events"] = {
    "auc_absolute_residual": ms(arr("prob", "auc_absresid")),
    "auc_predicted_sd": ms(arr("prob", "auc_sd")),
    "event_step_fraction": ms(arr("prob", "event_frac")),
}

with open(os.path.join(RES, "summary_v4.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("summary written:", summary["n_runs"], "runs")
