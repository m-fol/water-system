import json, os
import numpy as np
from scipy.stats import ttest_rel, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

runs = []
with open(os.path.join(RES, "results_architecture_controls.txt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("RESULT "):
            runs.append(json.loads(line[len("RESULT "):]))
by = {}
for r in runs:
    by.setdefault(r["config"], {})[r["seed"]] = r


def per_seed(config, fn):
    return np.array([fn(by[config][s]) for s in sorted(by[config])])


def mean_sets(r, key):
    return float(np.mean([e[key] for e in r["eval_sets"]]))


def ms(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


def paired(a, b, key="mse"):
    x = per_seed(a, lambda r: mean_sets(r, key))
    y = per_seed(b, lambda r: mean_sets(r, key))
    d = x - y
    se = d.std(ddof=1) / np.sqrt(d.size)
    h = tdist.ppf(0.975, d.size - 1) * se
    return {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "lower_seeds": int((d < 0).sum()), "n": int(d.size), "p_value": float(ttest_rel(x, y).pvalue)}


summary = {"n_runs": len(runs), "configs": {}, "paired": {}, "events": {}, "linear_baseline": {}}
for c in sorted(by):
    summary["configs"][c] = {
        "params": int(by[c][min(by[c])]["params"]),
        "mse": ms(per_seed(c, lambda r: mean_sets(r, "mse"))),
        "mse_antisym": ms(per_seed(c, lambda r: mean_sets(r, "mse_antisym"))),
        "mse_junction": ms(per_seed(c, lambda r: mean_sets(r, "mse_junction"))),
    }

for a, b in [("gnn", "dense_het"), ("diag_hom", "diag_het"), ("prob", "prob_isolated"),
             ("prob", "wmse_twostage"), ("prob", "gnn"), ("wmse_twostage", "gnn"), ("prob_isolated", "gnn"),
             ("pointnet", "gnn")]:
    if a in by and b in by:
        summary["paired"][f"{a} - {b}"] = paired(a, b)
        summary["paired"][f"{a} - {b} [antisym]"] = paired(a, b, key="mse_antisym")

if "prob" in by:
    rs = [by["prob"][s] for s in sorted(by["prob"])]
    summary["events"]["prob"] = {}
    for key in ["auc_observed_residual", "auc_predicted_sd", "auc_oracle_residual",
                "auc_observed_dropout", "auc_observed_pump_trip", "auc_observed_demand_spike",
                "auc_sd_dropout", "auc_sd_pump_trip", "auc_sd_demand_spike"]:
        summary["events"]["prob"][key] = ms([mean_sets(r, key) for r in rs])
    summary["events"]["prob"]["coverage95"] = ms([mean_sets(r, "coverage95") for r in rs])
    summary["events"]["prob"]["coverage95_event"] = ms([mean_sets(r, "coverage95_event") for r in rs])
    summary["events"]["prob"]["coverage95_calm"] = ms([mean_sets(r, "coverage95_calm") for r in rs])

with open(os.path.join(RES, "baseline_linear.json"), encoding="utf-8") as f:
    lin = json.load(f)
ts = lin["test_sets"]
summary["linear_baseline"] = {
    "mse": ms([t["mse"] for t in ts]),
    "mse_antisym": ms([t["mse_antisym"] for t in ts]),
    "mse_junction": ms([t["mse_junction"] for t in ts]),
    "auc_innovation": ms([t["auc_innovation"] for t in ts]),
    "auc_innovation_dropout": ms([t["auc_innovation_dropout"] for t in ts]),
    "auc_innovation_pump_trip": ms([t["auc_innovation_pump_trip"] for t in ts]),
    "auc_innovation_demand_spike": ms([t["auc_innovation_demand_spike"] for t in ts]),
    "note": "single ridge fit (no training seeds); spread is across the three test sets only",
}

with open(os.path.join(RES, "summary_controls.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("summary_controls.json written:", summary["n_runs"], "runs")
