import json, os
import numpy as np
from scipy.stats import ttest_rel, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

runs = []
with open(os.path.join(RES, "results_architectures.txt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("RESULT "):
            runs.append(json.loads(line[len("RESULT "):]))
by = {}
for r in runs:
    by.setdefault(r["config"], {})[r["seed"]] = r


def per_seed(config, fn):
    return np.array([fn(by[config][s]) for s in sorted(by[config])])


def mean_over_sets(r, key, sub=None):
    vals = [(e[key] if sub is None else e[key][sub]) for e in r["eval_sets"]]
    return float(np.mean(vals))


def ms(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size)}


def paired(a, b):
    x = per_seed(a, lambda r: mean_over_sets(r, "mse"))
    y = per_seed(b, lambda r: mean_over_sets(r, "mse"))
    d = x - y
    se = d.std(ddof=1) / np.sqrt(d.size)
    h = tdist.ppf(0.975, d.size - 1) * se
    return {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)],
            "lower_seeds": int((d < 0).sum()), "n": int(d.size), "p_value": float(ttest_rel(x, y).pvalue)}


summary = {"n_runs": len(runs), "configs": {}, "paired": {}, "prob": {}}
for c in sorted(by):
    per_seed_mean = per_seed(c, lambda r: mean_over_sets(r, "mse"))
    within = per_seed(c, lambda r: float(np.std([e["mse"] for e in r["eval_sets"]], ddof=1)))
    summary["configs"][c] = {
        "params": int(by[c][min(by[c])]["params"]),
        "mse_training": ms(per_seed_mean),
        "mse_test_sampling_sd_mean": float(within.mean()),
        "tank_mse": ms(per_seed(c, lambda r: mean_over_sets(r, "mse_tank"))),
        "junction_mse": ms(per_seed(c, lambda r: mean_over_sets(r, "mse_junction"))),
    }

for a, b in [("pointnet_pm", "gnn"), ("pointnet", "gnn"), ("pointnet_id", "gnn"), ("pointnetpp", "gnn"),
             ("gnndeep", "gnn"), ("randgraph", "gnn"), ("huber", "gnn"), ("prob", "gnn"),
             ("diag_hom", "gnn"), ("diag_hom", "pointnet_pm"), ("prob", "huber"), ("pointnet_id", "pointnet"),
             ("diag_hom", "dense_het"), ("diag_het", "dense_het")]:
    summary["paired"][f"{a} - {b}"] = paired(a, b)

if "prob" in by:
    rs = [by["prob"][s] for s in sorted(by["prob"])]
    levels = ["0.5", "0.8", "0.9", "0.95", "0.99"]
    summary["prob"]["coverage_nominal"] = {l: ms([mean_over_sets(r, "coverage", l) for r in rs]) for l in levels}
    summary["prob"]["coverage95_by_state"] = {
        f"state{i}": ms([np.mean([e["coverage95_by_state"][i] for e in r["eval_sets"]]) for r in rs]) for i in range(6)}
    summary["prob"]["coverage95_event"] = ms([mean_over_sets(r, "coverage95_event") for r in rs])
    summary["prob"]["coverage95_calm"] = ms([mean_over_sets(r, "coverage95_calm") for r in rs])
    summary["prob"]["nll"] = ms([mean_over_sets(r, "nll") for r in rs])
    summary["prob"]["auc_observed_residual"] = ms([mean_over_sets(r, "auc_observed_residual") for r in rs])
    summary["prob"]["auc_oracle_residual"] = ms([mean_over_sets(r, "auc_oracle_residual") for r in rs])
    summary["prob"]["auc_predicted_sd"] = ms([mean_over_sets(r, "auc_predicted_sd") for r in rs])
    summary["prob"]["event_step_fraction"] = ms([mean_over_sets(r, "event_frac") for r in rs])

with open(os.path.join(RES, "summary_architectures.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("summary_architectures.json written:", summary["n_runs"], "runs")
