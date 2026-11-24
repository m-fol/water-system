import json, os, sys
import numpy as np

RULES = ("random", "symmetry", "gram_E", "gram_trace", "bayes", "bayes_weak")


def load(path):
    d = json.load(open(path))
    return d["runs"] if isinstance(d, dict) else d


def weak_ratio(run, cfg, b, model, rule):
    real = run[cfg][b]["real"]
    best = min(v[1] for v in real.values())
    return run[cfg][b][model][rule]["weak"] / best


def total_ratio(run, cfg, b, model, rule):
    real = run[cfg][b]["real"]
    best = min(v[0] for v in real.values())
    return run[cfg][b][model][rule]["total"] / best


def summarise(runs):
    n_cfg = len(runs[0])
    out = {"n_runs": len(runs), "n_configs": n_cfg, "cells": {}, "pair_choice": {}}
    for b in ("b1", "b2"):
        for model in ("nominal", "fitted"):
            for metric, fn in (("weak", weak_ratio), ("total", total_ratio)):
                vals = {r: np.array([np.mean([fn(run, c, b, model, r) for c in range(n_cfg)]) for run in runs]) for r in RULES}
                best_freq = {}
                for r in RULES:
                    wins = 0
                    for k in range(len(runs)):
                        m = min(vals[x][k] for x in RULES)
                        if vals[r][k] <= m + 1e-9:
                            wins += 1
                    best_freq[r] = wins / len(runs)
                out["cells"]["%s_%s_%s" % (b, model, metric)] = {r: {"mean": float(vals[r].mean()), "sd": float(vals[r].std(ddof=1)) if len(runs) > 1 else 0.0, "min": float(vals[r].min()), "max": float(vals[r].max()), "best_frequency": best_freq[r]} for r in RULES}
            if b == "b2":
                for r in RULES:
                    counts = [sum(1 for c in range(n_cfg) if run[c][b][model][r]["sets"] == ["3,5"]) for run in runs]
                    out["pair_choice"]["%s_%s" % (model, r)] = {"configs_choosing_J2J4_per_run": counts}
    return out


if __name__ == "__main__":
    paths = sys.argv[1:-1]
    out_path = sys.argv[-1]
    runs = [load(p) for p in paths]
    res = summarise(runs)
    json.dump(res, open(out_path, "w"), indent=1)
    for key in ("b2_nominal_weak", "b2_fitted_weak", "b1_fitted_weak", "b2_fitted_total"):
        print(key)
        for r in RULES:
            c = res["cells"][key][r]
            print("   %-11s mean %.2f  sd %.2f  range %.2f-%.2f  best in %.0f%% of runs" % (r, c["mean"], c["sd"], c["min"], c["max"], 100 * c["best_frequency"]))
    for k, v in res["pair_choice"].items():
        print(k, v["configs_choosing_J2J4_per_run"])
