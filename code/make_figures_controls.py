import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

with open(os.path.join(RES, "summary_controls.json"), encoding="utf-8") as f:
    s6 = json.load(f)
runs = []
with open(os.path.join(RES, "results_architecture_controls.txt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("RESULT "):
            runs.append(json.loads(line[len("RESULT "):]))
by = {}
for r in runs:
    by.setdefault(r["config"], {})[r["seed"]] = r
with open(os.path.join(RES, "baseline_linear.json"), encoding="utf-8") as f:
    lin = json.load(f)


def per_seed(config, key):
    return np.array([np.mean([e[key] for e in by[config][s]["eval_sets"]]) for s in sorted(by[config])])


def bars(ax, labels, arrays, colors):
    x = np.arange(len(labels))
    ax.bar(x, [a.mean() for a in arrays], yerr=[a.std(ddof=1) for a in arrays], capsize=3,
           color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    for i, a in enumerate(arrays):
        ax.scatter(np.full(len(a), i) + np.linspace(-0.15, 0.15, len(a)), a, s=10, c="k", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")


labels = ["Plain GNN", "Weighted MSE\n(isolated variance)", "Probabilistic\n(Gaussian NLL)", "Weighted MSE\n(two-stage)"]
keys = ["gnn", "prob_isolated", "prob", "wmse_twostage"]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
bars(ax, labels, [per_seed(k, "mse") for k in keys], ["#8C8C8C", "#CCB974", "#55A868", "#C44E52"])
ax.set_ylabel("MSE (mean over three test sets)")
ax.set_title("Controls for the probabilistic gain (ten seeds)")
fig.savefig(os.path.join(FIG, "fig_controls.pdf"))
plt.close(fig)

labels = ["Dense GNN\n(hom.)", "Dense GNN\n(het.)", "Diagonal\n(hom.)", "Diagonal\n(het.)", "PointNet", "Probabilistic\nGNN", "Ridge\n(6-step window)"]
keys = ["gnn", "dense_het", "diag_hom", "diag_het", "pointnet", "prob"]
arrs = [per_seed(k, "mse_antisym") for k in keys] + [np.array([t["mse_antisym"] for t in lin["test_sets"]])]
fig, ax = plt.subplots(figsize=(6.6, 2.4))
colors = ["#4C72B0", "#4C72B0", "#C44E52", "#C44E52", "#55A868", "#55A868", "#8C8C8C"]
y = np.arange(len(labels))
ax.barh(y, [a.mean() for a in arrs], xerr=[a.std(ddof=1) for a in arrs], capsize=3,
        color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
for i, a in enumerate(arrs):
    ax.scatter(a, np.full(len(a), i) + np.linspace(-0.25, 0.25, len(a)), s=8, c="k", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([l.replace("\n", " ") for l in labels])
ax.invert_yaxis()
ax.set_xlabel(r"error in the J2$-$J4 mode, $(e\cdot v)^2$")
ax.set_title("Error in the direction the tank sensors cannot see")
fig.savefig(os.path.join(FIG, "fig_antisym.pdf"))
plt.close(fig)

p = s6["events"]["prob"]
labels = ["Pump trip\n(observer residual)", "Demand spike\n(observer residual)",
          "Pump trip\n(linear innovation)", "Demand spike\n(linear innovation)"]
vals = [p["auc_observed_pump_trip"], p["auc_observed_demand_spike"],
        s6["linear_baseline"]["auc_innovation_pump_trip"], s6["linear_baseline"]["auc_innovation_demand_spike"]]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.bar(range(4), [v["mean"] for v in vals], yerr=[v["sd"] for v in vals], capsize=3,
       color=["#55A868", "#55A868", "#8C8C8C", "#8C8C8C"], edgecolor="k", linewidth=0.5)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("ROC AUC for event-active steps")
ax.set_title("Detection by event type")
fig.savefig(os.path.join(FIG, "fig_events_types.pdf"))
plt.close(fig)
print("figures written: fig_controls, fig_antisym, fig_events_types")
