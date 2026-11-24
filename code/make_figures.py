import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})

runs = []
with open(os.path.join(RES, "results_v5.txt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("RESULT "):
            runs.append(json.loads(line[len("RESULT "):]))
by = {}
for r in runs:
    by.setdefault(r["config"], {})[r["seed"]] = r
with open(os.path.join(RES, "summary_v5.json"), encoding="utf-8") as f:
    summary = json.load(f)
with open(os.path.join(RES, "baselines.json"), encoding="utf-8") as f:
    baselines = json.load(f)
with open(os.path.join(RES, "smoother.json"), encoding="utf-8") as f:
    smooth = json.load(f)


def per_seed(config, fn):
    return np.array([fn(by[config][s]) for s in sorted(by[config])])


def seed_mse(config):
    return per_seed(config, lambda r: float(np.mean([e["mse"] for e in r["eval_sets"]])))


def bars(ax, labels, arrays, colors):
    x = np.arange(len(labels))
    ax.bar(x, [a.mean() for a in arrays], yerr=[a.std(ddof=1) for a in arrays], capsize=3,
           color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    for i, a in enumerate(arrays):
        ax.scatter(np.full(len(a), i) + np.linspace(-0.15, 0.15, len(a)), a, s=10, c="k", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")


archs = [("gnn", "Two-layer\nGNN"), ("gnndeep", "Four-layer\nGNN"), ("randgraph", "Random-\ngraph GNN"),
         ("pointnet", "PointNet"), ("pointnet_id", "PointNet\n+ID"), ("pointnet_pm", "PointNet\nmatched"),
         ("pointnetpp", "PointNet++"), ("huber", "Huber-loss\nGNN")]
fig, ax = plt.subplots(figsize=(6.6, 2.5))
arrs = [seed_mse(a) for a, _ in archs]
colors = ["#4C72B0", "#4C72B0", "#8172B2", "#55A868", "#55A868", "#55A868", "#55A868", "#DD8452"]
y = np.arange(len(archs))
ax.barh(y, [a.mean() for a in arrs], xerr=[a.std(ddof=1) for a in arrs], capsize=3,
        color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
for i, a in enumerate(arrs):
    ax.scatter(a, np.full(len(a), i) + np.linspace(-0.25, 0.25, len(a)), s=8, c="k", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([l.replace("\n", " ") for _, l in archs])
ax.invert_yaxis()
ax.set_xlabel("MSE (held-out trajectories)")
ax.set_title("Estimation error by encoder and control (ten seeds; bars mean ± sd, dots seeds)")
fig.savefig(os.path.join(FIG, "fig_architectures.pdf"))
plt.close(fig)

pairs = [("PointNet", "pointnet"), ("PointNet + node ID", "pointnet_id"), ("PointNet (param.-matched)", "pointnet_pm"),
         ("PointNet++", "pointnetpp"), ("Four-layer GNN", "gnndeep"), ("Random-graph GNN", "randgraph"),
         ("Huber-loss GNN", "huber"), ("Diagonal-gain GNN", "diag_hom"), ("Probabilistic GNN", "prob")]
gnn = seed_mse("gnn")
fig, ax = plt.subplots(figsize=(6.6, 2.7))
for i, (lab, c) in enumerate(pairs):
    d = seed_mse(c) - gnn
    ax.scatter(d, np.full(len(d), i), s=14, c="k", zorder=3)
    key = f"{c} - gnn"
    ci = summary["paired"][key]["ci95"]
    ax.plot(ci, [i, i], c="#C44E52", lw=1.5)
    ax.plot([d.mean(), d.mean()], [i - 0.2, i + 0.2], c="#C44E52", lw=2)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels([l for l, _ in pairs])
ax.set_xlabel("MSE minus two-layer GNN MSE (same seed)")
ax.set_title("Paired per-seed differences (dots: seeds; red bar: mean; line: 95% CI)")
fig.savefig(os.path.join(FIG, "fig_paired.pdf"))
plt.close(fig)

cfgs = [("Dense W\nhomogeneous", "gnn"), ("Dense W\nheterogeneous", "dense_het"),
        ("Diagonal W\nhomogeneous", "diag_hom"), ("Diagonal W\nheterogeneous", "diag_het"),
        ("Probabilistic\n(Gaussian NLL)", "prob")]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
bars(ax, [l for l, _ in cfgs], [seed_mse(c) for _, c in cfgs], ["#4C72B0", "#4C72B0", "#C44E52", "#C44E52", "#55A868"])
ax.set_ylabel("MSE")
ax.set_title("Dense vs diagonal-gain GNN, homogeneous vs heterogeneous pipes")
fig.savefig(os.path.join(FIG, "fig_dense_diag_het.pdf"))
plt.close(fig)

p = summary["prob"]
lv = [0.5, 0.8, 0.9, 0.95, 0.99]
emp = np.array([p["coverage_nominal"][str(l)]["mean"] for l in lv])
sdv = np.array([p["coverage_nominal"][str(l)]["sd"] for l in lv])
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.2))
a1, a2, a3 = axes
a1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
a1.errorbar(lv, emp, yerr=sdv, fmt="o-", color="#55A868", capsize=3, label="probabilistic GNN")
a1.set_xlabel("nominal central coverage")
a1.set_ylabel("empirical coverage")
a1.set_xlim(0.45, 1.0)
a1.set_ylim(0.45, 1.0)
a1.legend(frameon=False, loc="lower right", fontsize=7)
a1.set_title("Calibration curve")
st = [p["coverage95_by_state"][f"state{i}"] for i in range(6)]
a2.bar(range(6), [v["mean"] for v in st], yerr=[v["sd"] for v in st], capsize=3,
       color="#55A868", edgecolor="k", linewidth=0.5)
a2.axhline(0.95, color="k", ls="--", lw=1)
a2.set_xticks(range(6))
a2.set_xticklabels(["$\\ell_{T1}$", "$\\ell_{T2}$", "$h_{J1}$", "$h_{J2}$", "$h_{J3}$", "$h_{J4}$"], fontsize=7)
a2.set_ylim(0.85, 1.0)
a2.set_ylabel("95% coverage")
a2.set_title("By state")
ev = [p["coverage95_calm"], p["coverage95_event"]]
a3.bar(range(2), [v["mean"] for v in ev], yerr=[v["sd"] for v in ev], capsize=3,
       color=["#8C8C8C", "#C44E52"], edgecolor="k", linewidth=0.5)
a3.axhline(0.95, color="k", ls="--", lw=1)
a3.set_xticks(range(2))
a3.set_xticklabels(["calm steps", "event-active steps"])
a3.set_ylim(0.85, 1.0)
a3.set_ylabel("95% coverage")
a3.set_title("By regime")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_calibration.pdf"))
plt.close(fig)

ev_arrs = [per_seed("prob", lambda r: float(np.mean([e["auc_observed_residual"] for e in r["eval_sets"]]))),
           per_seed("prob", lambda r: float(np.mean([e["auc_predicted_sd"] for e in r["eval_sets"]]))),
           per_seed("prob", lambda r: float(np.mean([e["auc_oracle_residual"] for e in r["eval_sets"]])))]
fig, ax = plt.subplots(figsize=(6.0, 3.1))
bars(ax, ["Observed-sensor\nresidual (deployable)", "Predicted sd\n(deployable)", "True-state residual\n(oracle only)"],
     ev_arrs, ["#55A868", "#8C8C8C", "#CCB974"])
ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("ROC AUC for event-active steps")
ax.legend(frameon=False, loc="upper right")
ax.set_title("Detecting disturbance events from the observer output")
fig.savefig(os.path.join(FIG, "fig_events.pdf"))
plt.close(fig)


regs = smooth["regimes"]
x = np.arange(len(regs))
w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.bar(x - w / 2, [r["kf"] for r in regs], w, label="Kalman filter", color="#4C72B0", edgecolor="k", linewidth=0.5)
ax.bar(x + w / 2, [r["rts"] for r in regs], w, label="RTS smoother", color="#DD8452", edgecolor="k", linewidth=0.5)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in regs], rotation=10, ha="right")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Filter versus smoother by regime")
ax.legend(frameon=False)
fig.savefig(os.path.join(FIG, "fig_smoother.pdf"))
plt.close(fig)

edges = [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]
N = np.zeros((6, 6))
for j, (a, b) in enumerate(edges):
    N[a, j] = 1
    N[b, j] = -1
LG = N @ N.T
Ep = np.diag([1, 1, 0, 0, 0, 0.0])
dt, kp = 0.5, 0.6
ks = np.linspace(0.05, 1.2, 200)
norm_, mu = [], []
for k in ks:
    M = np.eye(6) - dt * (k * LG + kp * Ep)
    norm_.append(np.linalg.norm(M, 2))
    mu.append(dt * np.linalg.eigvalsh(k * LG + kp * Ep)[-1])
fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.plot(ks, norm_, label=r"$\|A\|_2$", color="#4C72B0")
ax.plot(ks, mu, label=r"$\Delta t\,\mu_{\max}$", color="#C44E52")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.axhline(2.0, color="k", ls="--", lw=1)
ax.axvline(0.35, color="#55A868", lw=1.2, label="simulator $k=0.35$")
ax.axvline(0.8384, color="#8172B2", lw=1.2, label=r"stability limit $k^\ast=0.838$")
ax.set_xlabel("pipe gain $k$")
ax.set_ylabel("value")
ax.set_ylim(0, 3)
ax.legend(frameon=False, fontsize=7)
ax.set_title("Gain sensitivity of the explicit update (unit tank areas)")
fig.savefig(os.path.join(FIG, "fig_gain.pdf"))
plt.close(fig)

print("figures written:", sorted(f for f in os.listdir(FIG) if f.endswith(".pdf")))
