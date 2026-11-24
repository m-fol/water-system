import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
J = lambda n: json.load(open(os.path.join(RES, n), encoding="utf-8"))
base, n10, e9, e13b, s6 = J("baselines.json"), J("neural_controls_five_seeds.json"), J("imm_true_regimes_fixed.json"), J("imm_common_test_set.json"), J("summary_controls.json")
ridge = J("neural_controls.json")["ridge"]["fixed"]["mse"]
sup = [r for r in e13b if r["name"] == "supervised"][0]["mse"]
lf = float(np.mean([r["mse"] for r in e13b if r["name"].startswith("label_free")]))
items = [
    ("Kalman\n(true matrix)", base["oracle"]["kf"], 0, "#8C8C8C"), ("RTS\n(true matrix)", base["oracle"]["rts"], 0, "#8C8C8C"),
    ("Ridge\n(64 steps)", ridge, 0, "#4C72B0"),
    ("MLP", n10["summary"]["mlp_fixed"]["mse_mean"], n10["summary"]["mlp_fixed"]["mse_sd"], "#55A868"),
    ("LSTM", n10["summary"]["lstm_fixed"]["mse_mean"], n10["summary"]["lstm_fixed"]["mse_sd"], "#55A868"),
    ("GRU", n10["summary"]["gru_fixed"]["mse_mean"], n10["summary"]["gru_fixed"]["mse_sd"], "#55A868"),
    ("Graph-\nLSTM", n10["summary"]["graph_fixed"]["mse_mean"], n10["summary"]["graph_fixed"]["mse_sd"], "#55A868"),
    ("IMM\n(true\nregimes)", e9["imm_known_start"]["mse"], 0, "#DD8452"),
    ("IMM\n(supervised\nident.)", sup, 0, "#DD8452"),
    ("IMM\n(label-free\nfull-state)", lf, 0, "#DD8452"),
    ("Original\nuntuned\nPointNet", s6["configs"]["pointnet"]["mse"]["mean"], s6["configs"]["pointnet"]["mse"]["sd"] if "sd" in s6["configs"]["pointnet"]["mse"] else 0, "#D0D0D0"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
for i, (lab, v, sd, c) in enumerate(items):
    ax.bar(i, v, color=c, edgecolor="k", linewidth=0.5, yerr=sd if sd else None, capsize=2)
    ax.text(i, v + (sd if sd else 0) + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
ax.axhline(0.123, color="k", lw=0.7, ls="--"); ax.text(len(items) - 0.5, 0.126, "time only (no sensors)", ha="right", fontsize=6.5)
ax.set_xticks(range(len(items))); ax.set_xticklabels([x[0] for x in items], fontsize=6)
ax.set_ylabel("MSE (fixed start)"); ax.set_title("Estimators on the benchmark: same tank readings, different training and model knowledge", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#8C8C8C", label="single-regime filters"), Patch(color="#4C72B0", label="ridge"), Patch(color="#55A868", label="tuned networks (5 seeds, bars: sd)"),
                   Patch(color="#DD8452", label="IMM filters"), Patch(color="#D0D0D0", label="original untuned (context)")], fontsize=6, frameon=False, loc="upper center", ncol=3)
ax.set_ylim(0, 0.27)
fig.savefig(os.path.join(FIG, "fig_baselines.pdf")); print("written fig_baselines.pdf")
