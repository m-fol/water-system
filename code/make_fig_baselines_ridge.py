import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

with open(os.path.join(RES, "baselines.json"), encoding="utf-8") as f:
    base = json.load(f)
with open(os.path.join(RES, "summary_controls.json"), encoding="utf-8") as f:
    s6 = json.load(f)
with open(os.path.join(RES, "ridge_gramian_analysis.json"), encoding="utf-8") as f:
    a7 = json.load(f)

prob = s6["configs"]["prob"]["mse"]["mean"]
pn = s6["configs"]["pointnet"]["mse"]["mean"]
r6 = a7["ridge_window_length"]["6"]["mse_mean"]
r64 = a7["ridge_window_length"]["64"]["mse_mean"]

names = ["Kalman\n(oracle)", "RTS\n(oracle)", "Kalman\n(fitted,\nprivileged)", "RTS\n(fitted,\nprivileged)",
         "Ridge\n(6 steps)", "Ridge\n(full\nhistory)", "PointNet", "Probabilistic\nGNN"]
vals = [base["oracle"]["kf"], base["oracle"]["rts"], base["fitted"]["kf"], base["fitted"]["rts"],
        r6, r64, pn, prob]
colors = ["#8C8C8C", "#8C8C8C", "#CCB974", "#CCB974", "#4C72B0", "#4C72B0", "#55A868", "#55A868"]

fig, ax = plt.subplots(figsize=(7.0, 3.3))
ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel("MSE")
ax.set_title("Classical baselines, ridge regression and the learned observers")
fig.savefig(os.path.join(FIG, "fig_baselines.pdf"))
plt.close(fig)
print("written fig_baselines.pdf")
