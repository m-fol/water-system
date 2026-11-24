import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

a = json.load(open(os.path.join(RES, "weak_mode_recoverability.json")))
sp = a["sensor_placement"]
prior = a["prior_only_J2_minus_J4"]["mse_const_mean_avg"]
names = ["none", "J1", "J2", "J3", "J4"]
x = np.arange(len(names)); w = 0.38

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax[0].bar(x - w / 2, [sp[n]["kf_mse"] for n in names], w, label="Kalman filter (true matrices)", color="#4C72B0")
ax[0].bar(x + w / 2, [sp[n]["ridge64_mse"] for n in names], w, label="Ridge, full history", color="#DD8452")
ax[0].set_xticks(x); ax[0].set_xticklabels(["tanks only", "+J1", "+J2", "+J3", "+J4"])
ax[0].set_ylabel("MSE, all states"); ax[0].set_title("Total error", fontsize=8); ax[0].legend(frameon=False, fontsize=7)
ax[1].bar(x - w / 2, [sp[n]["kf_mse_antisym"] for n in names], w, color="#4C72B0")
ax[1].bar(x + w / 2, [sp[n]["ridge64_mse_antisym"] for n in names], w, color="#DD8452")
ax[1].axhline(prior, color="k", lw=0.8, ls="--"); ax[1].text(4.45, prior + 0.002, "prior only", ha="right", fontsize=7)
ax[1].set_xticks(x); ax[1].set_xticklabels(["tanks only", "+J1", "+J2", "+J3", "+J4"])
ax[1].set_ylabel(r"MSE, $(e\cdot v)^2$"); ax[1].set_title("J2 minus J4 direction", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_sensor.pdf")); plt.close()

c = json.load(open(os.path.join(RES, "calibration_by_event.json")))["runs"]
kinds = ["calm", "dropout", "pump_trip", "demand_spike"]
lab = ["calm", "dropout", "pump trip", "demand spike"]
cov = np.array([[r["by_event"][k]["coverage95"] for k in kinds] for r in c])
wid = np.array([[r["by_event"][k]["mean_width"] for k in kinds] for r in c])
rm = np.array([[r["by_event"][k]["rmse"] for k in kinds] for r in c])
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
ax[0].bar(range(4), cov.mean(0), yerr=cov.std(0, ddof=1), color="#55A868", capsize=2)
ax[0].axhline(0.95, color="k", lw=0.8, ls="--"); ax[0].set_ylim(0.7, 1.0)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab); ax[0].set_ylabel("95% interval coverage")
ax[1].bar(np.arange(4) - 0.2, wid.mean(0) / (2 * 1.96), 0.4, label="mean predicted sd", color="#8172B2")
ax[1].bar(np.arange(4) + 0.2, rm.mean(0), 0.4, label="realised RMSE", color="#C44E52")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_calib_events.pdf")); plt.close()
print("figures written")
