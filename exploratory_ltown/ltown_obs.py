import sys, json, csv, re
import numpy as np

data_dir, out_path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)


def sections(path):
    sec, cur = {}, None
    for line in open(path, encoding="latin-1"):
        line = line.split(";")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"\s*\[(\w+)\]", line)
        if m:
            cur = m.group(1); sec[cur] = []; continue
        if cur:
            sec[cur].append(line.split())
    return sec


S = sections(f"{data_dir}/L-TOWN.inp")
junc = {r[0]: float(r[2]) for r in S["JUNCTIONS"]}
fixed = set(r[0] for r in S["RESERVOIRS"]) | set(r[0] for r in S["TANKS"]) | set(r[2] for r in S["VALVES"])
nodes = list(junc) + [n for n in fixed if n not in junc]
free = [n for n in junc if n not in fixed]
idx = {n: i for i, n in enumerate(free)}
nF = len(free)
Lm = np.zeros((nF, nF))
n_pipes = n_closed = 0
for r in S["PIPES"]:
    if r[-1].lower() == "closed":
        n_closed += 1; continue
    a, b, L, D, Cw = r[1], r[2], float(r[3]), float(r[4]) / 1000.0, float(r[5])
    k = (Cw ** 0.54) * (D ** 2.63) / (max(L, 1.0) ** 0.54)
    n_pipes += 1
    for u, v in ((a, b), (b, a)):
        if u in idx:
            Lm[idx[u], idx[u]] += k
            if v in idx:
                Lm[idx[u], idx[v]] -= k
ev = np.linalg.eigvalsh(Lm)
assert ev[0] > 1e-12, "singular reduced Laplacian (a junction cluster is not tied to a fixed head)"

with open(f"{data_dir}/2018_SCADA_Pressures.csv") as fh:
    rd = csv.reader(fh, delimiter=";"); hdr = next(rd)
    cols = {h: [] for h in hdr[1:]}
    for row in rd:
        for h, v in zip(hdr[1:], row[1:]):
            cols[h].append(float(v.replace(",", ".")))
sensors = [s for s in hdr[1:] if s in idx]
obs_sd = np.array([np.std(cols[s]) for s in sensors])
Cmat = np.zeros((len(sensors), nF))
for i, s in enumerate(sensors):
    Cmat[i, idx[s]] = 1.0

Linv = np.linalg.inv(Lm)
d_nom = np.array([junc[n] for n in free])
out = {"assumptions": {"free_junctions": nF, "fixed_head_nodes": len(fixed), "pipes_used": n_pipes, "pipes_closed": n_closed,
                       "sensors_used": len(sensors), "conductance": "C^0.54 D^2.63 / L^0.54 (linearised Hazen-Williams, scale calibrated)",
                       "median_measured_pressure_sd_m": float(np.median(obs_sd))}, "runs": []}


def analyse(rel, noise, sens_idx):
    Sd = np.diag((rel * d_nom + 0.02 * d_nom.mean()) ** 2)
    P = Linv @ Sd @ Linv.T
    sc = (np.median(obs_sd) ** 2) / np.median(np.diag(P)[[idx[s] for s in sensors]])
    P = P * sc
    return P


def posterior(P, sens_idx, noise):
    C = np.zeros((len(sens_idx), nF)); C[np.arange(len(sens_idx)), sens_idx] = 1.0
    K = P @ C.T @ np.linalg.inv(C @ P @ C.T + noise ** 2 * np.eye(len(sens_idx)))
    return P - K @ C @ P


sens_idx = [idx[s] for s in sensors]
for rel in (0.3,):
    P = analyse(rel, 0, sens_idx)
    prior_var = np.diag(P)
    Lp = np.linalg.cholesky(P + 1e-12 * np.eye(nF))
    Li = np.linalg.inv(Lp)
    for noise in (0.1, 0.5):
        Q = posterior(P, sens_idx, noise)
        unsensed = np.array([i for i in range(nF) if i not in set(sens_idx)])
        ratio = np.diag(Q) / prior_var
        W = np.linalg.eigvalsh(Li @ Q @ Li.T)
        run = {"demand_rel_sd": rel, "sensor_noise_sd_m": noise,
               "total_variance_left_fraction_unsensed": float(np.diag(Q)[unsensed].sum() / prior_var[unsensed].sum()),
               "median_node_variance_left": float(np.median(ratio[unsensed])),
               "share_unsensed_nodes_with_over_50pct_variance_left": float((ratio[unsensed] > 0.5).mean()),
               "worst_mode_variance_left": float(W[-1]),
               "modes_with_over_50pct_left": int((W > 0.5).sum()), "modes_with_over_90pct_left": int((W > 0.9).sum()),
               "n_modes": int(nF)}
        tot = lambda Qm: float(np.diag(Qm).sum() / prior_var.sum())
        run["existing_33_total_variance_left"] = tot(Q)
        rnd = [tot(posterior(P, list(rng.choice(nF, len(sens_idx), replace=False)), noise)) for _ in range(30)]
        run["random_33_total_variance_left_mean"] = float(np.mean(rnd)); run["random_33_total_variance_left_sd"] = float(np.std(rnd))
        def greedy(start_idx, n_add):
            Qg = posterior(P, start_idx, noise) if start_idx else P.copy()
            chosen, gains = [], []
            for _ in range(n_add):
                gain = (Qg ** 2).sum(axis=0) / (np.diag(Qg) + noise ** 2)
                gain[start_idx + chosen] = -1
                j = int(np.argmax(gain)); chosen.append(j); gains.append(float(gain[j] / prior_var.sum()))
                Qg = Qg - np.outer(Qg[:, j], Qg[:, j]) / (Qg[j, j] + noise ** 2)
            return chosen, gains, float(np.diag(Qg).sum() / prior_var.sum())
        c, g, t = greedy([], len(sens_idx))
        run["greedy_33_from_scratch_total_variance_left"] = t
        c5, g5, t5 = greedy(sens_idx, 5)
        run["add_5_to_existing"] = {"nodes": [free[j] for j in c5], "fraction_of_total_variance_removed_each": g5, "total_variance_left_after": t5}
        out["runs"].append(run)
        print(json.dumps(run, indent=1), flush=True)
json.dump(out, open(out_path, "w"), indent=1)
