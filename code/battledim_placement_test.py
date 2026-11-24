import csv, json, os, re
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BATTLEDIM_DIR", os.path.join(HERE, "..", "battledim_data"))
rng = np.random.default_rng(0)


def sections(path):
    sec, cur = {}, None
    for line in open(path, encoding="latin-1"):
        line = line.split(";")[0].rstrip()
        if not line.strip(): continue
        m = re.match(r"\s*\[(\w+)\]", line)
        if m: cur = m.group(1); sec[cur] = []; continue
        if cur: sec[cur].append(line.split())
    return sec


S = sections(os.path.join(DATA, "L-TOWN.inp"))
junc = {r[0]: float(r[2]) for r in S["JUNCTIONS"]}
fixed = set(r[0] for r in S["RESERVOIRS"]) | set(r[0] for r in S["TANKS"]) | set(r[2] for r in S["VALVES"])
free = [n for n in junc if n not in fixed]; idx = {n: i for i, n in enumerate(free)}; nF = len(free)
Lm = np.zeros((nF, nF))
for r in S["PIPES"]:
    if r[-1].lower() == "closed": continue
    a, b, L, D, Cw = r[1], r[2], float(r[3]), float(r[4]) / 1000.0, float(r[5])
    k = (Cw ** 0.54) * (D ** 2.63) / (max(L, 1.0) ** 0.54)
    for u, v in ((a, b), (b, a)):
        if u in idx:
            Lm[idx[u], idx[u]] += k
            if v in idx: Lm[idx[u], idx[v]] -= k
Linv = np.linalg.inv(Lm); d_nom = np.array([junc[n] for n in free])

with open(os.path.join(DATA, "2018_SCADA_Pressures.csv")) as fh:
    rd = csv.reader(fh, delimiter=";"); hdr = next(rd); rows = list(rd)
names = [h for h in hdr[1:]]
ts = [r[0] for r in rows]
P_raw = np.array([[float(v.replace(",", ".")) for v in r[1:]] for r in rows])
assert np.isfinite(P_raw).all() and all(n in idx for n in names)
n_t = len(P_raw)
day = np.arange(n_t) // 288
slot = np.arange(n_t) % (288 * 7)
train = day < 181; val = (day >= 181) & (day < 243); test = day >= 243
wk = np.zeros((2016, 33))
for s_ in range(2016):
    sel = train & (slot == s_); wk[s_] = P_raw[sel].mean(0)
Z = P_raw - wk[slot]
sd_z = Z[train].std(0)
noise = float(np.median(np.std(np.diff(Z[train], n=2, axis=0), axis=0) / np.sqrt(6)))
sens_idx = np.array([idx[n] for n in names])
Sd = np.diag((0.3 * d_nom + 0.02 * d_nom.mean()) ** 2)
Pfull = Linv @ Sd @ Linv.T
scale = np.median(sd_z ** 2) / np.median(np.diag(Pfull)[sens_idx])
Pm = (Pfull * scale)[np.ix_(sens_idx, sens_idx)]
print("noise sd (m)", round(noise, 4), "median sd z", round(float(np.median(sd_z)), 3), "n", n_t, flush=True)


def lagged(Zo, lags=(0, 6, 12)):
    out = []
    for l in lags:
        sh = np.roll(Zo, l, axis=0); sh[:l] = Zo[:l] if l else sh[:l]; out.append(sh)
    return np.concatenate(out, axis=1)


def fit_ridge(O, H):
    Xtr = lagged(Z[:, O]); best = None
    A = np.c_[Xtr[train], np.ones(train.sum())]; G = A.T @ A; B = A.T @ Z[train][:, H]
    Av = np.c_[Xtr[val], np.ones(val.sum())]
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B); v = float(((Av @ W - Z[val][:, H]) ** 2).mean())
        if best is None or v < best[0]: best = (v, W)
    return np.c_[Xtr, np.ones(n_t)] @ best[1]


def model_gain(O, H):
    Poo = Pm[np.ix_(O, O)] + noise ** 2 * np.eye(len(O)); Pho = Pm[np.ix_(H, O)]
    K = Pho @ np.linalg.inv(Poo); Q = Pm[np.ix_(H, H)] - K @ Pho.T
    return K, Q


def fvu(E, H):
    return float((E[test] ** 2).sum() / (Z[test][:, H] ** 2).sum())


def evaluate(O):
    O = list(O); H = [i for i in range(33) if i not in O]
    K, Q = model_gain(O, H)
    pred = float(np.trace(Q) / np.trace(Pm[np.ix_(H, H)]))
    est_m = Z[:, O] @ K.T; est_r = fit_ridge(O, H)
    return {"O": O, "pred_fvu": pred, "model_fvu": fvu(Z[:, H] - est_m, H), "ridge_fvu": fvu(Z[:, H] - est_r, H), "H": H, "K": K, "Q": Q, "est_r": est_r}


def greedy_set(m):
    O = []
    for _ in range(m):
        best = None
        for j in range(33):
            if j in O: continue
            H = [i for i in range(33) if i not in O + [j]]
            _, Q = model_gain(O + [j], H); v = float(np.trace(Q) / np.trace(Pm[np.ix_(H, H)]))
            if best is None or v < best[0]: best = (v, j)
        O.append(best[1])
    return O


res = {"noise_sd_m": noise, "median_sd_z_m": float(np.median(sd_z)), "n_test_rows": int(test.sum()), "placement": {}, "directions": {}}
for m in (2, 4, 8, 16):
    sets = [sorted(rng.choice(33, m, replace=False).tolist()) for _ in range(40)]
    ev = [evaluate(O) for O in sets]; g = evaluate(greedy_set(m))
    p = np.array([e["pred_fvu"] for e in ev]); mm = np.array([e["model_fvu"] for e in ev]); rr = np.array([e["ridge_fvu"] for e in ev])
    res["placement"][m] = {"spearman_pred_vs_model_estimator": float(spearmanr(p, mm)[0]), "spearman_pred_vs_ridge": float(spearmanr(p, rr)[0]),
                           "random_pred_mean": float(p.mean()), "random_model_fvu_mean": float(mm.mean()), "random_ridge_fvu_mean": float(rr.mean()),
                           "random_ridge_fvu_sd": float(rr.std()), "greedy_O": g["O"], "greedy_pred": g["pred_fvu"], "greedy_model_fvu": g["model_fvu"], "greedy_ridge_fvu": g["ridge_fvu"],
                           "share_random_ridge_worse_than_greedy": float((rr > g["ridge_fvu"]).mean())}
    print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["placement"][m].items()}, flush=True)
for m in (2, 4):
    out = []
    for _ in range(10):
        O = sorted(rng.choice(33, m, replace=False).tolist()); e = evaluate(O); H = e["H"]
        S0 = Pm[np.ix_(H, H)]; w, U = np.linalg.eigh(S0); S0h = U @ np.diag(w ** -0.5) @ U.T
        lam, Vv = np.linalg.eigh(S0h @ e["Q"] @ S0h); lam = np.clip(lam, 0, None); Dir = S0h @ Vv
        Zh = Z[test][:, H]; Err = Zh - e["est_r"][test]
        r = np.array([Err @ Dir[:, k] for k in range(len(H))]).var(axis=1) / np.array([Zh @ Dir[:, k] for k in range(len(H))]).var(axis=1)
        Errm = Zh - (Z[:, O] @ e["K"].T)[test]
        rm = np.array([Errm @ Dir[:, k] for k in range(len(H))]).var(axis=1) / np.array([Zh @ Dir[:, k] for k in range(len(H))]).var(axis=1)
        out.append({"O": O, "spearman_lambda_vs_measured_ratio": float(spearmanr(lam, r)[0]), "n_modes": len(H),
                    "n_lambda_gt_0.9": int((lam > 0.9).sum()), "mean_ratio_lambda_gt_0.9": float(r[lam > 0.9].mean()) if (lam > 0.9).any() else None,
                    "n_lambda_lt_0.1": int((lam < 0.1).sum()), "mean_ratio_lambda_lt_0.1": float(r[lam < 0.1].mean()) if (lam < 0.1).any() else None,
                    "spearman_lambda_vs_measured_ratio_model_estimator": float(spearmanr(lam, rm)[0]), "mean_ratio_model_estimator_lambda_gt_0.9": float(rm[lam > 0.9].mean()) if (lam > 0.9).any() else None, "mean_ratio_model_estimator_lambda_lt_0.1": float(rm[lam < 0.1].mean()) if (lam < 0.1).any() else None, "worst_mode_lambda": float(lam[-1]), "worst_mode_measured_ratio": float(r[-1]), "best_mode_lambda": float(lam[0]), "best_mode_measured_ratio": float(r[0])})
    res["directions"][m] = out
    print(m, "spearman", np.round([o["spearman_lambda_vs_measured_ratio"] for o in out], 2), "worst-mode ratio", np.round([o["worst_mode_measured_ratio"] for o in out], 2), flush=True)
json.dump(res, open(os.path.join(HERE, "..", "results", "battledim_placement_test.json"), "w"), indent=1)
print("done")
