import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import hyd_v23 as H
import est_v9 as E9
import est_v10 as E10
import est_v12 as E12
import neural_v9 as N9
from sim_v9 import simulate as _unused

V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
HERE = os.path.dirname(os.path.abspath(__file__))
K_LF = 9


_CWD_SET = []


def one(args):
    seed, = args
    if not _CWD_SET:
        import tempfile
        d = os.path.join(tempfile.gettempdir(), "epanet_worker_%d" % os.getpid()); os.makedirs(d, exist_ok=True); os.chdir(d); _CWD_SET.append(d)
    return H.simulate(np.random.default_rng(seed), simulator="epanet")


def gen(seed, n, pool):
    rng = np.random.default_rng(seed); seeds = rng.integers(0, 2 ** 31 - 1, n)
    out = pool.map(one, [(int(s),) for s in seeds], chunksize=20)
    o, m, x, r = zip(*out)
    return np.stack(o), np.stack(m), np.stack(x), np.stack(r)


def U_of(o, m):
    return np.concatenate([o[:, :, :2], m[:, :, :2].astype(np.float32)], axis=2).astype(np.float32)


def feats(x, regime=None):
    n, T, d = x.shape; rows = []; tgt = []; lab = []; tid = []
    for i in range(n):
        for t in range(1, T):
            ang = 2 * np.pi * t / 48
            rows.append(np.r_[x[i, t - 1], 1.0, np.sin(ang), np.cos(ang)]); tgt.append(x[i, t]); tid.append(i)
            if regime is not None: lab.append(regime[i, t])
    return np.array(rows), np.array(tgt), np.array(lab), np.array(tid)


def models_from(params):
    n = 6; models = []
    for B, Q in params:
        c0, c1, c2 = B[n], B[n + 1], B[n + 2]
        models.append((B[:n].T, (lambda t, c0=c0, c1=c1, c2=c2: c0 + c1 * np.sin(2 * np.pi * (t + 1) / 48) + c2 * np.cos(2 * np.pi * (t + 1) / 48)), Q))
    return models


def sc(E):
    return float((E ** 2).mean()), float(((E @ V) ** 2).mean())


def eval_filter(models, PI, mu0, prior, test):
    o, m, x = test; se = an = 0.0
    for i in range(len(o)):
        est = E10.imm(o[i, 1:], m[i, 1:], models, PI, mu0, prior, H.MEAS_STD ** 2)
        a, b = sc(est - x[i, 1:]); se += a; an += b
    return se / len(o), an / len(o)


def prior_of(xtr):
    x0 = xtr[:, 0].mean(0); return x0, np.cov(xtr[:, 0].T) + 1e-4 * np.eye(6)


def job(args):
    kind, seed, train, test = args
    t0 = time.time(); otr, mtr, xtr, rtr = train
    prior = prior_of(xtr)
    if kind == "kf":
        X, Y, _, _ = feats(xtr); B, *_ = np.linalg.lstsq(X, Y, rcond=None); Q = np.cov((Y - X @ B).T) + 1e-6 * np.eye(6)
        res = eval_filter(models_from([(B, Q)]), np.ones((1, 1)), np.ones(1), prior, test)
    elif kind == "imm_sup":
        X, Y, lab, tid = feats(xtr, rtr); params = []
        for r in range(7):
            idx = lab == r; B, *_ = np.linalg.lstsq(X[idx], Y[idx], rcond=None); params.append((B, np.cov((Y[idx] - X[idx] @ B).T) + 1e-6 * np.eye(6)))
        trans = np.zeros((7, 7))
        for i in range(1, len(lab)):
            if tid[i] == tid[i - 1]: trans[lab[i - 1], lab[i]] += 1
        PI = (trans + 1e-3) / (trans + 1e-3).sum(1, keepdims=True); mu0 = np.bincount(lab, minlength=7) / len(lab)
        res = eval_filter(models_from(params), PI, mu0, prior, test)
    elif kind == "imm_lf":
        X, Y, lab, tid = feats(xtr, rtr); B0, *_ = np.linalg.lstsq(X, Y, rcond=None); R0 = Y - X @ B0; Z = R0 / R0.std(0); best = None
        for r in range(3):
            z0 = E12.kmeans(Z, K_LF, np.random.default_rng(3 * seed + r)); params, z, ll = E12.hard_em(X, Y, z0, K_LF)
            if best is None or ll > best[0]: best = (ll, params, z)
        ll, params, z = best; trans = np.zeros((K_LF, K_LF))
        for i in range(1, len(z)):
            if tid[i] == tid[i - 1]: trans[z[i - 1], z[i]] += 1
        PI = (trans + 1e-3) / (trans + 1e-3).sum(1, keepdims=True); mu0 = np.bincount(z, minlength=K_LF) / len(z)
        res = eval_filter(models_from(params), PI, mu0 / mu0.sum(), prior, test)
        conf = np.zeros((K_LF, 7), int)
        for g, l in zip(z, lab): conf[g, l] += 1
        out = {"kind": kind, "seed": seed, "mse": res[0], "antisym": res[1], "train_purity": float(conf.max(1).sum() / conf.sum()),
               "regimes_represented": int(len(set(conf.argmax(1)))), "seconds": time.time() - t0}
        print(json.dumps(out), flush=True); return out
    elif kind == "gru":
        torch.set_num_threads(1); torch.manual_seed(seed)
        otr_, mtr_, xtr_, _ = train; Utr = U_of(otr_, mtr_); Uv, Xv = job.val
        sh = 7.0
        net = N9.make("gru", 64); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N9.STEPS); rng = np.random.default_rng(seed); best, state = 1e9, None
        def prep(U): U = U.copy(); U[:, :, :2] = (U[:, :, :2] - sh) * U[:, :, 2:]; return torch.tensor(U)
        Ptr, Pv = prep(Utr), prep(Uv); Ttr = torch.tensor((xtr_ - sh).astype(np.float32)); Tv = torch.tensor((Xv - sh).astype(np.float32))
        for step in range(N9.STEPS):
            idx = torch.tensor(rng.integers(0, len(Ptr), N9.BATCH))
            loss = ((net(Ptr[idx]) - Ttr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            if (step + 1) % 250 == 0:
                with torch.no_grad(): v = float(((net(Pv) - Tv) ** 2).mean())
                if v < best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(state); o, m, x = test
        with torch.no_grad(): est = net(prep(U_of(o, m))).numpy() + sh
        res = sc((est - x)[:, 1:])
    out = {"kind": kind, "seed": seed, "mse": res[0], "antisym": res[1], "seconds": time.time() - t0}
    print(json.dumps(out), flush=True); return out


def init_gru(val):
    job.val = val


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    t0 = time.time()
    with Pool(nproc) as p:
        train = gen(1000, 2000, p); val = gen(123, 300, p)
        te = [gen(s, 300, p) for s in (777, 888, 999)]
    test = tuple(np.concatenate([t[i] for t in te]) for i in range(3)); testfull = test
    print("data generated", round(time.time() - t0), "s", flush=True)
    xtr = train[2]; x0 = xtr[:, 0]
    info = {"state_mean": xtr.mean((0, 1)).tolist(), "state_std": xtr.std((0, 1)).tolist(), "antisym_std_train": float((xtr @ V).std()),
            "start_state_spread": float(x0.std(0).max())}
    tm = xtr.mean(0); te_o, te_m, te_x = test
    res = {"time_only": sc((np.broadcast_to(tm, te_x.shape) - te_x)[:, 1:])}
    Utr, Uv = U_of(train[0], train[1]), U_of(val[0], val[1]); A = N9.window_design(Utr).astype(np.float64)
    G = A.T @ A; Bm = A.T @ xtr.reshape(-1, 6).astype(np.float64); Av = N9.window_design(Uv); best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Bm); v = float(((Av @ W - val[2].reshape(-1, 6)) ** 2).mean())
        if best is None or v < best[0]: best = (v, lam, W)
    est = (N9.window_design(U_of(te_o, te_m)) @ best[2]).reshape(te_x.shape); res["ridge"] = sc((est - te_x)[:, 1:])
    print("time_only", res["time_only"], "ridge", res["ridge"], flush=True)
    specs = [("gru", s, train, test) for s in range(3)] + [("kf", 0, train, test), ("imm_sup", 0, train, test)] + [("imm_lf", s, train, test) for s in range(3)]
    with Pool(len(specs), initializer=init_gru, initargs=((Uv, val[2]),)) as p:
        runs = p.map(job, specs, chunksize=1)
    json.dump({"info": info, "time_only": res["time_only"], "ridge": res["ridge"], "ridge_lambda": best[1], "runs": runs, "n_train": 2000, "n_test": 900},
              open(os.path.join(HERE, "..", "results", "hyd_exp_v23.json"), "w"), indent=1)
    print("done", round(time.time() - t0), "s")
