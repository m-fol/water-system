import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import neural_controls as N9
import imm_true_regimes as E9
import sensor_only_identification as E21
import benchmark as S
from benchmark import FLOW_GAIN, PUMP_GAIN, BASE_DEMAND, MEASUREMENT_NOISE_STD

V = N9.V
HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = {
    "nominal": {},
    "event length 3": {"event_len": 3}, "event length 12": {"event_len": 12},
    "event rate 0.04": {"p_event": 0.04}, "event rate 0.16": {"p_event": 0.16},
    "overlap, rate 0.08": {"overlap": True}, "overlap, rate 0.16": {"overlap": True, "p_event": 0.16},
    "flow gain x0.8": {"k": FLOW_GAIN * 0.8}, "flow gain x1.2": {"k": FLOW_GAIN * 1.2},
    "pump gain x0.8": {"pump_gain": PUMP_GAIN * 0.8}, "pump gain x1.2": {"pump_gain": PUMP_GAIN * 1.2},
    "base demand x1.5": {"base_demand": BASE_DEMAND * 1.5}, "demand noise x2": {"noise_level": 0.10},
    "measurement noise x2": {"meas_std": MEASUREMENT_NOISE_STD * 2},
}


def test_set(name, n=300, seed=777):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        o, m, x = S.simulate_simulator_shifts(T=64, rng=rng, **SCEN[name]); out.append((o, m, x))
    U = np.stack([np.concatenate([o[:, :2], m[:, :2].astype(np.float32)], axis=1) for o, m, x in out]).astype(np.float32)
    X = np.stack([x for o, m, x in out]).astype(np.float32)
    return out, U, X


def sc(E):
    return float((E ** 2).mean()), float(((E @ V) ** 2).mean())


def ridge_fit(n):
    Utr, Xtr = N9.gen(1000, n, False); Uv, Xv = N9.gen(123, 500, False)
    A = N9.window_design(Utr).astype(np.float64); G = A.T @ A; B = A.T @ Xtr.reshape(-1, 6).astype(np.float64)
    Av = N9.window_design(Uv); best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B); v = float(((Av @ W - Xv.reshape(-1, 6)) ** 2).mean())
        if best is None or v < best[0]:
            best = (v, lam, W)
    return best[2], best[1]


def ridge_eval(W, U, X):
    return sc(N9.window_design(U) @ W - X.reshape(-1, 6))


def model_job(args):
    name, Q, tm = args
    out, U, X = test_set(name); res = {"time_only": [0.0, 0.0], "kf": [0.0, 0.0], "imm": [0.0, 0.0]}
    M = E21.Model(E21.TRUE)
    for o, m, x in out:
        for key, est in (("time_only", np.broadcast_to(tm, x.shape)), ("kf", E9.run_kf(o, m, 0.0, Q, E9.start_prior("known", False))),
                         ("imm", E21.run_imm(M, o, m)[0])):
            a, b = sc(est - x); res[key][0] += a / len(out); res[key][1] += b / len(out)
    print(name, {k: round(v[0], 4) for k, v in res.items()}, flush=True)
    return {"scenario": name, "est": res}


def gru_job(args):
    n_train, seed, scen_names = args
    torch.set_num_threads(1); torch.manual_seed(seed); t0 = time.time()
    Utr, Xtr = N9.gen(1000 + seed, n_train, False); Uv, Xv = N9.gen(123, 500, False)
    net = N9.make("gru", 64); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N9.STEPS); rng = np.random.default_rng(seed); best, state = 1e9, None
    for step in range(N9.STEPS):
        idx = rng.integers(0, n_train, N9.BATCH)
        loss = ((net(torch.tensor(Utr[idx])) - torch.tensor(Xtr[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            v, _ = N9.mse(net, Uv, Xv)
            if v < best:
                best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state); res = {}
    for name in scen_names:
        _, U, X = test_set(name); res[name] = N9.mse(net, U, X)
    print("gru", n_train, seed, {k: round(v[0], 4) for k, v in res.items()}, round(time.time() - t0), flush=True)
    return {"n_train": n_train, "seed": seed, "val": best, "test": res}


def ident_job(args):
    n_fit, seed = args
    return E21.job(("imm", seed, n_fit))


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    names = list(SCEN)
    Q = E9.calibrate_Q(False)
    train = E9.make_set(1, 300, False); tm = np.stack([x for _, _, x, _, _, _ in train]).mean(axis=0)
    with Pool(nproc) as p:
        async_gru = [p.apply_async(gru_job, ((n, s, names if n == 8000 else ["nominal"]),)) for n in (8000, 4000, 1000, 250) for s in range(3)]
        async_id = [p.apply_async(ident_job, ((nf, s),)) for nf in (5, 15) for s in range(3)]
        async_model = [p.apply_async(model_job, ((nm, Q, tm),)) for nm in names]
        ridge = {}
        for n in (250, 1000, 4000, 8000):
            W, lam = ridge_fit(n); ridge[n] = {"lambda": lam, "test": {}}
            for nm in (names if n == 8000 else ["nominal"]):
                _, U, X = test_set(nm); ridge[n]["test"][nm] = ridge_eval(W, U, X)
            print("ridge", n, {k: round(v[0], 4) for k, v in ridge[n]["test"].items()}, flush=True)
        model = [r.get() for r in async_model]; gru = [r.get() for r in async_gru]; ident = [r.get() for r in async_id]
    json.dump({"scenarios": SCEN, "ridge": {str(k): v for k, v in ridge.items()}, "model_based": model, "gru": gru, "identification": ident,
               "identification_n60": json.load(open(os.path.join(HERE, "..", "results", "sensor_only_identification.json")))["runs"]},
              open(os.path.join(HERE, "..", "results", "robustness_shifts.json"), "w"), indent=1)
    print("done")
