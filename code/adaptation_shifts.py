import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from multiprocessing import Pool
import neural_controls as N9
import sensor_only_identification as E
import benchmark as S
from benchmark import FLOW_GAIN, PUMP_GAIN

V = N9.V
SHIFTS = {"pump gain x0.8": {"pump_gain": PUMP_GAIN * 0.8}, "flow gain x0.8": {"k": FLOW_GAIN * 0.8}, "flow gain x1.2": {"k": FLOW_GAIN * 1.2}}


def gen(seed, n, kw):
    rng = np.random.default_rng(seed); out = [S.simulate_simulator_shifts(T=64, rng=rng, **kw) for _ in range(n)]
    U = np.stack([np.concatenate([o[:, :2], m[:, :2].astype(np.float32)], axis=1) for o, m, x in out]).astype(np.float32)
    X = np.stack([x for o, m, x in out]).astype(np.float32)
    return out, U, X


def gru_job(args):
    shift, seed = args; kw = SHIFTS[shift]; torch.set_num_threads(1); torch.manual_seed(seed); t0 = time.time()
    _, Utr, Xtr = gen(1000 + seed, 8000, kw); _, Uv, Xv = gen(123, 500, kw); _, Ut, Xt = gen(777, 300, kw)
    net = N9.make("gru", 64); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N9.STEPS); rng = np.random.default_rng(seed); best, state = 1e9, None
    for step in range(N9.STEPS):
        idx = rng.integers(0, len(Utr), N9.BATCH)
        loss = ((net(torch.tensor(Utr[idx])) - torch.tensor(Xtr[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            v, _ = N9.mse(net, Uv, Xv)
            if v < best: best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state); mse, an = N9.mse(net, Ut, Xt)
    r = {"kind": "gru", "shift": shift, "seed": seed, "mse": mse, "antisym": an, "seconds": time.time() - t0}; print(json.dumps(r), flush=True); return r


def imm_job(args):
    shift, seed = args; kw = SHIFTS[shift]; t0 = time.time()
    rng = np.random.default_rng(100 + seed); train = [S.simulate_simulator_shifts(T=64, rng=rng, **kw)[:2] for _ in range(30)]
    th, f, nfev = E.fit("imm", train, E.TRUE.copy())
    tests, _, _ = gen(777, 300, kw); mse, an = E.evaluate("imm", th, tests)
    r = {"kind": "imm_id", "shift": shift, "seed": seed, "theta": th.tolist(), "rel_err_to_nominal": (th / E.TRUE - 1).tolist(), "mse": mse, "antisym": an, "nfev": nfev, "seconds": time.time() - t0}
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for k, v in r.items()}), flush=True); return r


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    with Pool(nproc) as p:
        a_gru = [p.apply_async(gru_job, ((sh, s),)) for sh in SHIFTS for s in range(3)]
        a_imm = [p.apply_async(imm_job, ((sh, s),)) for sh in SHIFTS for s in range(3)]
        ridge = {}
        for sh, kw in SHIFTS.items():
            _, Utr, Xtr = gen(1000, 8000, kw); _, Uv, Xv = gen(123, 500, kw); _, Ut, Xt = gen(777, 300, kw)
            A = N9.window_design(Utr).astype(np.float64); G = A.T @ A; B = A.T @ Xtr.reshape(-1, 6).astype(np.float64); Av = N9.window_design(Uv); best = None
            for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                W = np.linalg.solve(G + lam * np.eye(G.shape[0]), B); v = float(((Av @ W - Xv.reshape(-1, 6)) ** 2).mean())
                if best is None or v < best[0]: best = (v, lam, W)
            Er = N9.window_design(Ut) @ best[2] - Xt.reshape(-1, 6); ridge[sh] = {"mse": float((Er ** 2).mean()), "antisym": float(((Er @ V) ** 2).mean()), "lambda": best[1]}
            print("ridge", sh, ridge[sh], flush=True)
        gru = [r.get() for r in a_gru]; imm = [r.get() for r in a_imm]
    json.dump({"ridge": ridge, "gru": gru, "imm_id": imm}, open(os.path.join("..", "results", "adaptation_shifts.json"), "w"), indent=1)
    print("done")
