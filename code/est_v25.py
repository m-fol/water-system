import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import est_v21 as E
from data import TANK_SETPOINT, TANK_IDX, JUNCTION_IDX
from network import STATE_DIM
from sim_v9 import diurnal

DT = 0.5


class Struct:
    def __init__(self, name):
        self.name = name
        mult = 2.0 if name == "spike_x2" else 4.0
        self.mult = mult
        if name == "calm_only":
            self.regimes = [(None, None)]
        elif name == "no_spike":
            self.regimes = [(None, None), (0, None), (1, None)]
        else:
            self.regimes = [(None, None)] + [(k, None) for k in range(2)] + [(None, j) for j in range(4)]
        n = len(self.regimes)
        scale = 4.0 if name == "wrong_rates" else 1.0
        exit_p = 1.0 / 3.0 if name == "wrong_rates" else 1.0 / 6.0
        p_trip, p_spike = 0.08 / 3 / 2 * scale, 0.08 / 3 / 4 * scale
        PI = np.zeros((n, n)); PI[0, 0] = 1.0
        if n > 1:
            PI[0, 1:3] = p_trip
            if n > 3: PI[0, 3:] = p_spike
            PI[0, 0] = 1 - PI[0, 1:].sum()
            for r in range(1, n): PI[r, 0] = exit_p; PI[r, r] = 1 - exit_p
        self.PI = PI
        mu = PI[0].copy() if n > 1 else np.ones(1); self.mu0 = mu / mu.sum()


class Model:
    def __init__(self, theta, st):
        k, pg, bd, sd, sv = [float(v) for v in theta]
        self.R = sv ** 2; self.pg, self.bd, self.st = pg, bd, st; self.A, self.Q = [], []
        for (trip, spike) in st.regimes:
            damp = np.zeros((STATE_DIM, STATE_DIM))
            for kk, idx in enumerate(TANK_IDX):
                if trip != kk: damp[idx, idx] = pg
            self.A.append(np.eye(STATE_DIM) + DT * (-k * E.LAP - damp))
            q = np.full(STATE_DIM, 1e-6)
            for kk, idx in enumerate(JUNCTION_IDX): q[idx] = (DT * sd * (st.mult if spike == kk else 1.0)) ** 2 + 1e-6
            self.Q.append(np.diag(q))

    def b(self, t, trip, spike):
        b = np.zeros(STATE_DIM)
        for kk, idx in enumerate(TANK_IDX):
            if trip != kk: b[idx] = DT * self.pg * TANK_SETPOINT
        for kk, idx in enumerate(JUNCTION_IDX):
            b[idx] = -DT * self.bd * diurnal(t, 0.0) * (self.st.mult if spike == kk else 1.0)
        return b


def run_imm(M, obs, mask):
    st = M.st; n = len(st.regimes); x0, P0 = E.prior()
    xs = [x0.copy() for _ in range(n)]; Ps = [P0.copy() for _ in range(n)]; mu = st.mu0.copy()
    out = np.zeros((obs.shape[0], STATE_DIM)); total = 0.0
    for t in range(obs.shape[0]):
        cbar = st.PI.T @ mu + 1e-300; xm, Pm = [], []
        for j in range(n):
            w = st.PI[:, j] * mu / cbar[j]
            xj = sum(w[i] * xs[i] for i in range(n)); Pj = sum(w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(n))
            xm.append(xj); Pm.append(Pj)
        ll = np.zeros(n)
        for j in range(n):
            xp = M.A[j] @ xm[j] + M.b(t, *st.regimes[j]); Pp = M.A[j] @ Pm[j] @ M.A[j].T + M.Q[j]
            xs[j], Ps[j], ll[j] = E.update(xp, Pp, obs[t], mask[t], M.R)
        ll = np.nan_to_num(ll, nan=-1e9, neginf=-1e9); mx = ll.max(); post = cbar * np.exp(ll - mx) + 1e-300
        total += mx + np.log(post.sum()); mu = post / post.sum()
        out[t] = sum(mu[j] * xs[j] for j in range(n))
    return out, total


def job(args):
    name, seed, n_fit = args
    t0 = time.time(); st = Struct(name); rng = np.random.default_rng(900 + seed)
    theta0 = E.TRUE * np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=5))
    train = E.train_set(100 + seed, n_fit)

    def nll(lt):
        M = Model(np.exp(lt), st); return -sum(run_imm(M, o, m)[1] for o, m in train)
    r = minimize(nll, np.log(theta0), method="Nelder-Mead", options={"xatol": 2e-3, "fatol": 1e-2, "maxiter": 500})
    th = np.exp(r.x); M = Model(th, st); se = an = 0.0; n = 0
    for o, m, x in E.test_set():
        e = run_imm(M, o, m)[0] - x; se += float((e ** 2).mean()); an += float(((e @ E.V) ** 2).mean()); n += 1
    res = {"structure": name, "seed": seed, "theta0": theta0.tolist(), "theta": th.tolist(), "rel_err_to_true": (th / E.TRUE - 1).tolist(),
           "init_factor": (theta0 / E.TRUE).tolist(), "nll_fitted": float(r.fun), "nll_at_true_params": float(nll(np.log(E.TRUE))), "nfev": int(r.nfev), "converged": bool(r.success),
           "mse": se / n, "antisym": an / n, "seconds": time.time() - t0}
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for k, v in res.items()}), flush=True)
    return res


if __name__ == "__main__":
    n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    specs = [(nm, s, n_fit) for nm in ("true", "no_spike", "spike_x2", "wrong_rates", "calm_only") for s in range(3)]
    with Pool(nproc) as p:
        res = p.map(job, specs, chunksize=1)
    json.dump({"n_fit": n_fit, "runs": res}, open(os.path.join("..", "results", "est_v25.json"), "w"), indent=1)
    print("done")
