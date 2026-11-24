import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from data import ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, TANK_IDX, JUNCTION_IDX
from network import STATE_DIM
from simulator import simulate, diurnal

DT = 0.5
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
LAP = np.diag(ADJ.sum(axis=1)) - ADJ
REGIMES = [(None, None)] + [(k, None) for k in range(2)] + [(None, j) for j in range(4)]
SIGMA_D = 0.05
TRUE = np.array([FLOW_GAIN, PUMP_GAIN, BASE_DEMAND, SIGMA_D, MEASUREMENT_NOISE_STD], dtype=float)
p_trip, p_spike, exit_p = 0.08 / 3 / 2, 0.08 / 3 / 4, 1.0 / 6.0
PI = np.zeros((7, 7))
PI[0, 1:3] = p_trip; PI[0, 3:] = p_spike; PI[0, 0] = 1 - PI[0, 1:].sum()
for _r in range(1, 7):
    PI[_r, 0] = exit_p; PI[_r, _r] = 1 - exit_p
MU0 = np.array([1 - 0.0533] + [0.0133] * 2 + [0.00667] * 4); MU0 = MU0 / MU0.sum()


class Model:
    def __init__(self, theta):
        k, pg, bd, sd, sv = [float(v) for v in theta]
        self.R = sv ** 2
        self.A, self.Q, self.pg, self.bd = [], [], pg, bd
        for (trip, spike) in REGIMES:
            damp = np.zeros((STATE_DIM, STATE_DIM))
            for kk, idx in enumerate(TANK_IDX):
                if trip != kk:
                    damp[idx, idx] = pg
            self.A.append(np.eye(STATE_DIM) + DT * (-k * LAP - damp))
            q = np.full(STATE_DIM, 1e-6)
            for kk, idx in enumerate(JUNCTION_IDX):
                q[idx] = (DT * sd * (4.0 if spike == kk else 1.0)) ** 2 + 1e-6
            self.Q.append(np.diag(q))

    def b(self, t, trip, spike):
        b = np.zeros(STATE_DIM)
        for kk, idx in enumerate(TANK_IDX):
            if trip != kk:
                b[idx] = DT * self.pg * TANK_SETPOINT
        for kk, idx in enumerate(JUNCTION_IDX):
            b[idx] = -DT * self.bd * diurnal(t, 0.0) * (4.0 if spike == kk else 1.0)
        return b


def update(x, P, z, mask, R):
    av = np.where(mask)[0]
    if len(av) == 0:
        return x, P, 0.0
    H = np.zeros((len(av), STATE_DIM)); H[np.arange(len(av)), av] = 1.0
    P = 0.5 * (P + P.T)
    S = H @ P @ H.T + R * np.eye(len(av))
    K = P @ H.T @ np.linalg.inv(S)
    innov = z[av] - H @ x
    ll = -0.5 * (innov @ np.linalg.solve(S, innov) + np.linalg.slogdet(S)[1] + len(av) * np.log(2 * np.pi))
    Pn = (np.eye(STATE_DIM) - K @ H) @ P
    return x + K @ innov, 0.5 * (Pn + Pn.T), ll


def prior():
    m = np.zeros(STATE_DIM); m[TANK_IDX] = TANK_SETPOINT; m[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    return m, np.eye(STATE_DIM) * 1e-4


def run_imm(M, obs, mask):
    n = len(REGIMES); x0, P0 = prior()
    xs = [x0.copy() for _ in range(n)]; Ps = [P0.copy() for _ in range(n)]; mu = MU0.copy()
    out = np.zeros((obs.shape[0], STATE_DIM)); total = 0.0
    for t in range(obs.shape[0]):
        cbar = PI.T @ mu; xm, Pm = [], []
        for j in range(n):
            w = PI[:, j] * mu / cbar[j]
            xj = sum(w[i] * xs[i] for i in range(n))
            Pj = sum(w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(n))
            xm.append(xj); Pm.append(Pj)
        ll = np.zeros(n)
        for j in range(n):
            xp = M.A[j] @ xm[j] + M.b(t, *REGIMES[j]); Pp = M.A[j] @ Pm[j] @ M.A[j].T + M.Q[j]
            xs[j], Ps[j], ll[j] = update(xp, Pp, obs[t], mask[t], M.R)
        ll = np.nan_to_num(ll, nan=-1e9, neginf=-1e9); mx = ll.max()
        post = cbar * np.exp(ll - mx) + 1e-300
        total += mx + np.log(post.sum()); mu = post / post.sum()
        out[t] = sum(mu[j] * xs[j] for j in range(n))
    return out, total


def run_kf(M, obs, mask, Q):
    x, P = prior(); out = np.zeros((obs.shape[0], STATE_DIM)); total = 0.0
    for t in range(obs.shape[0]):
        xp = M.A[0] @ x + M.b(t, None, None); Pp = M.A[0] @ P @ M.A[0].T + Q
        x, P, ll = update(xp, Pp, obs[t], mask[t], M.R); out[t] = x; total += ll
    return out, total


def kf_Q(sd):
    q = np.full(STATE_DIM, 1e-6)
    for idx in JUNCTION_IDX:
        q[idx] = (DT * sd) ** 2 + 1e-6
    return np.diag(q)


def train_set(seed, n):
    rng = np.random.default_rng(seed)
    return [simulate(T=64, rng=rng)[:2] for _ in range(n)]


def fit(kind, train, theta0):
    def nll(lt):
        th = np.exp(lt)
        M = Model(th); tot = 0.0
        for o, m in train:
            tot += (run_imm(M, o, m)[1] if kind == "imm" else run_kf(M, o, m, kf_Q(th[3]))[1])
        return -tot
    r = minimize(nll, np.log(theta0), method="Nelder-Mead", options={"xatol": 2e-3, "fatol": 1e-2, "maxiter": 220})
    return np.exp(r.x), float(r.fun), int(r.nfev)


def evaluate(kind, theta, tests):
    M = Model(theta); se = an = 0.0; n = 0
    for o, m, x in tests:
        e = (run_imm(M, o, m)[0] if kind == "imm" else run_kf(M, o, m, kf_Q(theta[3]))[0]) - x
        se += float((e ** 2).mean()); an += float(((e @ V) ** 2).mean()); n += 1
    return se / n, an / n


def test_set():
    out = []
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        for _ in range(300):
            o, m, x, *_ = simulate(T=64, rng=rng); out.append((o, m, x))
    return out


def job(args):
    kind, seed, n_fit = args
    t0 = time.time(); rng = np.random.default_rng(500 + seed)
    theta0 = TRUE * rng.uniform(0.6, 1.5, size=5)
    train = train_set(100 + seed, n_fit)
    th, f, nfev = fit(kind, train, theta0)
    mse, an = evaluate(kind, th, test_set())
    r = {"kind": kind, "seed": seed, "theta0": theta0.tolist(), "theta": th.tolist(), "rel_err_to_true": (th / TRUE - 1).tolist(),
         "mse": mse, "antisym": an, "nfev": nfev, "seconds": time.time() - t0}
    print(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v) for k, v in r.items()}), flush=True)
    return r


if __name__ == "__main__":
    n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    tests = test_set()
    ref = {"imm_true": evaluate("imm", TRUE, tests), "kf_true": evaluate("kf", TRUE, tests)}
    print("reference (true parameters):", ref, flush=True)
    specs = [(k, s, n_fit) for k in ("imm", "kf") for s in range(5)]
    with Pool(nproc) as p:
        res = p.map(job, specs, chunksize=1)
    json.dump({"n_fit": n_fit, "reference": ref, "runs": res}, open(os.path.join("..", "results", "sensor_only_identification.json"), "w"), indent=1)
