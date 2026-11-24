import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from benchmark import ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, TANK_IDX, JUNCTION_IDX
from benchmark import STATE_DIM
from benchmark import simulate, diurnal
from benchmark import TANK_SETPOINT, TANK_IDX, JUNCTION_IDX
from benchmark import diurnal
import sys
DT = 0.5
V = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)
LAP = np.diag(ADJ.sum(axis=1)) - ADJ
REGIMES = [(None, None)] + [(k, None) for k in range(2)] + [(None, j) for j in range(4)]
SIGMA_D = 0.05
TRUE = np.array([FLOW_GAIN, PUMP_GAIN, BASE_DEMAND, SIGMA_D, MEASUREMENT_NOISE_STD], dtype=float)
p_trip, p_spike, exit_p = (0.08 / 3 / 2, 0.08 / 3 / 4, 1.0 / 6.0)
PI = np.zeros((7, 7))
PI[0, 1:3] = p_trip
PI[0, 3:] = p_spike
PI[0, 0] = 1 - PI[0, 1:].sum()
for _r in range(1, 7):
    PI[_r, 0] = exit_p
    PI[_r, _r] = 1 - exit_p
MU0 = np.array([1 - 0.0533] + [0.0133] * 2 + [0.00667] * 4)
MU0 = MU0 / MU0.sum()

class Model:

    def __init__(self, theta):
        k, pg, bd, sd, sv = [float(v) for v in theta]
        self.R = sv ** 2
        self.A, self.Q, self.pg, self.bd = ([], [], pg, bd)
        for trip, spike in REGIMES:
            damp = np.zeros((STATE_DIM, STATE_DIM))
            for kk, idx in enumerate(TANK_IDX):
                if trip != kk:
                    damp[idx, idx] = pg
            self.A.append(np.eye(STATE_DIM) + DT * (-k * LAP - damp))
            q = np.full(STATE_DIM, 1e-06)
            for kk, idx in enumerate(JUNCTION_IDX):
                q[idx] = (DT * sd * (4.0 if spike == kk else 1.0)) ** 2 + 1e-06
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
        return (x, P, 0.0)
    H = np.zeros((len(av), STATE_DIM))
    H[np.arange(len(av)), av] = 1.0
    P = 0.5 * (P + P.T)
    S = H @ P @ H.T + R * np.eye(len(av))
    K = P @ H.T @ np.linalg.inv(S)
    innov = z[av] - H @ x
    ll = -0.5 * (innov @ np.linalg.solve(S, innov) + np.linalg.slogdet(S)[1] + len(av) * np.log(2 * np.pi))
    Pn = (np.eye(STATE_DIM) - K @ H) @ P
    return (x + K @ innov, 0.5 * (Pn + Pn.T), ll)

def prior():
    m = np.zeros(STATE_DIM)
    m[TANK_IDX] = TANK_SETPOINT
    m[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    return (m, np.eye(STATE_DIM) * 0.0001)

def run_imm(M, obs, mask):
    n = len(REGIMES)
    x0, P0 = prior()
    xs = [x0.copy() for _ in range(n)]
    Ps = [P0.copy() for _ in range(n)]
    mu = MU0.copy()
    out = np.zeros((obs.shape[0], STATE_DIM))
    total = 0.0
    for t in range(obs.shape[0]):
        cbar = PI.T @ mu
        xm, Pm = ([], [])
        for j in range(n):
            w = PI[:, j] * mu / cbar[j]
            xj = sum((w[i] * xs[i] for i in range(n)))
            Pj = sum((w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(n)))
            xm.append(xj)
            Pm.append(Pj)
        ll = np.zeros(n)
        for j in range(n):
            xp = M.A[j] @ xm[j] + M.b(t, *REGIMES[j])
            Pp = M.A[j] @ Pm[j] @ M.A[j].T + M.Q[j]
            xs[j], Ps[j], ll[j] = update(xp, Pp, obs[t], mask[t], M.R)
        ll = np.nan_to_num(ll, nan=-1000000000.0, neginf=-1000000000.0)
        mx = ll.max()
        post = cbar * np.exp(ll - mx) + 1e-300
        total += mx + np.log(post.sum())
        mu = post / post.sum()
        out[t] = sum((mu[j] * xs[j] for j in range(n)))
    return (out, total)

def run_kf(M, obs, mask, Q):
    x, P = prior()
    out = np.zeros((obs.shape[0], STATE_DIM))
    total = 0.0
    for t in range(obs.shape[0]):
        xp = M.A[0] @ x + M.b(t, None, None)
        Pp = M.A[0] @ P @ M.A[0].T + Q
        x, P, ll = update(xp, Pp, obs[t], mask[t], M.R)
        out[t] = x
        total += ll
    return (out, total)

def kf_Q(sd):
    q = np.full(STATE_DIM, 1e-06)
    for idx in JUNCTION_IDX:
        q[idx] = (DT * sd) ** 2 + 1e-06
    return np.diag(q)

def train_set(seed, n):
    rng = np.random.default_rng(seed)
    return [simulate(T=64, rng=rng)[:2] for _ in range(n)]

def fit(kind, train, theta0):

    def nll(lt):
        th = np.exp(lt)
        M = Model(th)
        tot = 0.0
        for o, m in train:
            tot += run_imm(M, o, m)[1] if kind == 'imm' else run_kf(M, o, m, kf_Q(th[3]))[1]
        return -tot
    r = minimize(nll, np.log(theta0), method='Nelder-Mead', options={'xatol': 0.002, 'fatol': 0.01, 'maxiter': 220})
    return (np.exp(r.x), float(r.fun), int(r.nfev))

def evaluate(kind, theta, tests):
    M = Model(theta)
    se = an = 0.0
    n = 0
    for o, m, x in tests:
        e = (run_imm(M, o, m)[0] if kind == 'imm' else run_kf(M, o, m, kf_Q(theta[3]))[0]) - x
        se += float((e ** 2).mean())
        an += float(((e @ V) ** 2).mean())
        n += 1
    return (se / n, an / n)

def test_set():
    out = []
    for s in (777, 888, 999):
        rng = np.random.default_rng(s)
        for _ in range(300):
            o, m, x, *_ = simulate(T=64, rng=rng)
            out.append((o, m, x))
    return out

def job(args):
    kind, seed, n_fit = args
    t0 = time.time()
    rng = np.random.default_rng(500 + seed)
    theta0 = TRUE * rng.uniform(0.6, 1.5, size=5)
    train = train_set(100 + seed, n_fit)
    th, f, nfev = fit(kind, train, theta0)
    mse, an = evaluate(kind, th, test_set())
    r = {'kind': kind, 'seed': seed, 'theta0': theta0.tolist(), 'theta': th.tolist(), 'rel_err_to_true': (th / TRUE - 1).tolist(), 'mse': mse, 'antisym': an, 'nfev': nfev, 'seconds': time.time() - t0}
    print(json.dumps({k: np.round(v, 4).tolist() if isinstance(v, (list, float)) else v for k, v in r.items()}), flush=True)
    return r

class Struct:

    def __init__(self, name):
        self.name = name
        mult = 2.0 if name == 'spike_x2' else 4.0
        self.mult = mult
        if name == 'calm_only':
            self.regimes = [(None, None)]
        elif name == 'no_spike':
            self.regimes = [(None, None), (0, None), (1, None)]
        else:
            self.regimes = [(None, None)] + [(k, None) for k in range(2)] + [(None, j) for j in range(4)]
        n = len(self.regimes)
        scale = 4.0 if name == 'wrong_rates' else 1.0
        exit_p = 1.0 / 3.0 if name == 'wrong_rates' else 1.0 / 6.0
        p_trip, p_spike = (0.08 / 3 / 2 * scale, 0.08 / 3 / 4 * scale)
        PI = np.zeros((n, n))
        PI[0, 0] = 1.0
        if n > 1:
            PI[0, 1:3] = p_trip
            if n > 3:
                PI[0, 3:] = p_spike
            PI[0, 0] = 1 - PI[0, 1:].sum()
            for r in range(1, n):
                PI[r, 0] = exit_p
                PI[r, r] = 1 - exit_p
        self.PI = PI
        mu = PI[0].copy() if n > 1 else np.ones(1)
        self.mu0 = mu / mu.sum()

class Model_wrong_structure:

    def __init__(self, theta, st):
        k, pg, bd, sd, sv = [float(v) for v in theta]
        self.R = sv ** 2
        self.pg, self.bd, self.st = (pg, bd, st)
        self.A, self.Q = ([], [])
        for trip, spike in st.regimes:
            damp = np.zeros((STATE_DIM, STATE_DIM))
            for kk, idx in enumerate(TANK_IDX):
                if trip != kk:
                    damp[idx, idx] = pg
            self.A.append(np.eye(STATE_DIM) + DT * (-k * LAP - damp))
            q = np.full(STATE_DIM, 1e-06)
            for kk, idx in enumerate(JUNCTION_IDX):
                q[idx] = (DT * sd * (st.mult if spike == kk else 1.0)) ** 2 + 1e-06
            self.Q.append(np.diag(q))

    def b(self, t, trip, spike):
        b = np.zeros(STATE_DIM)
        for kk, idx in enumerate(TANK_IDX):
            if trip != kk:
                b[idx] = DT * self.pg * TANK_SETPOINT
        for kk, idx in enumerate(JUNCTION_IDX):
            b[idx] = -DT * self.bd * diurnal(t, 0.0) * (self.st.mult if spike == kk else 1.0)
        return b

def run_imm_wrong_structure(M, obs, mask):
    st = M.st
    n = len(st.regimes)
    x0, P0 = prior()
    xs = [x0.copy() for _ in range(n)]
    Ps = [P0.copy() for _ in range(n)]
    mu = st.mu0.copy()
    out = np.zeros((obs.shape[0], STATE_DIM))
    total = 0.0
    for t in range(obs.shape[0]):
        cbar = st.PI.T @ mu + 1e-300
        xm, Pm = ([], [])
        for j in range(n):
            w = st.PI[:, j] * mu / cbar[j]
            xj = sum((w[i] * xs[i] for i in range(n)))
            Pj = sum((w[i] * (Ps[i] + np.outer(xs[i] - xj, xs[i] - xj)) for i in range(n)))
            xm.append(xj)
            Pm.append(Pj)
        ll = np.zeros(n)
        for j in range(n):
            xp = M.A[j] @ xm[j] + M.b(t, *st.regimes[j])
            Pp = M.A[j] @ Pm[j] @ M.A[j].T + M.Q[j]
            xs[j], Ps[j], ll[j] = update(xp, Pp, obs[t], mask[t], M.R)
        ll = np.nan_to_num(ll, nan=-1000000000.0, neginf=-1000000000.0)
        mx = ll.max()
        post = cbar * np.exp(ll - mx) + 1e-300
        total += mx + np.log(post.sum())
        mu = post / post.sum()
        out[t] = sum((mu[j] * xs[j] for j in range(n)))
    return (out, total)

def job_wrong_structure(args):
    name, seed, n_fit = args
    t0 = time.time()
    st = Struct(name)
    rng = np.random.default_rng(900 + seed)
    theta0 = TRUE * np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=5))
    train = train_set(100 + seed, n_fit)

    def nll(lt):
        M = Model_wrong_structure(np.exp(lt), st)
        return -sum((run_imm_wrong_structure(M, o, m)[1] for o, m in train))
    r = minimize(nll, np.log(theta0), method='Nelder-Mead', options={'xatol': 0.002, 'fatol': 0.01, 'maxiter': 500})
    th = np.exp(r.x)
    M = Model_wrong_structure(th, st)
    se = an = 0.0
    n = 0
    for o, m, x in test_set():
        e = run_imm_wrong_structure(M, o, m)[0] - x
        se += float((e ** 2).mean())
        an += float(((e @ V) ** 2).mean())
        n += 1
    res = {'structure': name, 'seed': seed, 'theta0': theta0.tolist(), 'theta': th.tolist(), 'rel_err_to_true': (th / TRUE - 1).tolist(), 'init_factor': (theta0 / TRUE).tolist(), 'nll_fitted': float(r.fun), 'nll_at_true_params': float(nll(np.log(TRUE))), 'nfev': int(r.nfev), 'converged': bool(r.success), 'mse': se / n, 'antisym': an / n, 'seconds': time.time() - t0}
    print(json.dumps({k: np.round(v, 4).tolist() if isinstance(v, (list, float)) else v for k, v in res.items()}), flush=True)
    return res

def job_multistart(args):
    name, seed, start, n_fit = args
    t0 = time.time()
    st = Struct(name)
    rng = np.random.default_rng(900 + seed + 1000 * start)
    theta0 = TRUE * np.exp(rng.uniform(np.log(0.2), np.log(5.0), size=5))
    train = train_set(100 + seed, n_fit)

    def nll(lt):
        M = Model_wrong_structure(np.exp(lt), st)
        return -sum((run_imm_wrong_structure(M, o, m)[1] for o, m in train))
    r = minimize(nll, np.log(theta0), method='Nelder-Mead', options={'xatol': 0.002, 'fatol': 0.01, 'maxiter': 500})
    th = np.exp(r.x)
    M = Model_wrong_structure(th, st)
    se = an = 0.0
    n = 0
    for o, m, x in test_set():
        e = run_imm_wrong_structure(M, o, m)[0] - x
        se += float((e ** 2).mean())
        an += float(((e @ V) ** 2).mean())
        n += 1
    res = {'structure': name, 'seed': seed, 'start': start, 'theta0': theta0.tolist(), 'theta': th.tolist(), 'rel_err_to_true': (th / TRUE - 1).tolist(), 'nll_fitted': float(r.fun), 'nll_at_true_params': float(nll(np.log(TRUE))), 'nfev': int(r.nfev), 'mse': se / n, 'antisym': an / n, 'seconds': time.time() - t0}
    print(json.dumps({k: np.round(v, 4).tolist() if isinstance(v, (list, float)) else v for k, v in res.items() if k != 'theta0'}), flush=True)
    return res
def job_independent(args):
    seed, start, n_fit = args
    t0 = time.time()
    rng = np.random.default_rng(1700 + seed + 1000 * start)
    lo = np.log([0.05, 0.1, 0.05, 0.005, 0.005])
    hi = np.log([1.0, 2.0, 1.5, 0.3, 0.2])
    theta0 = np.exp(rng.uniform(lo, hi))
    train = train_set(100 + seed, n_fit)

    def nll(lt):
        M = Model(np.exp(lt))
        return -sum((run_imm(M, o, m)[1] for o, m in train))
    r = minimize(nll, np.log(theta0), method='Nelder-Mead', options={'xatol': 0.002, 'fatol': 0.01, 'maxiter': 500})
    th = np.exp(r.x)
    mse, an = evaluate('imm', th, test_set())
    res = {'seed': seed, 'start': start, 'theta0': theta0.tolist(), 'theta': th.tolist(), 'rel_err_to_true': (th / TRUE - 1).tolist(), 'nll_fitted': float(r.fun), 'nll_at_true_params': float(nll(np.log(TRUE))), 'nfev': int(r.nfev), 'mse': mse, 'antisym': an, 'seconds': time.time() - t0}
    print(json.dumps({k: np.round(v, 4).tolist() if isinstance(v, (list, float)) else v for k, v in res.items() if k != 'theta0'}), flush=True)
    return res
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'identification':
        n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
        nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        tests = test_set()
        ref = {'imm_true': evaluate('imm', TRUE, tests), 'kf_true': evaluate('kf', TRUE, tests)}
        print('reference (true parameters):', ref, flush=True)
        specs = [(k, s, n_fit) for k in ('imm', 'kf') for s in range(5)]
        with Pool(nproc) as p:
            res = p.map(job, specs, chunksize=1)
        json.dump({'n_fit': n_fit, 'reference': ref, 'runs': res}, open(os.path.join('..', 'results', 'sensor_only_identification.json'), 'w'), indent=1)
    elif _cmd == 'wrong_structure':
        n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        specs = [(nm, s, n_fit) for nm in ('true', 'no_spike', 'spike_x2', 'wrong_rates', 'calm_only') for s in range(3)]
        with Pool(nproc) as p:
            res = p.map(job_wrong_structure, specs, chunksize=1)
        json.dump({'n_fit': n_fit, 'runs': res}, open(os.path.join('..', 'results', 'sensor_only_wrong_structure.json'), 'w'), indent=1)
        print('done')
    elif _cmd == 'independent_start':
        n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 12
        specs = [(s, j, n_fit) for j in range(n_starts) for s in range(3)]
        with Pool(nproc) as p:
            res = p.map(job_independent, specs, chunksize=1)
        json.dump({'n_fit': n_fit, 'n_starts': n_starts, 'runs': res}, open(os.path.join('..', 'results', 'sensor_only_independent_start.json'), 'w'), indent=1)
        print('done')
    elif _cmd == 'kf_diagnosis':
        d = json.load(open(os.path.join('..', 'results', 'sensor_only_identification.json')))
        n_fit = d['n_fit']
        rng_t = np.random.default_rng(777)
        tests = []
        for _ in range(300):
            o, m, x, kinds = simulate(T=64, rng=rng_t)[:4]
            tests.append((o, m, x, np.array([k in ('pump_trip', 'demand_spike') for k in kinds])))

        def by_regime(est_fn):
            calm = ev = 0.0
            nc = ne = 0
            for o, m, x, act in tests:
                e = ((est_fn(o, m) - x) ** 2).mean(axis=1)
                calm += float(e[~act].sum())
                nc += int((~act).sum())
                ev += float(e[act].sum())
                ne += int(act.sum())
            return (calm / nc, ev / ne)
        rows = []
        for run in d['runs']:
            if run['kind'] != 'kf':
                continue
            s_ = run['seed']
            th = np.array(run['theta'])
            train = train_set(100 + s_, n_fit)
            nll = lambda t: -sum((run_kf(Model(t), o, m, kf_Q(t[3]))[1] for o, m in train))
            th_fix = th.copy()

            def nll4(lt):
                t = np.exp(lt)
                full = np.array([t[0], t[1], t[2], TRUE[3], t[3]])
                return nll(full)
            r4 = minimize(nll4, np.log(th[[0, 1, 2, 4]]), method='Nelder-Mead', options={'xatol': 0.002, 'fatol': 0.01, 'maxiter': 220})
            t4 = np.exp(r4.x)
            th_fix = np.array([t4[0], t4[1], t4[2], TRUE[3], t4[3]])
            row = {'seed': s_, 'theta_fitted': th.tolist(), 'sigma_d_fitted': float(th[3]), 'nll_kf_fitted': float(nll(th)), 'nll_kf_true_params': float(nll(TRUE)),
                   'nll_imm_true_params': float(-sum((run_imm(Model(TRUE), o, m)[1] for o, m in train))), 'theta_sigma_d_fixed': th_fix.tolist(), 'nll_kf_sigma_d_fixed': float(r4.fun)}
            row['mse_calm_events_kf_fitted'] = by_regime(lambda o, m: run_kf(Model(th), o, m, kf_Q(th[3]))[0])
            row['mse_calm_events_kf_true_params'] = by_regime(lambda o, m: run_kf(Model(TRUE), o, m, kf_Q(TRUE[3]))[0])
            row['mse_calm_events_kf_sigma_d_fixed'] = by_regime(lambda o, m: run_kf(Model(th_fix), o, m, kf_Q(th_fix[3]))[0])
            row['mse_calm_events_imm_true_params'] = by_regime(lambda o, m: run_imm(Model(TRUE), o, m)[0])
            rows.append(row)
            print(json.dumps({k: np.round(v, 4).tolist() if isinstance(v, (list, float)) else v for k, v in row.items()}), flush=True)
        json.dump({'n_fit': n_fit, 'rows': rows}, open(os.path.join('..', 'results', 'single_regime_kf_diagnosis.json'), 'w'), indent=1)
    elif _cmd == 'multistart':
        n_fit = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        specs = [(nm, s, j, n_fit) for j in range(n_starts) for nm in ('true', 'no_spike', 'spike_x2', 'wrong_rates', 'calm_only') for s in range(3)]
        with Pool(nproc) as p:
            res = p.map(job_multistart, specs, chunksize=1)
        json.dump({'n_fit': n_fit, 'n_starts': n_starts, 'runs': res}, open(os.path.join('..', 'results', 'sensor_only_multistart.json'), 'w'), indent=1)
        print('done')
    else:
        raise SystemExit('usage: python sensor_only_identification.py <identification|wrong_structure|independent_start|kf_diagnosis|multistart> [args]')
