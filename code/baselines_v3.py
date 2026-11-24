import numpy as np, json, os
from data import generate_trajectory, MEASUREMENT_NOISE_STD, TANK_IDX
import kalman_smoother as K
from network import STATE_DIM
from candidates_v3 import C_BARRIER
import data as D

def kf_general(obs, mask, A, bfun, Q, R, x0, P0):
    T = obs.shape[0]; xf = np.zeros((T, STATE_DIM)); Pf = np.zeros((T, STATE_DIM, STATE_DIM))
    xp_ = np.zeros((T, STATE_DIM)); Pp_ = np.zeros((T, STATE_DIM, STATE_DIM))
    x, P = x0.copy(), P0.copy()
    for t in range(T):
        if t == 0: xp, Pp = x, P
        else:
            xp = A @ xf[t-1] + bfun(t-1); Pp = A @ Pf[t-1] @ A.T + Q
        xp_[t] = xp; Pp_[t] = Pp
        avail = np.where(mask[t])[0]
        if len(avail) == 0: xf[t], Pf[t] = xp, Pp; continue
        H = np.zeros((len(avail), STATE_DIM)); H[np.arange(len(avail)), avail] = 1
        Rm = np.eye(len(avail)) * R
        S = H @ Pp @ H.T + Rm; G = Pp @ H.T @ np.linalg.inv(S)
        xf[t] = xp + G @ (obs[t, avail] - H @ xp); Pf[t] = (np.eye(STATE_DIM) - G @ H) @ Pp
    return xf, Pf, xp_, Pp_

def rts(xf, Pf, xp_, Pp_, A):
    T = xf.shape[0]; xs = xf.copy()
    for t in range(T-2, -1, -1):
        Cg = Pf[t] @ A.T @ np.linalg.pinv(Pp_[t+1])
        xs[t] = xf[t] + Cg @ (xs[t+1] - xp_[t+1])
    return xs

def fit_linear(trajs):
    X, Y = [], []
    for x in trajs:
        for t in range(x.shape[0]-1):
            ph = 2*np.pi*t/48
            X.append(np.r_[x[t], 1.0, np.sin(ph), np.cos(ph)]); Y.append(x[t+1])
    X, Y = np.array(X), np.array(Y)
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = B[:STATE_DIM].T; c0 = B[STATE_DIM]; c1 = B[STATE_DIM+1]; c2 = B[STATE_DIM+2]
    resid = Y - X @ B
    return A, c0, c1, c2, np.cov(resid.T)

def eval_set(n=1000, seed=777):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        out.append((o, m, x))
    return out

if __name__ == "__main__":
    A_or = K.build_A(); Q_or, _ = K.calibrate_Q(A_or)
    R = MEASUREMENT_NOISE_STD ** 2
    ev = eval_set()
    rng = np.random.default_rng(1)
    train = [generate_trajectory(T=64, rng=rng, p_event=0.08)[2] for _ in range(300)]
    A_f, c0, c1, c2, Q_f = fit_linear(train)
    def bfun_or(t): return K.exogenous_input(t)
    def bfun_f(t):
        ph = 2*np.pi*t/48; b = np.array([c0, c0, 0,0,0,0]) * 0.0
        return None
    res = {}
    for name, A, Q, bfun in [("oracle", A_or, Q_or, bfun_or)]:
        kf_m, rts_m = [], []
        for o, m, x in ev:
            x0 = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5); P0 = np.eye(STATE_DIM) * 4.0
            xf, Pf, xp, Pp = kf_general(o, m, A, bfun, Q, R, x0, P0)
            kf_m.append(((xf - x) ** 2).mean()); rts_m.append(((rts(xf, Pf, xp, Pp, A) - x) ** 2).mean())
        res[name] = {"kf": float(np.mean(kf_m)), "rts": float(np.mean(rts_m))}
    fitted_off = lambda t: c0 + c1*np.sin(2*np.pi*t/48) + c2*np.cos(2*np.pi*t/48)
    kf_m, rts_m = [], []
    for o, m, x in ev:
        x0 = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5); P0 = np.eye(STATE_DIM) * 4.0
        xf, Pf, xp, Pp = kf_general(o, m, A_f, fitted_off, Q_f, R, x0, P0)
        kf_m.append(((xf - x) ** 2).mean()); rts_m.append(((rts(xf, Pf, xp, Pp, A_f) - x) ** 2).mean())
    res["fitted"] = {"kf": float(np.mean(kf_m)), "rts": float(np.mean(rts_m))}
    res["description"] = ("Kalman filter and RTS smoother MSE on the 1,000 evaluation trajectories (seed 777), "
                          "produced by code/baselines_v3.py. Oracle Q calibrated on seed 4242, not an evaluation seed.")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "baselines.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("BASE " + json.dumps(res), flush=True)
