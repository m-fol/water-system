import numpy as np
import torch
from benchmark import STATE_DIM
from benchmark import ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, TANK_IDX, JUNCTION_IDX, generate_trajectory, generate_batch, _diurnal_factor
from neural_models import ARCHITECTURES
import numpy as np, json, os
from benchmark import generate_trajectory, MEASUREMENT_NOISE_STD, TANK_IDX
from neural_models import C_BARRIER
import benchmark as D
import sys
DT = 0.5

def build_A():
    L = np.diag(ADJ.sum(axis=1)) - ADJ
    tank_damp = np.zeros((STATE_DIM, STATE_DIM))
    for idx in TANK_IDX:
        tank_damp[idx, idx] = PUMP_GAIN
    return np.eye(STATE_DIM) + DT * (-FLOW_GAIN * L - tank_damp)

def exogenous_input(t):
    b = np.zeros(STATE_DIM)
    for idx in TANK_IDX:
        b[idx] = DT * PUMP_GAIN * TANK_SETPOINT
    for idx in JUNCTION_IDX:
        b[idx] = -DT * BASE_DEMAND * _diurnal_factor(t)
    return b

def calibrate_Q(A, n_traj=200, T=64, p_event=0.08, seed=4242):
    rng = np.random.default_rng(seed)
    residuals = []
    for _ in range(n_traj):
        _, _, x_true, _ = generate_trajectory(T=T, rng=rng, p_event=p_event)
        for t in range(T - 1):
            pred = A @ x_true[t] + exogenous_input(t)
            residuals.append(x_true[t + 1] - pred)
    residuals = np.array(residuals)
    Q = np.cov(residuals.T)
    return (Q, residuals)

def kalman_filter(obs, obs_mask, A, Q, R_diag):
    T = obs.shape[0]
    x_filt = np.zeros((T, STATE_DIM))
    P_filt = np.zeros((T, STATE_DIM, STATE_DIM))
    x_pred = np.zeros((T, STATE_DIM))
    P_pred = np.zeros((T, STATE_DIM, STATE_DIM))
    x = np.full(STATE_DIM, TANK_SETPOINT - 0.5)
    P = np.eye(STATE_DIM) * 4.0
    for t in range(T):
        if t == 0:
            xp, Pp = (x, P)
        else:
            xp = A @ x_filt[t - 1] + exogenous_input(t - 1)
            Pp = A @ P_filt[t - 1] @ A.T + Q
        x_pred[t], P_pred[t] = (xp, Pp)
        avail = np.where(obs_mask[t])[0]
        if len(avail) == 0:
            x_filt[t], P_filt[t] = (xp, Pp)
            continue
        H = np.zeros((len(avail), STATE_DIM))
        for i, idx in enumerate(avail):
            H[i, idx] = 1.0
        R = np.eye(len(avail)) * MEASUREMENT_NOISE_STD ** 2
        z = obs[t, avail]
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        x_filt[t] = xp + K @ (z - H @ xp)
        P_filt[t] = (np.eye(STATE_DIM) - K @ H) @ Pp
    return (x_filt, P_filt, x_pred, P_pred)

def rts_smoother(x_filt, P_filt, x_pred, P_pred, A):
    T = x_filt.shape[0]
    x_smooth = x_filt.copy()
    for t in range(T - 2, -1, -1):
        C = P_filt[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
    return x_smooth

def run_comparison(n_scenarios=200, T=64, p_event=0.08, seed=123, ckpt_path='checkpoint_plots_barrier_a0.3.pt', architecture='gnn'):
    A = build_A()
    print('Calibrating Q from simulated one-step residuals...')
    Q, _ = calibrate_Q(A)
    print(f'Q diagonal (tanks, junctions): {np.round(np.diag(Q), 5)}')
    observer = ARCHITECTURES[architecture]()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    observer.load_state_dict(ckpt['observer'])
    observer.eval()
    rng = np.random.default_rng(seed)
    obs_np, mask_np, x_np = generate_batch(batch_size=n_scenarios, T=T, rng=rng, p_event=p_event)
    with torch.no_grad():
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        x_hat_lstm, _ = observer(obs_t)
        x_hat_lstm = x_hat_lstm.numpy()
    mse_filter = np.zeros((n_scenarios, STATE_DIM))
    mse_smooth = np.zeros((n_scenarios, STATE_DIM))
    mse_lstm = np.zeros((n_scenarios, STATE_DIM))
    for i in range(n_scenarios):
        x_filt, P_filt, x_pred, P_pred = kalman_filter(obs_np[i], mask_np[i], A, Q, None)
        x_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, A)
        err_f = x_filt - x_np[i]
        err_s = x_smooth - x_np[i]
        err_l = x_hat_lstm[i] - x_np[i]
        mse_filter[i] = (err_f ** 2).mean(axis=0)
        mse_smooth[i] = (err_s ** 2).mean(axis=0)
        mse_lstm[i] = (err_l ** 2).mean(axis=0)
    names = ['T1', 'T2', 'J1', 'J2', 'J3', 'J4']

    def summarize(mse, label):
        overall = mse.mean()
        tank = mse[:, TANK_IDX].mean()
        junc = mse[:, JUNCTION_IDX].mean()
        print(f'\n{label}:')
        print(f'  overall MSE: {overall:.5f}   tanks: {tank:.5f}   junctions: {junc:.5f}')
        for j, n in enumerate(names):
            print(f'    {n}: {mse[:, j].mean():.5f}')
        return (overall, tank, junc)
    print(f'\n=== {n_scenarios} held-out trajectories, T={T}, p_event={p_event} ===')
    r_f = summarize(mse_filter, 'Kalman filter (causal)')
    r_s = summarize(mse_smooth, 'RTS smoother (non-causal, whole trajectory)')
    r_l = summarize(mse_lstm, f'Trained LSTM+{architecture} observer (causal, {ckpt_path})')
    print('\n=== Summary table (overall / tank / junction MSE) ===')
    print(f'  Kalman filter        : {r_f[0]:.5f} / {r_f[1]:.5f} / {r_f[2]:.5f}')
    print(f'  RTS smoother         : {r_s[0]:.5f} / {r_s[1]:.5f} / {r_s[2]:.5f}')
    print(f'  LSTM+{architecture:<10s}: {r_l[0]:.5f} / {r_l[1]:.5f} / {r_l[2]:.5f}')
    return dict(kalman=mse_filter, smoother=mse_smooth, lstm=mse_lstm, obs=obs_np, mask=mask_np, x_true=x_np, x_hat_lstm=x_hat_lstm, A=A, Q=Q)

def kf_general(obs, mask, A, bfun, Q, R, x0, P0):
    T = obs.shape[0]
    xf = np.zeros((T, STATE_DIM))
    Pf = np.zeros((T, STATE_DIM, STATE_DIM))
    xp_ = np.zeros((T, STATE_DIM))
    Pp_ = np.zeros((T, STATE_DIM, STATE_DIM))
    x, P = (x0.copy(), P0.copy())
    for t in range(T):
        if t == 0:
            xp, Pp = (x, P)
        else:
            xp = A @ xf[t - 1] + bfun(t - 1)
            Pp = A @ Pf[t - 1] @ A.T + Q
        xp_[t] = xp
        Pp_[t] = Pp
        avail = np.where(mask[t])[0]
        if len(avail) == 0:
            xf[t], Pf[t] = (xp, Pp)
            continue
        H = np.zeros((len(avail), STATE_DIM))
        H[np.arange(len(avail)), avail] = 1
        Rm = np.eye(len(avail)) * R
        S = H @ Pp @ H.T + Rm
        G = Pp @ H.T @ np.linalg.inv(S)
        xf[t] = xp + G @ (obs[t, avail] - H @ xp)
        Pf[t] = (np.eye(STATE_DIM) - G @ H) @ Pp
    return (xf, Pf, xp_, Pp_)

def rts(xf, Pf, xp_, Pp_, A):
    T = xf.shape[0]
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        Cg = Pf[t] @ A.T @ np.linalg.pinv(Pp_[t + 1])
        xs[t] = xf[t] + Cg @ (xs[t + 1] - xp_[t + 1])
    return xs

def fit_linear(trajs):
    X, Y = ([], [])
    for x in trajs:
        for t in range(x.shape[0] - 1):
            ph = 2 * np.pi * t / 48
            X.append(np.r_[x[t], 1.0, np.sin(ph), np.cos(ph)])
            Y.append(x[t + 1])
    X, Y = (np.array(X), np.array(Y))
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = B[:STATE_DIM].T
    c0 = B[STATE_DIM]
    c1 = B[STATE_DIM + 1]
    c2 = B[STATE_DIM + 2]
    resid = Y - X @ B
    return (A, c0, c1, c2, np.cov(resid.T))

def eval_set(n=1000, seed=777):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        out.append((o, m, x))
    return out
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'smoother':
        run_comparison()
    elif _cmd == 'baselines_uncorrected':
        A_or = build_A()
        Q_or, _ = calibrate_Q(A_or)
        R = MEASUREMENT_NOISE_STD ** 2
        ev = eval_set()
        rng = np.random.default_rng(1)
        train = [generate_trajectory(T=64, rng=rng, p_event=0.08)[2] for _ in range(300)]
        A_f, c0, c1, c2, Q_f = fit_linear(train)

        def bfun_or(t):
            return exogenous_input(t)

        def bfun_f(t):
            ph = 2 * np.pi * t / 48
            b = np.array([c0, c0, 0, 0, 0, 0]) * 0.0
            return None
        res = {}
        for name, A, Q, bfun in [('oracle', A_or, Q_or, bfun_or)]:
            kf_m, rts_m = ([], [])
            for o, m, x in ev:
                x0 = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5)
                P0 = np.eye(STATE_DIM) * 4.0
                xf, Pf, xp, Pp = kf_general(o, m, A, bfun, Q, R, x0, P0)
                kf_m.append(((xf - x) ** 2).mean())
                rts_m.append(((rts(xf, Pf, xp, Pp, A) - x) ** 2).mean())
            res[name] = {'kf': float(np.mean(kf_m)), 'rts': float(np.mean(rts_m))}
        fitted_off = lambda t: c0 + c1 * np.sin(2 * np.pi * t / 48) + c2 * np.cos(2 * np.pi * t / 48)
        kf_m, rts_m = ([], [])
        for o, m, x in ev:
            x0 = np.full(STATE_DIM, D.TANK_SETPOINT - 0.5)
            P0 = np.eye(STATE_DIM) * 4.0
            xf, Pf, xp, Pp = kf_general(o, m, A_f, fitted_off, Q_f, R, x0, P0)
            kf_m.append(((xf - x) ** 2).mean())
            rts_m.append(((rts(xf, Pf, xp, Pp, A_f) - x) ** 2).mean())
        res['fitted'] = {'kf': float(np.mean(kf_m)), 'rts': float(np.mean(rts_m))}
        res['description'] = 'Kalman filter and RTS smoother MSE on the 1,000 evaluation trajectories (seed 777), produced by code/kalman.py baselines_uncorrected. Oracle Q calibrated on seed 4242, not an evaluation seed.'
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'baselines_uncorrected_timing.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        print('BASE ' + json.dumps(res), flush=True)
    else:
        raise SystemExit('usage: python kalman.py <smoother|baselines_uncorrected> [args]')
