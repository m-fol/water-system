import sys, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from benchmark import generate_batch, generate_trajectory
from neural_models import ARCHITECTURES, GraphConv, StabilityObserverGNNLSTM, _normalized_adjacency
from benchmark import STATE_DIM
import neural_models as CV
import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
import benchmark as D
from neural_models import StabilityObserverGNNLSTM, StabilityObserverPointNetLSTM, PointNetEncoder
import neural_models as V2
from scipy.stats import rankdata, norm
from benchmark import generate_trajectory, generate_batch
from neural_models import StabilityObserverGNNLSTM
import sys
torch.set_num_threads(2)
ALPHA = 0.3
C_FLOOR = 4 * (0.5 * 0.05) ** 2
TAU = 0.5

class DeepGNNEncoder(nn.Module):

    def __init__(self, layers=4, hidden=16, out_dim=24):
        super().__init__()
        A = _normalized_adjacency()
        self.state_dim = STATE_DIM
        self.node_embed = nn.Linear(1, hidden)
        self.gcs = nn.ModuleList([GraphConv(hidden, hidden, A) for _ in range(layers)])
        self.readout = nn.Linear(hidden * STATE_DIM, out_dim)
        self.act = nn.ReLU()

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        for gc in self.gcs:
            h = self.act(gc(h))
        return self.readout(h.reshape(h.size(0), -1))

def make_observer(arch):
    if arch == 'gnn':
        return ARCHITECTURES['gnn']()
    if arch == 'pointnet':
        return ARCHITECTURES['pointnet']()
    if arch == 'pointnetpp':
        return ARCHITECTURES['pointnetpp']()
    if arch == 'gnndeep':
        obs = StabilityObserverGNNLSTM()
        obs.encoder = DeepGNNEncoder()
        return obs
    if arch == 'randgraph':
        rng = np.random.default_rng(2024)
        A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)
        pairs = [(i, j) for i in range(STATE_DIM) for j in range(i + 1, STATE_DIM)]
        for k in rng.choice(len(pairs), 6, replace=False):
            i, j = pairs[k]
            A[i, j] = A[j, i] = 1.0
        A += np.eye(STATE_DIM, dtype=np.float32)
        return StabilityObserverGNNLSTM(adj=torch.tensor(A))
    raise ValueError(arch)

def train(arch, cand, lam, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = make_observer(arch)
    params = list(obs.parameters())
    lyap = CV.CANDIDATES[cand]() if cand != 'none' else None
    if lyap is not None:
        params += list(lyap.parameters())
    opt = optim.Adam(params, lr=0.001)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        xh, _ = obs(o)
        loss = nn.functional.mse_loss(xh, x)
        if lyap is not None and lam > 0:
            V = lyap((xh - x).reshape(-1, STATE_DIM)).reshape(B, T)
            loss = loss + lam * torch.relu(V[:, 1:] - (1 - ALPHA) * V[:, :-1] - C_FLOOR).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (obs, lyap)

def evaluate(obs, lyap, n=1000):
    rng = np.random.default_rng(777)
    obs_l, x_l, ev_l = ([], [], [])
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        obs_l.append(o)
        x_l.append(x)
        ev_l.append(ev)
    O = torch.tensor(np.stack(obs_l), dtype=torch.float32)
    X = np.stack(x_l)
    with torch.no_grad():
        xh, _ = obs(O)
    E = xh.numpy() - X
    per_traj = (E ** 2).mean(axis=(1, 2))
    out = {'mse': float(per_traj.mean()), 'tail_p95': float(np.percentile(per_traj, 95)), 'tail_p99': float(np.percentile(per_traj, 99))}
    rec, censored = ([], 0)
    for i in range(n):
        for t0, t1, kind, node in ev_l[i]:
            norms = np.linalg.norm(E[i, t1:], axis=1)
            idx = np.where(norms < TAU)[0]
            if len(idx):
                rec.append(int(idx[0]))
            else:
                censored += 1
    out['recovery_median'] = float(np.median(rec)) if rec else None
    out['recovery_p90'] = float(np.percentile(rec, 90)) if rec else None
    out['recovery_censored'] = censored
    out['n_events'] = int(sum((len(e) for e in ev_l)))
    if lyap is not None:
        with torch.enable_grad():
            Ve = lyap(torch.tensor(E.reshape(-1, STATE_DIM), dtype=torch.float32)).reshape(n, -1).detach().numpy()
            outside = lyap.outside(torch.tensor(E.reshape(-1, STATE_DIM), dtype=torch.float32)).reshape(n, -1).numpy() if hasattr(lyap, 'outside') else np.zeros((n, 64), bool)
        viol = Ve[:, 1:] > (1 - ALPHA) * Ve[:, :-1] + C_FLOOR
        out['onestep_violation_rate'] = float(viol.mean())
        out['traj_zero_violation_rate'] = float((~viol.any(axis=1)).mean())
        out['domain_exit_rate'] = float(outside.mean())
    return out

class ProbObserver(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = StabilityObserverGNNLSTM()
        self.head_logsd = nn.Linear(32, STATE_DIM)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.base.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hs, _ = self.base.lstm(spatial)
        return (self.base.readout(hs), self.head_logsd(hs))

def n_params(m):
    return sum((p.numel() for p in m.parameters()))

def pointnet_pm_hidden(target):
    best = None
    for h in range(8, 64):
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetEncoder(hidden=h, out_dim=24)
        c = abs(n_params(m) - target)
        if best is None or c < best[0]:
            best = (c, h, n_params(m))
    return (best[1], best[2])

def make(config):
    if config == 'gnn':
        return StabilityObserverGNNLSTM()
    if config in ('diag_hom', 'diag_het', 'dense_het'):
        return V2.make_observer(config.startswith('diag'))
    if config == 'pointnet':
        return StabilityObserverPointNetLSTM()
    if config == 'pointnet_pm':
        target = n_params(StabilityObserverGNNLSTM())
        h, _ = pointnet_pm_hidden(target)
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetEncoder(hidden=h, out_dim=24)
        return m
    if config == 'pointnetpp':
        from models import StabilityObserverPointNetPPLSTM
        return StabilityObserverPointNetPPLSTM()
    if config == 'prob':
        return ProbObserver()
    if config in ('gnndeep', 'randgraph'):
        from train_stability_surrogate import make_observer
        return make_observer(config)
    raise ValueError(config)

def train_ten_seed_study(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config == 'diag_het' or config == 'dense_het':
        V2.set_pipe_conductance(V2.hetero_K())
    net = make(config)
    opt = optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        if config == 'prob':
            mu, logsd = net(o)
            loss = (0.5 * ((x - mu) / logsd.exp()) ** 2 + logsd).mean()
        else:
            xh, _ = net(o)
            loss = nn.functional.mse_loss(xh, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net

def auc(scores, labels):
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, bool).ravel()
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = labels.size - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def evaluate_ten_seed_study(net, config, n=1000):
    rng = np.random.default_rng(777)
    O, X, active = ([], [], [])
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o)
        X.append(x)
        act = np.zeros(64, bool)
        for t0, t1, kind, node in ev:
            act[t0:t1] = True
        active.append(act)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X)
    active = np.stack(active)
    with torch.no_grad():
        if config == 'prob':
            mu, logsd = net(O)
            mu = mu.numpy()
            sd = logsd.exp().numpy()
        else:
            mu = net(O)[0].numpy()
            sd = None
    E = mu - X
    out = {'mse': float((E ** 2).mean()), 'mse_tank': float((E[..., :2] ** 2).mean()), 'mse_junction': float((E[..., 2:] ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        from scipy.stats import norm
        levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        out['coverage'] = {str(l): float((z <= norm.ppf(0.5 + l / 2)).mean()) for l in levels}
        out['coverage95_tank'] = float((z[..., :2] <= 1.96).mean())
        out['coverage95_junction'] = float((z[..., 2:] <= 1.96).mean())
        lab = np.repeat(active[:, :, None], 6, axis=2)
        out['auc_sd'] = auc(sd.mean(axis=2), active)
        out['auc_absresid'] = auc(np.abs(E).mean(axis=2), active)
        out['event_frac'] = float(active.mean())
    return out

class PointNetIDEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24):
        super().__init__()
        self.node_embed = nn.Linear(1 + STATE_DIM, hidden)
        self.point_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Linear(hidden, out_dim)
        self.register_buffer('onehot', torch.eye(STATE_DIM))

    def forward(self, obs_t):
        b = obs_t.shape[0]
        ids = self.onehot.unsqueeze(0).expand(b, -1, -1)
        h = self.node_embed(torch.cat([obs_t.unsqueeze(-1), ids], dim=-1))
        h = self.point_mlp(h)
        pooled, _ = h.max(dim=1)
        return self.readout(pooled)

class HuberWrap:
    pass

def make_architectures(config):
    if config == 'gnn' or config == 'huber' or config == 'prob':
        return ProbObserver() if config == 'prob' else StabilityObserverGNNLSTM()
    if config in ('diag_hom', 'diag_het', 'dense_het'):
        return V2.make_observer(config.startswith('diag'))
    if config == 'pointnet':
        return StabilityObserverPointNetLSTM()
    if config == 'pointnet_pm':
        target = n_params(StabilityObserverGNNLSTM())
        h, _ = pointnet_pm_hidden(target)
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetEncoder(hidden=h, out_dim=24)
        return m
    if config == 'pointnet_id':
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetIDEncoder(hidden=16, out_dim=24)
        return m
    if config == 'pointnetpp':
        from models import StabilityObserverPointNetPPLSTM
        return StabilityObserverPointNetPPLSTM()
    if config in ('gnndeep', 'randgraph'):
        from train_stability_surrogate import make_observer
        return make_observer(config)
    raise ValueError(config)

def train_architectures(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config in ('diag_het', 'dense_het'):
        V2.set_pipe_conductance(V2.hetero_K())
    net = make_architectures(config)
    opt = optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        if config == 'prob':
            mu, logsd = net(o)
            loss = (0.5 * ((x - mu) / logsd.exp()) ** 2 + logsd).mean()
        elif config == 'huber':
            xh, _ = net(o)
            loss = nn.functional.huber_loss(xh, x, delta=0.5)
        else:
            xh, _ = net(o)
            loss = nn.functional.mse_loss(xh, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net

def eval_set(net, config, seed, n=1000):
    rng = np.random.default_rng(seed)
    O, X, M, active = ([], [], [], [])
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o)
        X.append(x)
        M.append(m)
        act = np.zeros(64, bool)
        for t0, t1, kind, node in ev:
            act[t0:t1] = True
        active.append(act)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X)
    M = np.stack(M)
    active = np.stack(active)
    with torch.no_grad():
        if config == 'prob':
            mu, logsd = net(O)
            mu = mu.numpy()
            sd = logsd.exp().numpy()
        else:
            mu = net(O)[0].numpy()
            sd = None
    E = mu - X
    out = {'seed_eval': seed, 'mse': float((E ** 2).mean()), 'mse_tank': float((E[..., :2] ** 2).mean()), 'mse_junction': float((E[..., 2:] ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        out['coverage'] = {str(l): float((z <= norm.ppf(0.5 + l / 2)).mean()) for l in levels}
        out['coverage95_by_state'] = [float((z[..., i] <= 1.96).mean()) for i in range(STATE_DIM)]
        out['coverage95_event'] = float((z[active] <= 1.96).mean())
        out['coverage95_calm'] = float((z[~active] <= 1.96).mean())
        out['nll'] = float((0.5 * (E / sd) ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)).mean())
        obs_res = np.zeros(active.shape)
        tank_mask = M[..., :2]
        for t_ in range(2):
            pass
        ob = np.abs(O.numpy()[..., :2] - mu[..., :2])
        cnt = tank_mask.sum(axis=2)
        obs_res = np.where(cnt > 0, (ob * tank_mask).sum(axis=2) / np.maximum(cnt, 1), 0.0)
        out['auc_observed_residual'] = auc(obs_res, active)
        out['auc_oracle_residual'] = auc(np.abs(E).mean(axis=2), active)
        out['auc_predicted_sd'] = auc(sd.mean(axis=2), active)
        out['event_frac'] = float(active.mean())
    return out

def evaluate_architectures(net, config):
    return [eval_set(net, config, s) for s in (777, 888, 999)]
make_architecture = make_architectures
V_ANTI = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)

def auc_architecture_controls(scores, labels):
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, bool).ravel()
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = labels.size - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def train_architecture_controls(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config in ('diag_het', 'dense_het'):
        V2.set_pipe_conductance(V2.hetero_K())
    if config == 'wmse_twostage':
        stage1 = train_architecture_controls('prob', seed, epochs=epochs, B=B, T=T)
        stage1.eval()
        net = StabilityObserverGNNLSTM()
        opt = optim.Adam(net.parameters(), lr=0.001)
        for _ in range(epochs):
            o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
            o = torch.tensor(o, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                _, logsd = stage1(o)
                w = torch.exp(-2 * logsd)
            xh, _ = net(o)
            loss = (w * (xh - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net
    if config == 'prob_isolated':
        net = ProbObserver()
        head = net.head_logsd
        opt = optim.Adam(net.parameters(), lr=0.001)
        for _ in range(epochs):
            o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
            o = torch.tensor(o, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            spatial = torch.stack([net.base.encoder(o[:, t]) for t in range(o.shape[1])], dim=1)
            hs, _ = net.base.lstm(spatial)
            mu = net.base.readout(hs)
            logsd_enc = head(hs.detach())
            w = torch.exp(-2 * logsd_enc.detach())
            loss_mu = (w * (x - mu) ** 2).mean()
            loss_sig = (logsd_enc + 0.5 * ((x - mu.detach()) / logsd_enc.exp()) ** 2).mean()
            loss = loss_mu + loss_sig
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net
    net = make_architecture(config)
    opt = optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        if config == 'prob':
            mu, logsd = net(o)
            loss = (0.5 * ((x - mu) / logsd.exp()) ** 2 + logsd).mean()
        else:
            xh, _ = net(o)
            loss = nn.functional.mse_loss(xh, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net

def eval_set_architecture_controls(net, config, seed, n=1000):
    rng = np.random.default_rng(seed)
    O, X, M, kinds = ([], [], [], [])
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o)
        X.append(x)
        M.append(m)
        k = np.full(64, 'calm', dtype=object)
        for t0, t1, kind, node in ev:
            k[t0:t1] = kind
        kinds.append(k)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X)
    M = np.stack(M)
    kinds = np.stack(kinds)
    isprob = config in ('prob', 'prob_isolated')
    with torch.no_grad():
        if isprob:
            if config == 'prob_isolated':
                spatial = torch.stack([net.base.encoder(O[:, t]) for t in range(O.shape[1])], dim=1)
                hs, _ = net.base.lstm(spatial)
                mu = net.base.readout(hs).numpy()
                sd = net.head_logsd(hs).exp().numpy()
            else:
                mu_t, logsd_t = net(O)
                mu = mu_t.numpy()
                sd = logsd_t.exp().numpy()
        else:
            out = net(O)
            mu = (out[0] if isinstance(out, tuple) else out).numpy()
            sd = None
    E = mu - X
    res = {'seed_eval': seed, 'mse': float((E ** 2).mean()), 'mse_tank': float((E[..., :2] ** 2).mean()), 'mse_junction': float((E[..., 2:] ** 2).mean()), 'mse_antisym': float(((E @ V_ANTI) ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        active = kinds != 'calm'
        res['coverage95'] = float((z <= 1.96).mean())
        res['coverage95_event'] = float((z[active] <= 1.96).mean())
        res['coverage95_calm'] = float((z[~active] <= 1.96).mean())
        ob = np.abs(O.numpy()[..., :2] - mu[..., :2])
        cnt = M[..., :2].sum(axis=2)
        obs_res = np.where(cnt > 0, (ob * M[..., :2]).sum(axis=2) / np.maximum(cnt, 1), 0.0)
        res['auc_observed_residual'] = auc_architecture_controls(obs_res, active)
        res['auc_predicted_sd'] = auc_architecture_controls(sd.mean(axis=2), active)
        res['auc_oracle_residual'] = auc_architecture_controls(np.abs(E).mean(axis=2), active)
        for kind in ('dropout', 'pump_trip', 'demand_spike'):
            sel = (kinds == kind) | (kinds == 'calm')
            lab = (kinds == kind)[sel]
            res[f'auc_observed_{kind}'] = auc_architecture_controls(obs_res[sel], lab)
            res[f'auc_sd_{kind}'] = auc_architecture_controls(sd.mean(axis=2)[sel], lab)
    return res
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'stability_surrogate':
        arch, cand, lam, seed = (sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
        obs, lyap = train(arch, cand, lam, seed)
        res = evaluate(obs, lyap)
        res.update(arch=arch, cand=cand, lam=lam, seed=seed)
        print('RESULT ' + json.dumps(res), flush=True)
    elif _cmd == 'ten_seed_study':
        config, seed = (sys.argv[1], int(sys.argv[2]))
        net = train_ten_seed_study(config, seed)
        res = evaluate_ten_seed_study(net, config)
        res.update(config=config, seed=seed, params=n_params(net))
        print('RESULT ' + json.dumps(res), flush=True)
    elif _cmd == 'architectures':
        config, seed = (sys.argv[1], int(sys.argv[2]))
        net = train_architectures(config, seed)
        sets = evaluate_architectures(net, config)
        res = {'config': config, 'seed': seed, 'params': n_params(net), 'eval_sets': sets}
        print('RESULT ' + json.dumps(res), flush=True)
    elif _cmd == 'architecture_controls':
        config, seed = (sys.argv[1], int(sys.argv[2]))
        net = train_architecture_controls(config, seed)
        sets = [eval_set_architecture_controls(net, config, s) for s in (777, 888, 999)]
        print('RESULT ' + json.dumps({'config': config, 'seed': seed, 'params': n_params(net), 'eval_sets': sets}), flush=True)
    else:
        raise SystemExit('usage: python training.py <stability_surrogate|ten_seed_study|architectures|architecture_controls> [args]')
