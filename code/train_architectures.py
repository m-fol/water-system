import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata, norm

torch.set_num_threads(2)
import data as D
from data import generate_batch, generate_trajectory
from models import StabilityObserverGNNLSTM, StabilityObserverPointNetLSTM, PointNetEncoder
import diagonal_gain_gnn as V2
from network import STATE_DIM
from train_ten_seed_study import ProbObserver, n_params, pointnet_pm_hidden


class PointNetIDEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24):
        super().__init__()
        self.node_embed = nn.Linear(1 + STATE_DIM, hidden)
        self.point_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.readout = nn.Linear(hidden, out_dim)
        self.register_buffer("onehot", torch.eye(STATE_DIM))

    def forward(self, obs_t):
        b = obs_t.shape[0]
        ids = self.onehot.unsqueeze(0).expand(b, -1, -1)
        h = self.node_embed(torch.cat([obs_t.unsqueeze(-1), ids], dim=-1))
        h = self.point_mlp(h)
        pooled, _ = h.max(dim=1)
        return self.readout(pooled)


class HuberWrap:
    pass


def make(config):
    if config == "gnn" or config == "huber" or config == "prob":
        return ProbObserver() if config == "prob" else StabilityObserverGNNLSTM()
    if config in ("diag_hom", "diag_het", "dense_het"):
        return V2.make_observer(config.startswith("diag"))
    if config == "pointnet":
        return StabilityObserverPointNetLSTM()
    if config == "pointnet_pm":
        target = n_params(StabilityObserverGNNLSTM())
        h, _ = pointnet_pm_hidden(target)
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetEncoder(hidden=h, out_dim=24)
        return m
    if config == "pointnet_id":
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetIDEncoder(hidden=16, out_dim=24)
        return m
    if config == "pointnetpp":
        from models import StabilityObserverPointNetPPLSTM
        return StabilityObserverPointNetPPLSTM()
    if config in ("gnndeep", "randgraph"):
        from train_stability_surrogate import make_observer
        return make_observer(config)
    raise ValueError(config)


def train(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config in ("diag_het", "dense_het"):
        V2.set_pipe_conductance(V2.hetero_K())
    net = make(config)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
        o = torch.tensor(o, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        if config == "prob":
            mu, logsd = net(o)
            loss = (0.5 * ((x - mu) / logsd.exp()) ** 2 + logsd).mean()
        elif config == "huber":
            xh, _ = net(o)
            loss = nn.functional.huber_loss(xh, x, delta=0.5)
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


def eval_set(net, config, seed, n=1000):
    rng = np.random.default_rng(seed)
    O, X, M, active = [], [], [], []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o); X.append(x); M.append(m)
        act = np.zeros(64, bool)
        for (t0, t1, kind, node) in ev:
            act[t0:t1] = True
        active.append(act)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X); M = np.stack(M); active = np.stack(active)
    with torch.no_grad():
        if config == "prob":
            mu, logsd = net(O)
            mu = mu.numpy(); sd = logsd.exp().numpy()
        else:
            mu = net(O)[0].numpy(); sd = None
    E = mu - X
    out = {"seed_eval": seed,
           "mse": float((E ** 2).mean()),
           "mse_tank": float((E[..., :2] ** 2).mean()),
           "mse_junction": float((E[..., 2:] ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        out["coverage"] = {str(l): float((z <= norm.ppf(0.5 + l / 2)).mean()) for l in levels}
        out["coverage95_by_state"] = [float((z[..., i] <= 1.96).mean()) for i in range(STATE_DIM)]
        out["coverage95_event"] = float((z[active] <= 1.96).mean())
        out["coverage95_calm"] = float((z[~active] <= 1.96).mean())
        out["nll"] = float((0.5 * (E / sd) ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)).mean())
        obs_res = np.zeros(active.shape)
        tank_mask = M[..., :2]
        for t_ in range(2):
            pass
        ob = np.abs(O.numpy()[..., :2] - mu[..., :2])
        cnt = tank_mask.sum(axis=2)
        obs_res = np.where(cnt > 0, (ob * tank_mask).sum(axis=2) / np.maximum(cnt, 1), 0.0)
        out["auc_observed_residual"] = auc(obs_res, active)
        out["auc_oracle_residual"] = auc(np.abs(E).mean(axis=2), active)
        out["auc_predicted_sd"] = auc(sd.mean(axis=2), active)
        out["event_frac"] = float(active.mean())
    return out


def evaluate(net, config):
    return [eval_set(net, config, s) for s in (777, 888, 999)]


if __name__ == "__main__":
    config, seed = sys.argv[1], int(sys.argv[2])
    net = train(config, seed)
    sets = evaluate(net, config)
    res = {"config": config, "seed": seed, "params": n_params(net), "eval_sets": sets}
    print("RESULT " + json.dumps(res), flush=True)
