import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata

torch.set_num_threads(2)
import data as D
from data import generate_batch, generate_trajectory
from models import StabilityObserverGNNLSTM, StabilityObserverPointNetLSTM, PointNetEncoder
import diag_k as V2
from network import STATE_DIM


class ProbObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = StabilityObserverGNNLSTM()
        self.head_logsd = nn.Linear(32, STATE_DIM)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.base.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hs, _ = self.base.lstm(spatial)
        return self.base.readout(hs), self.head_logsd(hs)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def pointnet_pm_hidden(target):
    best = None
    for h in range(8, 64):
        m = StabilityObserverPointNetLSTM()
        m.encoder = PointNetEncoder(hidden=h, out_dim=24)
        c = abs(n_params(m) - target)
        if best is None or c < best[0]:
            best = (c, h, n_params(m))
    return best[1], best[2]


def make(config):
    if config == "gnn":
        return StabilityObserverGNNLSTM()
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
    if config == "pointnetpp":
        from models import StabilityObserverPointNetPPLSTM
        return StabilityObserverPointNetPPLSTM()
    if config == "prob":
        return ProbObserver()
    if config in ("gnndeep", "randgraph"):
        from train_v3 import make_observer
        return make_observer(config)
    raise ValueError(config)


def train(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config == "diag_het" or config == "dense_het":
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


def evaluate(net, config, n=1000):
    rng = np.random.default_rng(777)
    O, X, active = [], [], []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o)
        X.append(x)
        act = np.zeros(64, bool)
        for (t0, t1, kind, node) in ev:
            act[t0:t1] = True
        active.append(act)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X)
    active = np.stack(active)
    with torch.no_grad():
        if config == "prob":
            mu, logsd = net(O)
            mu = mu.numpy()
            sd = logsd.exp().numpy()
        else:
            mu = net(O)[0].numpy()
            sd = None
    E = mu - X
    out = {"mse": float((E ** 2).mean()),
           "mse_tank": float((E[..., :2] ** 2).mean()),
           "mse_junction": float((E[..., 2:] ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        from scipy.stats import norm
        levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        out["coverage"] = {str(l): float((z <= norm.ppf(0.5 + l / 2)).mean()) for l in levels}
        out["coverage95_tank"] = float((z[..., :2] <= 1.96).mean())
        out["coverage95_junction"] = float((z[..., 2:] <= 1.96).mean())
        lab = np.repeat(active[:, :, None], 6, axis=2)
        out["auc_sd"] = auc(sd.mean(axis=2), active)
        out["auc_absresid"] = auc(np.abs(E).mean(axis=2), active)
        out["event_frac"] = float(active.mean())
    return out


if __name__ == "__main__":
    config, seed = sys.argv[1], int(sys.argv[2])
    net = train(config, seed)
    res = evaluate(net, config)
    res.update(config=config, seed=seed, params=n_params(net))
    print("RESULT " + json.dumps(res), flush=True)
