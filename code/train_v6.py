import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata, norm

torch.set_num_threads(2)
from data import generate_trajectory, generate_batch
from models import StabilityObserverGNNLSTM
import diag_k as V2
from network import STATE_DIM
from train_v4 import ProbObserver, n_params
from train_v5 import make as make_v5

V_ANTI = np.array([0, 0, 0, 1, 0, -1.0]) / np.sqrt(2)


def auc(scores, labels):
    scores = np.asarray(scores, float).ravel()
    labels = np.asarray(labels, bool).ravel()
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = labels.size - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train(config, seed, epochs=500, B=16, T=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if config in ("diag_het", "dense_het"):
        V2.set_pipe_conductance(V2.hetero_K())

    if config == "wmse_twostage":
        stage1 = train("prob", seed, epochs=epochs, B=B, T=T)
        stage1.eval()
        net = StabilityObserverGNNLSTM()
        opt = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(epochs):
            o, m, x = generate_batch(batch_size=B, T=T, rng=rng, p_event=0.08)
            o = torch.tensor(o, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                _, logsd = stage1(o)
                w = torch.exp(-2 * logsd)
            xh, _ = net(o)
            loss = (w * (xh - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return net

    if config == "prob_isolated":
        net = ProbObserver()
        head = net.head_logsd
        opt = optim.Adam(net.parameters(), lr=1e-3)
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
            opt.zero_grad(); loss.backward(); opt.step()
        return net

    net = make_v5(config)
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
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def eval_set(net, config, seed, n=1000):
    rng = np.random.default_rng(seed)
    O, X, M, kinds = [], [], [], []
    for _ in range(n):
        o, m, x, ev = generate_trajectory(T=64, rng=rng, p_event=0.08)
        O.append(o); X.append(x); M.append(m)
        k = np.full(64, "calm", dtype=object)
        for (t0, t1, kind, node) in ev:
            k[t0:t1] = kind
        kinds.append(k)
    O = torch.tensor(np.stack(O), dtype=torch.float32)
    X = np.stack(X); M = np.stack(M); kinds = np.stack(kinds)
    isprob = config in ("prob", "prob_isolated")
    with torch.no_grad():
        if isprob:
            if config == "prob_isolated":
                spatial = torch.stack([net.base.encoder(O[:, t]) for t in range(O.shape[1])], dim=1)
                hs, _ = net.base.lstm(spatial)
                mu = net.base.readout(hs).numpy(); sd = net.head_logsd(hs).exp().numpy()
            else:
                mu_t, logsd_t = net(O)
                mu = mu_t.numpy(); sd = logsd_t.exp().numpy()
        else:
            out = net(O)
            mu = (out[0] if isinstance(out, tuple) else out).numpy()
            sd = None
    E = mu - X
    res = {"seed_eval": seed,
           "mse": float((E ** 2).mean()),
           "mse_tank": float((E[..., :2] ** 2).mean()),
           "mse_junction": float((E[..., 2:] ** 2).mean()),
           "mse_antisym": float(((E @ V_ANTI) ** 2).mean())}
    if sd is not None:
        z = np.abs(E) / sd
        active = kinds != "calm"
        res["coverage95"] = float((z <= 1.96).mean())
        res["coverage95_event"] = float((z[active] <= 1.96).mean())
        res["coverage95_calm"] = float((z[~active] <= 1.96).mean())
        ob = np.abs(O.numpy()[..., :2] - mu[..., :2])
        cnt = M[..., :2].sum(axis=2)
        obs_res = np.where(cnt > 0, (ob * M[..., :2]).sum(axis=2) / np.maximum(cnt, 1), 0.0)
        res["auc_observed_residual"] = auc(obs_res, active)
        res["auc_predicted_sd"] = auc(sd.mean(axis=2), active)
        res["auc_oracle_residual"] = auc(np.abs(E).mean(axis=2), active)
        for kind in ("dropout", "pump_trip", "demand_spike"):
            sel = (kinds == kind) | (kinds == "calm")
            lab = (kinds == kind)[sel]
            res[f"auc_observed_{kind}"] = auc(obs_res[sel], lab)
            res[f"auc_sd_{kind}"] = auc(sd.mean(axis=2)[sel], lab)
    return res


if __name__ == "__main__":
    config, seed = sys.argv[1], int(sys.argv[2])
    net = train(config, seed)
    sets = [eval_set(net, config, s) for s in (777, 888, 999)]
    print("RESULT " + json.dumps({"config": config, "seed": seed, "params": n_params(net), "eval_sets": sets}), flush=True)
