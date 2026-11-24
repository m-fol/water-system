import torch
import torch.nn as nn

from network import STATE_DIM, state_adjacency_matrix


def _normalized_adjacency(adj=None):
    A = torch.tensor(state_adjacency_matrix(add_self_loops=True)) if adj is None else torch.as_tensor(adj)
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, adj_norm):
        super().__init__()
        self.register_buffer("A", adj_norm)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        h = torch.einsum("ij,bjf->bif", self.A, h)
        return self.lin(h)


class GNNEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24, adj=None, state_dim=None):
        super().__init__()
        A = _normalized_adjacency(adj)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM
        self.node_embed = nn.Linear(1, hidden)
        self.gc1 = GraphConv(hidden, hidden, A)
        self.gc2 = GraphConv(hidden, hidden, A)
        self.readout = nn.Linear(hidden * self.state_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        h = self.act(self.gc1(h))
        h = self.act(self.gc2(h))
        return self.readout(h.reshape(h.size(0), -1))


class PointNetEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24):
        super().__init__()
        self.node_embed = nn.Linear(1, hidden)
        self.point_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, obs_t):
        h = self.node_embed(obs_t.unsqueeze(-1))
        h = self.point_mlp(h)
        pooled, _ = h.max(dim=1)
        return self.readout(pooled)


class PointNetPlusPlusEncoder(nn.Module):

    def __init__(self, hidden=16, out_dim=24, adj=None, state_dim=None):
        super().__init__()
        if adj is None:
            adj = torch.tensor(state_adjacency_matrix(add_self_loops=True))
        else:
            adj = torch.as_tensor(adj)
        self.register_buffer("group_mask", adj > 0)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM

        self.node_embed = nn.Linear(1, hidden)
        self.local_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.global_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, obs_t):
        batch = obs_t.size(0)
        h = self.node_embed(obs_t.unsqueeze(-1))
        h_local = self.local_mlp(h)

        n = h_local.size(1)
        expanded = h_local.unsqueeze(1).expand(batch, n, n, -1)
        mask = self.group_mask.unsqueeze(0).unsqueeze(-1)
        masked = expanded.masked_fill(~mask, float("-inf"))
        local_pooled, _ = masked.max(dim=2)

        h_global = self.global_mlp(local_pooled)
        pooled, _ = h_global.max(dim=1)
        return self.readout(pooled)


class StabilityObserverGNNLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, adj=None, state_dim=None):
        super().__init__()
        self.encoder = GNNEncoder(out_dim=spatial_dim, adj=adj, state_dim=state_dim)
        self.state_dim = self.encoder.state_dim
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return x_hat, hidden_seq


class StabilityObserverPointNetLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, state_dim=None):
        super().__init__()
        self.encoder = PointNetEncoder(out_dim=spatial_dim)
        self.state_dim = state_dim if state_dim is not None else STATE_DIM
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return x_hat, hidden_seq


class StabilityObserverPointNetPPLSTM(nn.Module):

    def __init__(self, spatial_dim=24, hidden_dim=32, adj=None, state_dim=None):
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(out_dim=spatial_dim, adj=adj, state_dim=state_dim)
        self.state_dim = self.encoder.state_dim
        self.lstm = nn.LSTM(input_size=spatial_dim, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, obs_seq):
        batch, T, _ = obs_seq.shape
        spatial = torch.stack([self.encoder(obs_seq[:, t]) for t in range(T)], dim=1)
        hidden_seq, _ = self.lstm(spatial)
        x_hat = self.readout(hidden_seq)
        return x_hat, hidden_seq


ARCHITECTURES = {
    "gnn": StabilityObserverGNNLSTM,
    "pointnet": StabilityObserverPointNetLSTM,
    "pointnetpp": StabilityObserverPointNetPPLSTM,
}


if __name__ == "__main__":
    for name, cls in ARCHITECTURES.items():
        model = cls()
        dummy = torch.randn(4, 10, STATE_DIM)
        x_hat, hidden_seq = model(dummy)
        print(f"{name}: x_hat={tuple(x_hat.shape)}  hidden_seq={tuple(hidden_seq.shape)}  "
              f"params={sum(p.numel() for p in model.parameters())}")
