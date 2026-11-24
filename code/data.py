import numpy as np

from network import STATE_NODES, STATE_INDEX, STATE_DIM, STATE_MASK, state_adjacency_matrix

TANK_IDX = [STATE_INDEX["T1"], STATE_INDEX["T2"]]
JUNCTION_IDX = [STATE_INDEX[n] for n in ("J1", "J2", "J3", "J4")]

FLOW_GAIN = 0.35
PUMP_GAIN = 0.6
TANK_SETPOINT = 5.0
BASE_DEMAND = 0.4
MEASUREMENT_NOISE_STD = 0.03


def _adjacency_no_self_loops():
    A = state_adjacency_matrix(add_self_loops=False)
    return A


ADJ = _adjacency_no_self_loops()


def _diurnal_factor(t, period=48):
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / period)


def _flow_term(x, adj, nonlinear, gain):
    dh = x[None, :] - x[:, None]
    if nonlinear:
        edge_flow = np.sign(dh) * np.sqrt(np.abs(dh) + 1e-6)
    else:
        edge_flow = dh
    return gain * (adj * edge_flow).sum(axis=1)


class Event:
    NONE, DROPOUT, PUMP_TRIP, DEMAND_SPIKE = "none", "dropout", "pump_trip", "demand_spike"


def generate_trajectory(T=96, dt=0.5, noise_level=0.05, p_event=0.03,
                         event_len=6, rng=None, dynamics="linear",
                         state_dim=None, adj=None, tank_idx=None, junction_idx=None,
                         tank_names=None, junction_names=None, state_mask=None):
    rng = rng or np.random.default_rng()
    nonlinear = (dynamics == "nonlinear")

    state_dim = STATE_DIM if state_dim is None else state_dim
    adj = ADJ if adj is None else adj
    tank_idx = TANK_IDX if tank_idx is None else tank_idx
    junction_idx = JUNCTION_IDX if junction_idx is None else junction_idx
    tank_names = ["T1", "T2"] if tank_names is None else tank_names
    junction_names = ["J1", "J2", "J3", "J4"] if junction_names is None else junction_names
    state_mask = STATE_MASK if state_mask is None else state_mask

    x = np.zeros(state_dim, dtype=np.float32)
    x[tank_idx] = TANK_SETPOINT
    x[junction_idx] = TANK_SETPOINT - 1.0

    true_state = np.zeros((T, state_dim), dtype=np.float32)
    obs = np.zeros((T, state_dim), dtype=np.float32)
    obs_mask = np.tile(state_mask, (T, 1)).copy()

    events = []
    active_event = None

    for t in range(T):
        if active_event is not None and t >= active_event[0]:
            active_event = None

        if active_event is None and rng.random() < p_event:
            kind = rng.choice([Event.DROPOUT, Event.PUMP_TRIP, Event.DEMAND_SPIKE])
            if kind in (Event.DROPOUT, Event.PUMP_TRIP):
                node = rng.choice(tank_names)
            else:
                node = rng.choice(junction_names)
            active_event = (t + event_len, kind, node)
            events.append([t, t + event_len, kind, node])

        flow_net = _flow_term(x, adj, nonlinear, FLOW_GAIN)

        pump_inflow = np.zeros(state_dim, dtype=np.float32)
        for idx, name in zip(tank_idx, tank_names):
            gain = PUMP_GAIN
            if active_event is not None and active_event[1] == Event.PUMP_TRIP and active_event[2] == name:
                gain = 0.0
            pump_inflow[idx] = gain * (TANK_SETPOINT - x[idx])

        demand = np.zeros(state_dim, dtype=np.float32)
        for idx, name in zip(junction_idx, junction_names):
            d = BASE_DEMAND * _diurnal_factor(t) + rng.normal(0, noise_level)
            if active_event is not None and active_event[1] == Event.DEMAND_SPIKE and active_event[2] == name:
                d *= 4.0
            demand[idx] = max(d, 0.0)

        dx = flow_net + pump_inflow - demand
        x = x + dt * dx
        true_state[t] = x

        o = np.zeros(state_dim, dtype=np.float32)
        mask_t = state_mask.copy()
        for idx, name in zip(tank_idx, tank_names):
            if active_event is not None and active_event[1] == Event.DROPOUT and active_event[2] == name:
                mask_t[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, MEASUREMENT_NOISE_STD)
        obs[t] = o
        obs_mask[t] = mask_t

    return obs, obs_mask, true_state, events


def generate_batch(batch_size=16, T=96, rng=None, **kwargs):
    rng = rng or np.random.default_rng()
    obs_b, mask_b, x_b = [], [], []
    for _ in range(batch_size):
        obs, mask, x, _ = generate_trajectory(T=T, rng=rng, **kwargs)
        obs_b.append(obs)
        mask_b.append(mask)
        x_b.append(x)
    return (np.stack(obs_b), np.stack(mask_b), np.stack(x_b))


if __name__ == "__main__":
    obs, mask, x, events = generate_trajectory(T=48, rng=np.random.default_rng(0))
    print("obs shape:", obs.shape, "true_state shape:", x.shape)
    print("state nodes:", STATE_NODES)
    print("events injected:", events)
    print("tank T1 trajectory (first 10 steps):", np.round(x[:10, STATE_INDEX["T1"]], 3))
