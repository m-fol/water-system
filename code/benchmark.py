import numpy as np
import json, os
import sys
NODES = ['R1', 'P1', 'T1', 'P3', 'J1', 'J2', 'J3', 'J4', 'P4', 'T2', 'P2', 'R2']
NODE_INDEX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)
NODE_TYPE = {'R1': 'reservoir', 'R2': 'reservoir', 'P1': 'primary_pump', 'P2': 'primary_pump', 'T1': 'tank', 'T2': 'tank', 'P3': 'booster_pump', 'P4': 'booster_pump', 'J1': 'junction', 'J2': 'junction', 'J3': 'junction', 'J4': 'junction'}
INSTRUMENTED = {n: NODE_TYPE[n] != 'junction' for n in NODES}
EDGES = [('R1', 'P1'), ('P1', 'T1'), ('T1', 'P3'), ('P3', 'J1'), ('R2', 'P2'), ('P2', 'T2'), ('T2', 'P4'), ('P4', 'J3'), ('J1', 'J2'), ('J2', 'J3'), ('J3', 'J4'), ('J4', 'J1')]
STATE_NODES = ['T1', 'T2', 'J1', 'J2', 'J3', 'J4']
STATE_INDEX = {n: i for i, n in enumerate(STATE_NODES)}
STATE_DIM = len(STATE_NODES)
STATE_MASK = np.array([INSTRUMENTED[n] for n in STATE_NODES], dtype=bool)

def adjacency_matrix(add_self_loops=True):
    A = np.zeros((N, N), dtype=np.float32)
    for a, b in EDGES:
        i, j = (NODE_INDEX[a], NODE_INDEX[b])
        A[i, j] = 1.0
        A[j, i] = 1.0
    if add_self_loops:
        A += np.eye(N, dtype=np.float32)
    return A

def state_adjacency_matrix(add_self_loops=True):
    reduced_edges = [('T1', 'J1'), ('T2', 'J3'), ('J1', 'J2'), ('J2', 'J3'), ('J3', 'J4'), ('J4', 'J1')]
    A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float32)
    for a, b in reduced_edges:
        i, j = (STATE_INDEX[a], STATE_INDEX[b])
        A[i, j] = 1.0
        A[j, i] = 1.0
    if add_self_loops:
        A += np.eye(STATE_DIM, dtype=np.float32)
    return A

def has_cycle(edges=None, nodes=None):
    edges = edges or EDGES
    nodes = nodes or NODES
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = (find(a), find(b))
        if ra == rb:
            return True
        parent[ra] = rb
    return False
TANK_IDX = [STATE_INDEX['T1'], STATE_INDEX['T2']]
JUNCTION_IDX = [STATE_INDEX[n_data] for n_data in ('J1', 'J2', 'J3', 'J4')]
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
        edge_flow = np.sign(dh) * np.sqrt(np.abs(dh) + 1e-06)
    else:
        edge_flow = dh
    return gain * (adj * edge_flow).sum(axis=1)

class Event:
    NONE, DROPOUT, PUMP_TRIP, DEMAND_SPIKE = ('none', 'dropout', 'pump_trip', 'demand_spike')

def generate_trajectory(T=96, dt=0.5, noise_level=0.05, p_event=0.03, event_len=6, rng=None, dynamics='linear', state_dim=None, adj=None, tank_idx=None, junction_idx=None, tank_names=None, junction_names=None, state_mask=None):
    rng = rng or np.random.default_rng()
    nonlinear = dynamics == 'nonlinear'
    state_dim = STATE_DIM if state_dim is None else state_dim
    adj = ADJ if adj is None else adj
    tank_idx = TANK_IDX if tank_idx is None else tank_idx
    junction_idx = JUNCTION_IDX if junction_idx is None else junction_idx
    tank_names = ['T1', 'T2'] if tank_names is None else tank_names
    junction_names = ['J1', 'J2', 'J3', 'J4'] if junction_names is None else junction_names
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
            if active_event is not None and active_event[1] == Event.PUMP_TRIP and (active_event[2] == name):
                gain = 0.0
            pump_inflow[idx] = gain * (TANK_SETPOINT - x[idx])
        demand = np.zeros(state_dim, dtype=np.float32)
        for idx, name in zip(junction_idx, junction_names):
            d = BASE_DEMAND * _diurnal_factor(t) + rng.normal(0, noise_level)
            if active_event is not None and active_event[1] == Event.DEMAND_SPIKE and (active_event[2] == name):
                d *= 4.0
            demand[idx] = max(d, 0.0)
        dx = flow_net + pump_inflow - demand
        x = x + dt * dx
        true_state[t] = x
        o = np.zeros(state_dim, dtype=np.float32)
        mask_t = state_mask.copy()
        for idx, name in zip(tank_idx, tank_names):
            if active_event is not None and active_event[1] == Event.DROPOUT and (active_event[2] == name):
                mask_t[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, MEASUREMENT_NOISE_STD)
        obs[t] = o
        obs_mask[t] = mask_t
    return (obs, obs_mask, true_state, events)

def generate_batch(batch_size=16, T=96, rng=None, **kwargs):
    rng = rng or np.random.default_rng()
    obs_b, mask_b, x_b = ([], [], [])
    for _ in range(batch_size):
        obs, mask, x, _ = generate_trajectory(T=T, rng=rng, **kwargs)
        obs_b.append(obs)
        mask_b.append(mask)
        x_b.append(x)
    return (np.stack(obs_b), np.stack(mask_b), np.stack(x_b))
TANKS = ['T1', 'T2']
JUNCTIONS = ['J1', 'J2', 'J3', 'J4']

def diurnal(t, phase=0.0, period=48):
    return 1.0 + 0.5 * np.sin(2 * np.pi * (t + phase) / period)

def simulate(T=64, rng=None, p_event=0.08, event_len=6, noise_level=0.05, random=False, init_rng=None, return_nodes=False):
    rng = rng or np.random.default_rng()
    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[TANK_IDX] = TANK_SETPOINT
    x[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    phase = 0.0
    if random:
        init_rng = init_rng or np.random.default_rng()
        x[TANK_IDX] = init_rng.normal(TANK_SETPOINT, 0.5, len(TANK_IDX))
        x[JUNCTION_IDX] = init_rng.normal(TANK_SETPOINT - 1.0, 0.5, len(JUNCTION_IDX))
        phase = float(init_rng.uniform(0, 48))
    x0 = x.copy()
    state = np.zeros((T, STATE_DIM), dtype=np.float32)
    obs = np.zeros((T, STATE_DIM), dtype=np.float32)
    mask = np.tile(STATE_MASK, (T, 1)).copy()
    kinds = np.full(T, 'calm', dtype=object)
    nodes = np.full(T, '', dtype=object)
    active = None
    for t in range(T):
        if active is not None and t >= active[0]:
            active = None
        if active is None and rng.random() < p_event:
            kind = rng.choice(['dropout', 'pump_trip', 'demand_spike'])
            node = rng.choice(TANKS) if kind in ('dropout', 'pump_trip') else rng.choice(JUNCTIONS)
            active = (t + event_len, kind, node)
        if active is not None:
            kinds[t] = active[1]
            nodes[t] = active[2]
        dh = x[None, :] - x[:, None]
        flow = FLOW_GAIN * (ADJ * dh).sum(axis=1)
        pump = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(TANK_IDX, TANKS):
            g = PUMP_GAIN
            if active is not None and active[1] == 'pump_trip' and (active[2] == name):
                g = 0.0
            pump[idx] = g * (TANK_SETPOINT - x[idx])
        demand = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(JUNCTION_IDX, JUNCTIONS):
            d = BASE_DEMAND * diurnal(t, phase) + rng.normal(0, noise_level)
            if active is not None and active[1] == 'demand_spike' and (active[2] == name):
                d *= 4.0
            demand[idx] = max(d, 0.0)
        x = x + 0.5 * (flow + pump - demand)
        state[t] = x
        o = np.zeros(STATE_DIM, dtype=np.float32)
        m = STATE_MASK.copy()
        for idx, name in zip(TANK_IDX, TANKS):
            if active is not None and active[1] == 'dropout' and (active[2] == name):
                m[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, MEASUREMENT_NOISE_STD)
        obs[t] = o
        mask[t] = m
    if return_nodes:
        return (obs, mask, state, kinds, phase, x0, nodes)
    return (obs, mask, state, kinds, phase, x0)
sim9 = simulate

def diurnal_simulator_shifts(t, period=48):
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / period)

def simulate_simulator_shifts(T=64, rng=None, p_event=0.08, event_len=6, noise_level=0.05, k=FLOW_GAIN, pump_gain=PUMP_GAIN, base_demand=BASE_DEMAND, meas_std=MEASUREMENT_NOISE_STD, overlap=False):
    rng = rng or np.random.default_rng()
    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[TANK_IDX] = TANK_SETPOINT
    x[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    state = np.zeros((T, STATE_DIM), dtype=np.float32)
    obs = np.zeros((T, STATE_DIM), dtype=np.float32)
    mask = np.tile(STATE_MASK, (T, 1)).copy()
    acts = []
    for t in range(T):
        acts = [a for a in acts if t < a[0]]
        if (overlap or not acts) and rng.random() < p_event:
            kind = rng.choice(['dropout', 'pump_trip', 'demand_spike'])
            node = rng.choice(TANKS) if kind in ('dropout', 'pump_trip') else rng.choice(JUNCTIONS)
            acts.append((t + event_len, kind, node))
        dh = x[None, :] - x[:, None]
        flow = k * (ADJ * dh).sum(axis=1)
        pump = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(TANK_IDX, TANKS):
            g = pump_gain
            if any((a[1] == 'pump_trip' and a[2] == name for a in acts)):
                g = 0.0
            pump[idx] = g * (TANK_SETPOINT - x[idx])
        demand = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(JUNCTION_IDX, JUNCTIONS):
            d = base_demand * diurnal_simulator_shifts(t) + rng.normal(0, noise_level)
            if any((a[1] == 'demand_spike' and a[2] == name for a in acts)):
                d *= 4.0
            demand[idx] = max(d, 0.0)
        x = x + 0.5 * (flow + pump - demand)
        state[t] = x
        o = np.zeros(STATE_DIM, dtype=np.float32)
        m = STATE_MASK.copy()
        for idx, name in zip(TANK_IDX, TANKS):
            if any((a[1] == 'dropout' and a[2] == name for a in acts)):
                m[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, meas_std)
        obs[t] = o
        mask[t] = m
    return (obs, mask, state)
DT, KP, K0, SV2 = (0.5, 0.6, 0.35, 0.03 ** 2)
TOPOS = {'paper (T1-J1, T2-J3, loop J1-J2-J3-J4)': (6, [0, 1], [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2)]), 'tanks on adjacent junctions (T1-J1, T2-J2)': (6, [0, 1], [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 2)]), 'both tanks on the same junction (T1-J1, T2-J1)': (6, [0, 1], [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 2)]), 'loop plus chord J1-J3': (6, [0, 1], [(0, 2), (1, 4), (2, 3), (3, 4), (4, 5), (5, 2), (2, 4)]), '8 nodes: 6-junction ring, tanks at opposite junctions': (8, [0, 1], [(0, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]), '8 nodes: 6-junction ring, tanks at adjacent junctions': (8, [0, 1], [(0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2)]), '8 nodes: 6-junction ring + 2 chords, tanks opposite': (8, [0, 1], [(0, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 2), (3, 6), (4, 7)])}

def build(n, tanks, edges, k):
    L = np.zeros((n, n))
    for e, (a, b) in enumerate(edges):
        L[a, a] += k[e]
        L[b, b] += k[e]
        L[a, b] -= k[e]
        L[b, a] -= k[e]
    E = np.zeros((n, n))
    for t in tanks:
        E[t, t] = 1.0
    return np.eye(n) - DT * (L + KP * E)

def obs_matrix(A, tanks, n):
    C = np.zeros((len(tanks), n))
    for i, t in enumerate(tanks):
        C[i, t] = 1.0
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return (O, C)

def gram(A, C, T):
    W = np.zeros((A.shape[0],) * 2)
    Ak = np.eye(A.shape[0])
    for _ in range(T):
        W += Ak.T @ C.T @ C @ Ak
        Ak = A @ Ak
    return W
if __name__ == '__main__':
    _cmd = sys.argv.pop(1) if len(sys.argv) > 1 else None
    if _cmd == 'network':
        print(f'{N} nodes, {len(EDGES)} edges')
        print('instrumented:', sum(INSTRUMENTED.values()), '/ unmeasured:', N - sum(INSTRUMENTED.values()))
        print('contains a cycle:', has_cycle())
        print('state adjacency:\n', state_adjacency_matrix())
    elif _cmd == 'data':
        obs, mask, x, events = generate_trajectory(T=48, rng=np.random.default_rng(0))
        print('obs shape:', obs.shape, 'true_state shape:', x.shape)
        print('state nodes:', STATE_NODES)
        print('events injected:', events)
        print('tank T1 trajectory (first 10 steps):', np.round(x[:10, STATE_INDEX['T1']], 3))
    elif _cmd == 'simulator':
        for s in (777, 888):
            a = np.random.default_rng(s)
            b = np.random.default_rng(s)
            for _ in range(3):
                o1, m1, x1, _ = generate_trajectory(T=64, rng=a, p_event=0.08)
                o2, m2, x2, k, ph, x0 = simulate(T=64, rng=b, p_event=0.08)
                assert np.allclose(o1, o2) and np.array_equal(m1, m2) and np.allclose(x1, x2), 'mismatch'
        print('fixed mode reproduces data.generate_trajectory exactly')
    elif _cmd == 'simulator_shifts':
        for s in (777, 888):
            a = np.random.default_rng(s)
            b = np.random.default_rng(s)
            for _ in range(3):
                o1, m1, x1, *_ = sim9(T=64, rng=a, p_event=0.08)
                o2, m2, x2 = simulate_simulator_shifts(T=64, rng=b)
                assert np.allclose(o1, o2) and np.array_equal(m1, m2) and np.allclose(x1, x2), 'mismatch'
        print('nominal mode reproduces simulator.simulate exactly')
    else:
        raise SystemExit('usage: python benchmark.py <network|data|simulator|simulator_shifts> [args]')
