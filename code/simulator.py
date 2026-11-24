import numpy as np
from data import (ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD,
                  TANK_IDX, JUNCTION_IDX, generate_trajectory)
from network import STATE_DIM, STATE_MASK

TANKS = ["T1", "T2"]
JUNCTIONS = ["J1", "J2", "J3", "J4"]


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
    kinds = np.full(T, "calm", dtype=object)
    nodes = np.full(T, "", dtype=object)
    active = None
    for t in range(T):
        if active is not None and t >= active[0]:
            active = None
        if active is None and rng.random() < p_event:
            kind = rng.choice(["dropout", "pump_trip", "demand_spike"])
            node = rng.choice(TANKS) if kind in ("dropout", "pump_trip") else rng.choice(JUNCTIONS)
            active = (t + event_len, kind, node)
        if active is not None:
            kinds[t] = active[1]; nodes[t] = active[2]
        dh = x[None, :] - x[:, None]
        flow = FLOW_GAIN * (ADJ * dh).sum(axis=1)
        pump = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(TANK_IDX, TANKS):
            g = PUMP_GAIN
            if active is not None and active[1] == "pump_trip" and active[2] == name:
                g = 0.0
            pump[idx] = g * (TANK_SETPOINT - x[idx])
        demand = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(JUNCTION_IDX, JUNCTIONS):
            d = BASE_DEMAND * diurnal(t, phase) + rng.normal(0, noise_level)
            if active is not None and active[1] == "demand_spike" and active[2] == name:
                d *= 4.0
            demand[idx] = max(d, 0.0)
        x = x + 0.5 * (flow + pump - demand)
        state[t] = x
        o = np.zeros(STATE_DIM, dtype=np.float32)
        m = STATE_MASK.copy()
        for idx, name in zip(TANK_IDX, TANKS):
            if active is not None and active[1] == "dropout" and active[2] == name:
                m[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, MEASUREMENT_NOISE_STD)
        obs[t] = o; mask[t] = m
    if return_nodes:
        return obs, mask, state, kinds, phase, x0, nodes
    return obs, mask, state, kinds, phase, x0


if __name__ == "__main__":
    for s in (777, 888):
        a = np.random.default_rng(s); b = np.random.default_rng(s)
        for _ in range(3):
            o1, m1, x1, _ = generate_trajectory(T=64, rng=a, p_event=0.08)
            o2, m2, x2, k, ph, x0 = simulate(T=64, rng=b, p_event=0.08)
            assert np.allclose(o1, o2) and np.array_equal(m1, m2) and np.allclose(x1, x2), "mismatch"
    print("fixed mode reproduces data.generate_trajectory exactly")
