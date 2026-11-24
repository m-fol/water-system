import numpy as np
from data import ADJ, FLOW_GAIN, PUMP_GAIN, TANK_SETPOINT, BASE_DEMAND, MEASUREMENT_NOISE_STD, TANK_IDX, JUNCTION_IDX
from network import STATE_DIM, STATE_MASK
from simulator import simulate as sim9

TANKS = ["T1", "T2"]
JUNCTIONS = ["J1", "J2", "J3", "J4"]


def diurnal(t, period=48):
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / period)


def simulate(T=64, rng=None, p_event=0.08, event_len=6, noise_level=0.05, k=FLOW_GAIN, pump_gain=PUMP_GAIN, base_demand=BASE_DEMAND,
             meas_std=MEASUREMENT_NOISE_STD, overlap=False):
    rng = rng or np.random.default_rng()
    x = np.zeros(STATE_DIM, dtype=np.float32)
    x[TANK_IDX] = TANK_SETPOINT; x[JUNCTION_IDX] = TANK_SETPOINT - 1.0
    state = np.zeros((T, STATE_DIM), dtype=np.float32); obs = np.zeros((T, STATE_DIM), dtype=np.float32)
    mask = np.tile(STATE_MASK, (T, 1)).copy()
    acts = []
    for t in range(T):
        acts = [a for a in acts if t < a[0]]
        if (overlap or not acts) and rng.random() < p_event:
            kind = rng.choice(["dropout", "pump_trip", "demand_spike"])
            node = rng.choice(TANKS) if kind in ("dropout", "pump_trip") else rng.choice(JUNCTIONS)
            acts.append((t + event_len, kind, node))
        dh = x[None, :] - x[:, None]
        flow = k * (ADJ * dh).sum(axis=1)
        pump = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(TANK_IDX, TANKS):
            g = pump_gain
            if any(a[1] == "pump_trip" and a[2] == name for a in acts):
                g = 0.0
            pump[idx] = g * (TANK_SETPOINT - x[idx])
        demand = np.zeros(STATE_DIM, dtype=np.float32)
        for idx, name in zip(JUNCTION_IDX, JUNCTIONS):
            d = base_demand * diurnal(t) + rng.normal(0, noise_level)
            if any(a[1] == "demand_spike" and a[2] == name for a in acts):
                d *= 4.0
            demand[idx] = max(d, 0.0)
        x = x + 0.5 * (flow + pump - demand)
        state[t] = x
        o = np.zeros(STATE_DIM, dtype=np.float32); m = STATE_MASK.copy()
        for idx, name in zip(TANK_IDX, TANKS):
            if any(a[1] == "dropout" and a[2] == name for a in acts):
                m[idx] = False
            else:
                o[idx] = x[idx] + rng.normal(0, meas_std)
        obs[t] = o; mask[t] = m
    return obs, mask, state


if __name__ == "__main__":
    for s in (777, 888):
        a = np.random.default_rng(s); b = np.random.default_rng(s)
        for _ in range(3):
            o1, m1, x1, *_ = sim9(T=64, rng=a, p_event=0.08)
            o2, m2, x2 = simulate(T=64, rng=b)
            assert np.allclose(o1, o2) and np.array_equal(m1, m2) and np.allclose(x1, x2), "mismatch"
    print("nominal mode reproduces simulator.simulate exactly")
