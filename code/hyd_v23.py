import os, tempfile, time, glob
import numpy as np
import wntr

T = 64
STEP = 1800
TANK_NODES = ["T1", "T2"]; JUNCTIONS = ["J1", "J2", "J3", "J4"]; NODES = TANK_NODES + JUNCTIONS
PIPES = [("T1", "J1"), ("T2", "J3"), ("J1", "J2"), ("J1", "J4"), ("J2", "J3"), ("J3", "J4")]
SET_LEVEL = 8.0
SUPPLY_HEAD = 9.0
BASE_Q = 0.010
NOISE_Q = 0.004
MEAS_STD = 0.03
TANK_DIAM = 14.0


def build(pipe_d=0.25, pipe_len=600.0, hw=100.0, hetero=None, rng=None):
    wn = wntr.network.WaterNetworkModel()
    wn.options.time.duration = (T - 1) * STEP
    wn.options.time.hydraulic_timestep = STEP
    wn.options.time.pattern_timestep = STEP
    wn.options.time.report_timestep = STEP
    wn.options.hydraulic.demand_model = "DDA"
    for tn in TANK_NODES:
        wn.add_tank(tn, elevation=0.0, init_level=SET_LEVEL, min_level=0.0, max_level=50.0, diameter=TANK_DIAM)
    for jn in JUNCTIONS:
        wn.add_junction(jn, base_demand=0.0, elevation=0.0)
    for i, tn in enumerate(TANK_NODES):
        wn.add_reservoir("S%d" % (i + 1), base_head=SUPPLY_HEAD)
        wn.add_pipe("SP%d" % (i + 1), "S%d" % (i + 1), tn, length=pipe_len, diameter=pipe_d, roughness=hw)
    for k, (a, b) in enumerate(PIPES):
        d = pipe_d if hetero is None else pipe_d * hetero[k]
        wn.add_pipe("P%d" % k, a, b, length=pipe_len, diameter=d, roughness=hw)
    return wn


def simulate(rng, p_event=0.08, event_len=6, hetero=None, simulator="epanet", build_kw=None):
    wn = build(hetero=hetero, **(build_kw or {}))
    diurnal = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(T) / 48)
    regime = np.zeros(T, dtype=int); drop = np.zeros((T, 2), dtype=bool)
    mult = {jn: diurnal.copy() for jn in JUNCTIONS}
    trip_windows = []
    t = 0
    while t < T:
        if rng.random() < p_event:
            kind = rng.choice(["dropout", "pump_trip", "demand_spike"])
            end = min(t + event_len, T)
            if kind == "dropout":
                drop[t:end, rng.integers(2)] = True
            elif kind == "pump_trip":
                k = int(rng.integers(2)); regime[t:end] = 1 + k; trip_windows.append((k, t, end))
            else:
                j = int(rng.integers(4)); regime[t:end] = 3 + j; mult[JUNCTIONS[j]][t:end] *= 4.0
            t += event_len
        else:
            t += 1
    for j, jn in enumerate(JUNCTIONS):
        noise = rng.normal(0, NOISE_Q / BASE_Q, T)
        pat = np.maximum(mult[jn] + noise * np.where(regime == 3 + j, 4.0, 1.0), 0.0)
        wn.add_pattern("pat_" + jn, pat.tolist())
        wn.get_node(jn).add_demand(BASE_Q, "pat_" + jn)
    for k, a, b in trip_windows:
        pipe = wn.get_link("SP%d" % (k + 1))
        c = wntr.network.controls
        wn.add_control("close_%d_%d" % (k, a), c.Control(c.SimTimeCondition(wn, "=", a * STEP), c.ControlAction(pipe, "status", wntr.network.LinkStatus.Closed)))
        if b < T:
            wn.add_control("open_%d_%d" % (k, a), c.Control(c.SimTimeCondition(wn, "=", b * STEP), c.ControlAction(pipe, "status", wntr.network.LinkStatus.Open)))
    sim = wntr.sim.EpanetSimulator(wn) if simulator == "epanet" else wntr.sim.WNTRSimulator(wn)
    if simulator == "epanet":
        prefix = os.path.join(tempfile.gettempdir(), "hyd_%d_%d" % (os.getpid(), time.time_ns()))
        for attempt in range(5):
            try:
                res = sim.run_sim(file_prefix=prefix + "_%d" % attempt); break
            except Exception:
                if attempt == 4: raise
                time.sleep(0.2 * (attempt + 1))
        for f in glob.glob(prefix + "*"):
            try: os.remove(f)
            except OSError: pass
    else:
        res = sim.run_sim()
    head = res.node["head"][NODES].values[:T]
    state = head.astype(np.float32)
    obs = np.zeros((T, 6), dtype=np.float32); mask = np.zeros((T, 6), dtype=bool)
    for k in range(2):
        obs[:, k] = state[:, k] + rng.normal(0, MEAS_STD, T)
        mask[:, k] = ~drop[:, k]; obs[~mask[:, k], k] = 0.0
    return obs, mask, state, regime


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t0 = time.time(); o, m, x, r = simulate(rng); print("epanet s", time.time() - t0)
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print(x[::8]); print(r)
    t0 = time.time(); o, m, x, r = simulate(np.random.default_rng(0), simulator="wntr"); print("wntr s", time.time() - t0); print(x[::8])
