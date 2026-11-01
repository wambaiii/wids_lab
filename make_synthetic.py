"""
make_synthetic.py  -  STEP 1: generate SYNTHETIC 802.11 frame logs.

!! This is simulated traffic, NOT a real capture. Every rate, signal level and
!! attack parameter below is an ASSUMPTION chosen for the lab, not a measurement

Output (one row = one frame):
    time, src, dst, type, subtype, rssi, is_attack_frame
        type    : 0 = management, 2 = data
        subtype : (type 0) 4 = probe request, 8 = beacon, 10 = disassoc, 12 = deauth
        rssi    : signal strength in dBm
        is_attack_frame : GROUND TRUTH (1 = frame sent by the simulated attacker).
                          Used ONLY to create labels and to score the detector.

Two independent runs are created so the model is tested on traffic it has never seen:
    data/train_frames.csv  (seed 1)      data/test_frames.csv  (seed 2)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

AP = "aa:aa:aa:00:00:01"                                   # the access point
CLIENTS = [f"cc:cc:cc:00:00:{i:02x}" for i in range(1, 9)]  # 8 normal devices
ATTACKER = "ee:ee:ee:66:66:66"                             # the simulated attacker
BCAST = "ff:ff:ff:ff:ff:ff"

DURATION = 900     # seconds of simulated traffic per run
N_ATTACKS = 12     # number of deauth attack episodes per run


def poisson_times(rng, rate, start, end):
    """Random frame timestamps: each second gets Poisson(rate) frames at random
    moments inside that second. `rate` may be a number or one value per second."""
    secs = np.arange(start, end)
    counts = rng.poisson(np.broadcast_to(rate, secs.shape))
    return np.repeat(secs, counts) + rng.random(counts.sum())


class Log:
    """Collects frames from every device, then merges them into one table."""

    def __init__(self):
        self.parts = []

    def add(self, t, src, dst, ftype, subtype, rssi, attack):
        if len(t) == 0:
            return
        self.parts.append(pd.DataFrame({
            "time": t, "src": src, "dst": dst, "type": ftype, "subtype": subtype,
            "rssi": rssi, "is_attack_frame": int(attack)}))

    def table(self):
        df = pd.concat(self.parts, ignore_index=True).sort_values("time")
        df["time"] = df["time"].round(6)
        df["rssi"] = df["rssi"].round().astype(int)
        return df.reset_index(drop=True)


def generate(seed):
    rng = np.random.default_rng(seed)       # fixed seed -> reproducible results
    log, episodes = Log(), []

    # ---- Access point: beacons (~10/s), data to clients (~30/s), rare normal deauth
    t = poisson_times(rng, 10, 0, DURATION)
    log.add(t, AP, BCAST, 0, 8, rng.normal(-40, 3, len(t)), False)
    t = poisson_times(rng, 30, 0, DURATION)
    log.add(t, AP, rng.choice(CLIENTS, len(t)), 2, 8, rng.normal(-40, 3, len(t)), False)
    t = poisson_times(rng, 0.01, 0, DURATION)              # legit: AP removes a client
    log.add(t, AP, rng.choice(CLIENTS, len(t)), 0, 12, rng.normal(-40, 3, len(t)), False)

    # ---- Normal clients: mostly data, a few probe requests, rare legit deauth
    base = {}
    for mac in CLIENTS:
        rate = rng.uniform(3, 30)                          # each client has its own activity
        base[mac] = rng.uniform(-72, -45)                  # and its own distance from the AP
        active = rng.random(DURATION) < 0.8                # idle ~20% of seconds
        t = poisson_times(rng, rate * active, 0, DURATION)
        log.add(t, mac, AP, 2, 8, rng.normal(base[mac], 3, len(t)), False)
        t = poisson_times(rng, 0.5, 0, DURATION)           # occasional probe request
        log.add(t, mac, BCAST, 0, 4, rng.normal(base[mac], 3, len(t)), False)
        t = poisson_times(rng, 0.003, 0, DURATION)         # legit: client leaves
        log.add(t, mac, AP, 0, 12, rng.normal(base[mac], 3, len(t)), False)

    # ---- HARD NORMAL CASE: a client scanning for networks (burst of probe requests).
    #      All management frames at a high rate, so it looks a bit like a flood.
    for _ in range(3):
        mac = str(rng.choice(CLIENTS))
        dur = int(rng.integers(5, 13))
        s = int(rng.integers(0, DURATION - dur))
        t = poisson_times(rng, rng.uniform(40, 150), s, s + dur)
        log.add(t, mac, BCAST, 0, 4, rng.normal(base[mac], 3, len(t)), False)

    # ---- ATTACK: deauthentication floods from a different device
    slot = DURATION // N_ATTACKS
    for k in range(N_ATTACKS):
        dur = int(rng.integers(8, 41))
        start = k * slot + int(rng.integers(0, slot - dur))
        rate = float(np.exp(rng.uniform(np.log(5), np.log(300))))   # 5..300 frames/s
        mode = str(rng.choice(["single", "broadcast", "sweep"]))
        t = poisson_times(rng, rate, start, start + dur)
        n = len(t)
        if mode == "single":                               # hit one victim
            dst = np.full(n, str(rng.choice(CLIENTS)))
        elif mode == "broadcast":                          # hit everybody at once
            dst = np.full(n, BCAST)
        else:                                              # cycle through victims
            dst = rng.choice(CLIENTS, n)
        subtype = np.where(rng.random(n) < 0.15, 10, 12)   # mostly deauth, some disassoc
        log.add(t, ATTACKER, dst, 0, subtype, rng.normal(-52, 4, n), True)
        episodes.append({"start": start, "end": start + dur,
                         "rate_fps": round(rate, 1), "mode": mode})

    return log.table(), pd.DataFrame(episodes)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, seed in [("train", 1), ("test", 2)]:
        frames, episodes = generate(seed)
        frames.to_csv(DATA / f"{name}_frames.csv", index=False)
        episodes.to_csv(DATA / f"{name}_episodes.csv", index=False)
        print(f"{name}: {len(frames):,} frames, "
              f"{int(frames['is_attack_frame'].sum()):,} attack frames, "
              f"{len(episodes)} attack episodes -> data/{name}_frames.csv")
    print("\nREMINDER: this is simulated data. Results measure how separable the")
    print("simulated classes are, not performance on real Wi-Fi attacks.")
