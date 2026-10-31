"""
features.py  -  STEP 2: turn frames into ML features.

Idea: the classifier does not look at single frames. We group frames into
1-second windows PER SOURCE MAC and describe each (window, device) pair with
a few numbers. One row of the output = "what this device did in this second".

Run it on its own to inspect the result:
    python features.py data/train_frames.csv
"""
import sys

import pandas as pd

# The features the model sees (it never sees the MAC address or the answer).
FEATURES = [
    "frames_per_sec",   # how many frames this device sent in the second
    "deauth_count",     # how many of them were deauthentication frames
    "mgmt_ratio",       # share of its frames that were management frames (0..1)
    "unique_dst",       # how many different receivers it sent to
    "mean_rssi",        # average signal strength (dBm) of its frames
]


def load_frames(path):
    return pd.read_csv(path)


def frames_to_windows(frames):
    """Frame table -> one row per (1-second window, source MAC)."""
    f = frames.copy()
    f["window"] = f["time"].astype(int)                       # second 0, 1, 2, ...
    f["is_deauth"] = ((f["type"] == 0) & (f["subtype"] == 12)).astype(int)
    f["is_mgmt"] = (f["type"] == 0).astype(int)

    w = f.groupby(["window", "src"]).agg(
        frames_per_sec=("time", "size"),
        deauth_count=("is_deauth", "sum"),
        mgmt_ratio=("is_mgmt", "mean"),
        unique_dst=("dst", "nunique"),
        mean_rssi=("rssi", "mean"),
        n_attack=("is_attack_frame", "sum"),                  # ground truth, labels only
    ).reset_index()

    # Label: 1 if the device sent any attacker frame in that second, else 0.
    w["label"] = (w["n_attack"] > 0).astype(int)
    return w.drop(columns="n_attack")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train_frames.csv"
    windows = frames_to_windows(load_frames(path))
    pd.set_option("display.width", 120)
    print(windows.head(10).to_string(index=False))
    print("\nWindows per class (0 = normal, 1 = attack):")
    print(windows["label"].value_counts().to_string())
    print("\nAverage feature values per class:")
    print(windows.groupby("label")[FEATURES].mean().round(2).to_string())
