#replays a capture, raises alerts and blocklists MACs

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import frames_to_windows, load_frames

BASE = Path(__file__).resolve().parent
AP_MAC = "aa:aa:aa:00:00:01"

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=str(BASE / "data" / "test_frames.csv"))
ap.add_argument("--model", default=str(BASE / "models" / "model_all.joblib"))
ap.add_argument("--block-after", type=int, default=2,
                help="consecutive flagged seconds before a MAC is blocklisted")
ap.add_argument("--allow", nargs="*", default=[AP_MAC],
                help="MACs that must never be auto-blocked (known AP)")
ap.add_argument("--quiet", action="store_true", help="print only the first alert per MAC")
args = ap.parse_args()

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
alert_file, block_file = RESULTS / "alerts.log", RESULTS / "blocklist.txt"
alert_file.write_text("")          # start each run with clean log files
block_file.write_text("")

# load model + traffic
bundle = joblib.load(args.model)
model, feats = bundle["model"], bundle["features"]
windows = frames_to_windows(load_frames(args.frames))
windows["pred"] = model.predict(windows[feats])            # 1 = attack
print(f"Model: {Path(args.model).name} | features: {feats}")
print(f"Replaying {len(windows):,} (second, device) rows ...\n")

# replay in time order 
streak = {}                # mac -> (last flagged second, consecutive count)
blocked, alerted = set(), set()
log_lines = []

for r in windows.sort_values("window").itertuples():
    if r.pred != 1:
        continue
    last, count = streak.get(r.src, (-99, 0))
    count = count + 1 if r.window - last == 1 else 1       # consecutive seconds?
    streak[r.src] = (r.window, count)

    action = f"flag {count}/{args.block_after}"
    if count >= args.block_after and r.src not in blocked:
        if r.src in args.allow:
            action = "NOT blocked: allow-listed (known AP) -> review manually"
        else:
            blocked.add(r.src)
            action = "ACTION: MAC BLOCKLISTED"
            with open(block_file, "a") as bf:
                bf.write(f"{r.src}\t{r.window}s\tdeauth flood\n")
    elif r.src in blocked:
        action = "already blocklisted"

    line = (f"[ALERT] t={r.window}s possible deauth flood from {r.src} "
            f"({r.deauth_count} deauths, {r.frames_per_sec} frames in 1s) | {action}")
    log_lines.append(line)
    if not args.quiet or r.src not in alerted:
        print(line)
    alerted.add(r.src)

alert_file.write_text("\n".join(log_lines) + "\n")

# score the run against the ground truth
tp = int(((windows.pred == 1) & (windows.label == 1)).sum())
fp = int(((windows.pred == 1) & (windows.label == 0)).sum())
fn = int(((windows.pred == 0) & (windows.label == 1)).sum())

# an "episode" = a run of consecutive attack seconds from the same device
truth = windows[windows.label == 1].sort_values("window")
episode_id = (truth.window.diff().fillna(1) > 1).cumsum()
detected, delays = 0, []
for _, ep in truth.groupby(episode_id):
    hit = ep[ep.pred == 1]
    if len(hit):
        detected += 1
        delays.append(int(hit.window.min() - ep.window.min()))

print("\n SUMMARY :")
print(f"Alerts raised (in seconds): {tp + fp}  (true {tp}, false {fp}); missed attack seconds: {fn}")
n_ep = episode_id.nunique()
print(f"Attack episodes detected: {detected}/{n_ep}")
if delays:
    print(f"Detection delay: median {pd.Series(delays).median():.0f}, "
          f"max {max(delays)}")
print(f"MACs blocklisted: {sorted(blocked) if blocked else 'none'}")
true_attackers = set(frames_to_windows(load_frames(args.frames)).query("label == 1").src)
print(f"Real attacker MACs: {sorted(true_attackers)}")
wrongly = blocked - true_attackers
print(f"Innocent devices wrongly blocklisted: {sorted(wrongly) if wrongly else 'none'}")
print(f"\nSaved: {alert_file.relative_to(BASE)}, {block_file.relative_to(BASE)}")
