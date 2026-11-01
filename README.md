# WIDS / WIPS lab: deauthentication flood detection with scikit-learn

**Data is SYNTHETIC (simulated), not a real capture.** All traffic rates, signal levels and
attack parameters are assumptions made in `make_synthetic.py`. Accuracy therefore measures how
separable the simulated classes are, not performance on real Wi-Fi attacks.

## Run 
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python make_synthetic.py            # step 1: create data/train_frames.csv and data/test_frames.csv
    python features.py data\train_frames.csv   # step 2: look at the features
    python train.py                     # step 3: train, accuracy, confusion matrices -> results/
    python detect.py                    # step 4: alerts + WIPS blocklist -> results/

## Files
| File | Purpose |
|---|---|
| make_synthetic.py | Simulates AP, 8 clients, benign scan bursts, and 12 deauth floods per run |
| features.py | Frames -> 1-second windows per MAC -> 5 features |
| train.py | Decision Tree + Random Forest, two experiments, metrics, confusion matrices |
| detect.py | Replays traffic, prints alerts, blocklists repeat offenders (allow-lists the AP) |
