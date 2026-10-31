"""
train.py  -  STEP 3: train classifiers, measure accuracy, draw confusion matrices.

Train on data/train_frames.csv, test on data/test_frames.csv (a separate run the
model has never seen). Two experiments are run:
    all              : all 5 features
    no_deauth_count  : same, without deauth_count (a harder, more honest test)

Outputs: results/metrics.csv, results/confusion_*.png, results/decision_tree_rules.txt,
         results/feature_importance.csv, models/model_*.joblib
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier, export_text

from features import FEATURES, frames_to_windows, load_frames

BASE = Path(__file__).resolve().parent
RESULTS, MODELS = BASE / "results", BASE / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ---- load data -------------------------------------------------------------
train = frames_to_windows(load_frames(BASE / "data" / "train_frames.csv"))
test = frames_to_windows(load_frames(BASE / "data" / "test_frames.csv"))
print(f"Train windows: {len(train):,} (attack {int(train.label.sum())})")
print(f"Test  windows: {len(test):,} (attack {int(test.label.sum())})\n")

experiments = {
    "all": FEATURES,
    "no_deauth_count": [f for f in FEATURES if f != "deauth_count"],
}
make_models = {
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                                   random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                   random_state=42),
}

rows, importances = [], []
for exp, feats in experiments.items():
    for name, make in make_models.items():
        model = make().fit(train[feats], train["label"])      # learn from the train run
        pred = model.predict(test[feats])                     # predict the unseen run
        y = test["label"]
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "experiment": exp, "model": name,
            "accuracy": accuracy_score(y, pred),
            "attack_precision": precision_score(y, pred, zero_division=0),
            "attack_recall": recall_score(y, pred, zero_division=0),
            "attack_f1": f1_score(y, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

        disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} ({exp})\naccuracy = {accuracy_score(y, pred):.4f}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"confusion_{name}_{exp}.png", dpi=200)
        plt.close()

        if name == "RandomForest":
            joblib.dump({"model": model, "features": feats}, MODELS / f"model_{exp}.joblib")
            for f, imp in zip(feats, model.feature_importances_):
                importances.append({"experiment": exp, "feature": f, "importance": imp})
        if name == "DecisionTree" and exp == "all":
            (RESULTS / "decision_tree_rules.txt").write_text(
                export_text(model, feature_names=feats))

metrics = pd.DataFrame(rows)
metrics.to_csv(RESULTS / "metrics.csv", index=False)
pd.DataFrame(importances).to_csv(RESULTS / "feature_importance.csv", index=False)

pd.set_option("display.width", 140)
print(metrics.round(4).to_string(index=False))
print("\nRandom Forest feature importance:")
print(pd.DataFrame(importances).round(3).to_string(index=False))
