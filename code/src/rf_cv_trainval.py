"""Grouped 5-fold CV on train + validation (247 students, 741 rows) for the team comparison.

The test split is never used. The fold of each student is saved to
data/folds_trainval.json on the first run; share that file so every teammate
scores their model on exactly the same folds (sklearn versions can otherwise
assign GroupKFold folds differently). If the file exists, it is loaded instead.

Teammates: copy `load_folds` (or the JSON) and score your own model fold by fold.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from data import LABELS, ROOT, get_split
from rf_features import RFFeatures, is_blank_row

SEED = 42
FOLDS_PATH = ROOT / "data" / "folds_trainval.json"
BEST = json.loads((ROOT / "results" / "rf" / "best_params.json").read_text())["params"]


def load_folds(df):
    """Fold number (0-4) per row of df, by student id; creates the shared file once."""
    if FOLDS_PATH.exists():
        by_id = {int(k): v for k, v in json.loads(FOLDS_PATH.read_text()).items()}
    else:
        by_id = {}
        for k, (_, idx) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["id"])):
            for i in df["id"].iloc[idx].unique():
                by_id[int(i)] = k
        FOLDS_PATH.write_text(json.dumps({str(i): k for i, k in sorted(by_id.items())}, indent=0))
        print(f"Saved fold assignment for {len(by_id)} students to {FOLDS_PATH} - share this file.")
    return df["id"].map(by_id).to_numpy()


train, val, _test = get_split()  # test is not touched
df = pd.concat([train, val], ignore_index=True)
folds = load_folds(df)

f1s, accs = [], []
for k in range(5):
    tr, va = df[folds != k], df[folds == k]
    tr = tr[~is_blank_row(tr)]
    fe = RFFeatures().fit(tr)
    model = RandomForestClassifier(**BEST, random_state=SEED, n_jobs=-1).fit(fe.transform(tr), tr["label"])
    pred = model.predict(fe.transform(va))
    f1s.append(f1_score(va["label"], pred, average="macro", labels=LABELS))
    accs.append(accuracy_score(va["label"], pred))

print(f"Random forest, grouped 5-fold CV on train + validation ({len(df)} rows, {df.id.nunique()} students)")
print(f"  macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}   per fold {np.round(f1s, 3)}")
print(f"  accuracy {np.mean(accs):.3f} ± {np.std(accs):.3f}")
out = ROOT / "results" / "rf" / "cv_trainval.json"
out.write_text(json.dumps({"macro_f1": f1s, "accuracy": accs, "params": BEST}, indent=1, default=str))
