"""Random forest step 1: sanity-check the features, then a default-RF baseline.

Grouped 5-fold CV on the 609 training rows (team folds from data.group_folds).
Inside each fold the features are re-fit on the training folds only, and the
blank survey rows are dropped from the training folds only.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from data import LABELS, get_split, group_folds
from rf_features import RFFeatures, is_blank_row

SEED = 42

train, val, _test = get_split()  # test is not touched here

# --- sanity checks against the proposal -------------------------------------
fe = RFFeatures().fit(train[~is_blank_row(train)])
X = fe.transform(train)
n_text = sum(len(v) for v in fe.p["vocab"].values())
n_multi = sum(len(v) for v in fe.p["options"].values())
print(f"Rows: train {len(train)}, val {len(val)}")
print(f"Features: {X.shape[1]} total = 7 numeric/flags + {n_multi} multi-hot + {n_text} text words")
print("Vocab per field:", {c: len(v) for c, v in fe.p["vocab"].items()})
print("Song missing (incl. non-answers):", int(X[:, fe.feature_names.index('song_missing')].sum()))
print(f"Pay clip (train 99th pct): ${fe.p['pay_clip']:.0f}")


# --- baseline: default RandomForestClassifier --------------------------------
def cv_scores(make_model, df, folds):
    accs, f1s, train_accs = [], [], []
    for k in range(folds.max() + 1):
        tr, va = df[folds != k], df[folds == k]
        tr = tr[~is_blank_row(tr)]
        feats = RFFeatures().fit(tr)
        model = make_model().fit(feats.transform(tr), tr["label"])
        pred = model.predict(feats.transform(va))
        train_accs.append(accuracy_score(tr["label"], model.predict(feats.transform(tr))))
        accs.append(accuracy_score(va["label"], pred))
        f1s.append(f1_score(va["label"], pred, average="macro", labels=LABELS))
    return np.array(accs), np.array(f1s), np.array(train_accs)


folds = group_folds(train)
acc, f1, tr_acc = cv_scores(lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1), train, folds)
print("\nDefault RF, grouped 5-fold CV on train:")
print(f"  macro-F1 {f1.mean():.3f} ± {f1.std():.3f}   per fold {np.round(f1, 3)}")
print(f"  accuracy {acc.mean():.3f} ± {acc.std():.3f}   (train accuracy {tr_acc.mean():.3f})")
