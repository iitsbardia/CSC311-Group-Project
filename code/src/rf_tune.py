"""Random forest step 2: hyperparameter tuning with grouped 5-fold CV on the training split.

What it does
  1. Builds the features once per fold (they don't depend on the forest settings),
     fitting them on the training folds only and dropping blank rows from training folds.
  2. Random search: N_ITER random combinations from GRID, scored by mean macro-F1.
  3. Validation curves: varies max_depth, then max_features, one at a time around the
     best combination, recording train and CV scores (shows over/underfitting).
  4. Extra checks at the best settings: no-text ablation and top feature importances.

Outputs (results/rf/)
  tuning.csv           every combination tried, sorted by CV macro-F1
  best_params.json     the winning settings (rf_final.py reads this)
  curve_depth.csv / curve_max_features.csv and the matching .png plots
  importances.csv      mean impurity-based importance per feature

Validation and test rows are never used here.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler

from data import LABELS, ROOT, get_split, group_folds
from rf_features import RFFeatures, is_blank_row

SEED = 42
N_ITER = 60
OUT = ROOT / "results" / "rf"
OUT.mkdir(parents=True, exist_ok=True)

GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, 30, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.05, 0.1, 0.2],
    "criterion": ["gini", "entropy"],
}
DEPTH_CURVE = [2, 4, 6, 8, 10, 15, 20, 30, 50, None]
# log2(554) = 9 and sqrt(554) = 23 features per split; small ints extend the range downwards.
MAXF_CURVE = [3, 5, "log2", "sqrt", 0.05, 0.1, 0.2, 0.3, 0.5]
TREES_CURVE = [25, 50, 100, 200, 300, 500, 1000]
# Values that are already the natural limit, so "best at the edge" can't be widened further.
NATURAL_LIMITS = {"max_depth": "None", "min_samples_leaf": "1"}
# These get their own wider curve below, so the grid-edge warning doesn't apply to them.
HAS_CURVE = {"max_depth", "max_features", "n_estimators"}

train, _val, _test = get_split()
folds = group_folds(train)
K = folds.max() + 1

# ---- 1. features per fold (fit on training folds only) ----------------------
FOLD_DATA = []
for k in range(K):
    tr, va = train[folds != k], train[folds == k]
    tr = tr[~is_blank_row(tr)]
    fe = RFFeatures().fit(tr)
    FOLD_DATA.append((fe.transform(tr), tr["label"].to_numpy(),
                      fe.transform(va), va["label"].to_numpy(), fe.feature_names))


def evaluate(params, cols=None):
    """Mean/std of CV macro-F1 and accuracy, plus mean train accuracy, for one setting."""
    f1s, accs, tr_accs = [], [], []
    for Xtr, ytr, Xva, yva, _ in FOLD_DATA:
        if cols is not None:
            Xtr, Xva = Xtr[:, cols], Xva[:, cols]
        m = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        p = m.predict(Xva)
        f1s.append(f1_score(yva, p, average="macro", labels=LABELS))
        accs.append(accuracy_score(yva, p))
        tr_accs.append(accuracy_score(ytr, m.predict(Xtr)))
    return {"cv_f1": np.mean(f1s), "cv_f1_std": np.std(f1s), "cv_acc": np.mean(accs),
            "cv_acc_std": np.std(accs), "train_acc": np.mean(tr_accs)}


def show(v):
    return "None" if v is None else str(v)


# ---- 2. random search -------------------------------------------------------
t0 = time.time()
default = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1,
           "max_features": "sqrt", "criterion": "gini"}
candidates = [default] + list(ParameterSampler(GRID, n_iter=N_ITER, random_state=SEED))
rows = []
for i, params in enumerate(candidates, 1):
    rows.append({**{k: show(v) for k, v in params.items()}, **evaluate(params)})
    if i % 10 == 0:
        print(f"  {i}/{len(candidates)} done ({time.time() - t0:.0f}s)")
res = pd.DataFrame(rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
res.round(4).to_csv(OUT / "tuning.csv", index=False)

best_idx = int(np.argmax([r["cv_f1"] for r in rows]))
best = candidates[best_idx]
print(f"\nRandom search: {len(candidates)} settings in {time.time() - t0:.0f}s. Top 5:")
print(res.head(5).round(3).to_string(index=False))
print("Default RF:", {k: round(v, 3) for k, v in rows[0].items() if k.startswith(("cv", "train"))})


# ---- 3. validation curves around the best setting ---------------------------
def curve(name, values, fname, xlabel):
    out = []
    for v in values:
        r = evaluate({**best, name: v})
        out.append({name: show(v), **r})
    df = pd.DataFrame(out)
    df.round(4).to_csv(OUT / f"{fname}.csv", index=False)
    xs = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(xs, df["train_acc"], "o-", label="train accuracy")
    ax.errorbar(xs, df["cv_f1"], yerr=df["cv_f1_std"], fmt="o-", capsize=3, label="CV macro-F1 (±1 SD)")
    ax.set_xticks(xs, df[name]); ax.set_xlabel(xlabel); ax.set_ylabel("score")
    ax.set_ylim(min(0.8, df["cv_f1"].min() - 0.05), 1.01); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"{fname}.png", dpi=200); plt.close(fig)
    print(f"\n{xlabel} curve:"); print(df[[name, "cv_f1", "cv_f1_std", "train_acc"]].round(3).to_string(index=False))
    return df


depth_df = curve("max_depth", DEPTH_CURVE, "curve_depth", "max_depth (others at best)")
maxf_df = curve("max_features", MAXF_CURVE, "curve_max_features", "max_features (others at best)")
trees_df = curve("n_estimators", TREES_CURVE, "curve_n_estimators", "n_estimators (others at best)")

# Warn if the best value sits at the edge of the searched range (template Appendix B 4b).
for name, grid in GRID.items():
    if NATURAL_LIMITS.get(name) == show(best[name]) or name in HAS_CURVE:
        continue
    if len(grid) > 2 and show(best[name]) in (show(grid[0]), show(grid[-1])):
        print(f"NOTE: best {name}={show(best[name])} is at the edge of its range {grid}; consider widening it.")

# ---- 4. extra checks at the best setting ------------------------------------
best_r = evaluate(best)
notext_scores = []
for Xtr, ytr, Xva, yva, nm in FOLD_DATA:
    cols = [j for j, n in enumerate(nm) if ":" not in n]  # text columns are "field:word"
    m = RandomForestClassifier(**best, random_state=SEED, n_jobs=-1).fit(Xtr[:, cols], ytr)
    notext_scores.append(f1_score(yva, m.predict(Xva[:, cols]), average="macro", labels=LABELS))
print(f"\nBest setting CV macro-F1 {best_r['cv_f1']:.3f} ± {best_r['cv_f1_std']:.3f}"
      f" | without text features {np.mean(notext_scores):.3f} ± {np.std(notext_scores):.3f}")

imp = {}
for Xtr, ytr, _, _, nm in FOLD_DATA:
    m = RandomForestClassifier(**best, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    for n, v in zip(nm, m.feature_importances_):
        imp[n] = imp.get(n, 0) + v / K
imp = pd.Series(imp).sort_values(ascending=False)
imp.round(5).to_frame("importance").to_csv(OUT / "importances.csv")
print("\nTop 15 features:"); print(imp.head(15).round(4).to_string())

(OUT / "best_params.json").write_text(json.dumps(
    {"params": best, "cv_f1": best_r["cv_f1"], "cv_f1_std": best_r["cv_f1_std"],
     "cv_acc": best_r["cv_acc"], "cv_f1_no_text": float(np.mean(notext_scores)),
     "n_iter": N_ITER, "seed": SEED}, indent=1, default=str))
print(f"\nSaved results to {OUT}  (total {time.time() - t0:.0f}s)")
