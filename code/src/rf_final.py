"""Random forest step 3: validation score, one-time test score, export for pred.py.

    python rf_final.py val     # train on train split, score on the 132 validation rows.
                               # Share this number with the team to pick the final model.
    python rf_final.py final   # only if the random forest is the team's final model:
                               #   a) train on train+val, score ONCE on the 129 test rows
                               #   b) retrain on all 870 rows, export rf_preprocess.json +
                               #      rf_forest.npz into submission/, copy pred.py there
                               #   c) check pred.py gives the same predictions as sklearn,
                               #      runs fast, and the submission is under 10 MB

Settings come from results/rf/best_params.json (written by rf_tune.py).
"""

import json
import shutil
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import DATA_PATH, LABELS, ROOT, clean, get_split, load_raw
from rf_features import RFFeatures, is_blank_row

SEED = 42
OUT = ROOT / "results" / "rf"
SUB = ROOT / "submission"
BEST = json.loads((OUT / "best_params.json").read_text())["params"]


def fit(df):
    """Fit features + forest on df (blank survey rows dropped)."""
    df = df[~is_blank_row(df)]
    fe = RFFeatures().fit(df)
    model = RandomForestClassifier(**BEST, random_state=SEED, n_jobs=-1)
    model.fit(fe.transform(df), df["label"])
    return fe, model


def report(name, y, pred):
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=LABELS)
    cm = confusion_matrix(y, pred, labels=LABELS)
    print(f"\n{name}: accuracy {acc:.3f}, macro-F1 {f1:.3f}  (n = {len(y)})")
    print(classification_report(y, pred, labels=LABELS, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Random forest, {name}")
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{name}.png", dpi=200); plt.close(fig)
    return {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(), "n": len(y), "params": BEST}


def show_errors(df, pred, n=8):
    wrong = df[df["label"].to_numpy() != pred].assign(predicted=pred[df["label"].to_numpy() != pred])
    cols = ["label", "predicted", "colour", "song", "seasons", "music"]
    print(f"\nSome misclassified rows ({len(wrong)} total):")
    print(wrong[cols].head(n).to_string(index=False, max_colwidth=30))


def export(fe, model):
    """Write rf_preprocess.json and rf_forest.npz (all trees as flat arrays)."""
    SUB.mkdir(exist_ok=True)
    prep = dict(fe.p, classes=[str(c) for c in model.classes_])
    (SUB / "rf_preprocess.json").write_text(json.dumps(prep))

    left, right, feat, thr, val, roots = [], [], [], [], [], []
    offset = 0
    for est in model.estimators_:
        t = est.tree_
        roots.append(offset)
        l, r = t.children_left.copy(), t.children_right.copy()
        l[l != -1] += offset
        r[r != -1] += offset
        v = t.value[:, 0, :]
        left.append(l); right.append(r); feat.append(t.feature); thr.append(t.threshold)
        val.append(v / v.sum(axis=1, keepdims=True))  # class probabilities at each node
        offset += t.node_count
    np.savez_compressed(
        SUB / "rf_forest.npz",
        left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
        feature=np.concatenate(feat).astype(np.int32), threshold=np.concatenate(thr).astype(np.float64),
        value=np.concatenate(val).astype(np.float64), roots=np.array(roots, dtype=np.int32))
    shutil.copy(ROOT / "code" / "src" / "pred.py", SUB / "pred.py")
    print(f"\nExported {len(model.estimators_)} trees ({offset} nodes) to {SUB}")


def check_pred(model, fe, raw_rows):
    """pred.py (numpy only) must agree with sklearn on raw, uncleaned rows."""
    sys.path.insert(0, str(SUB))
    sys.dont_write_bytecode = True  # keep submission/ to exactly the three files
    import pred
    pred._MODEL = None
    with tempfile.TemporaryDirectory() as tmp:
        csv = f"{tmp}/check.csv"
        raw_rows.drop(columns=raw_rows.columns[-1]).to_csv(csv, index=False)  # no label column
        t0 = time.time()
        ours = np.array(pred.predict_all(csv))
        secs = time.time() - t0
    ref = model.predict(fe.transform(clean(load_raw_like(raw_rows))))
    agree = (ours == ref).mean()
    size = sum(f.stat().st_size for f in SUB.iterdir()) / 1e6
    print(f"pred.py vs sklearn: {agree:.1%} identical on {len(ours)} rows, {secs:.2f}s, submission {size:.2f} MB")
    assert agree == 1.0, "pred.py disagrees with sklearn - do not submit; paste this output to Claude"
    assert size < 10, "submission is over 10 MB"


def load_raw_like(raw_rows):
    df = raw_rows.copy()
    df.columns = load_raw().columns[: len(df.columns)]
    return df


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "val"
    train, val, test = get_split()

    if mode == "val":
        fe, model = fit(train)
        pred_val = model.predict(fe.transform(val))
        res = report("validation", val["label"].to_numpy(), pred_val)
        show_errors(val, pred_val)
        (OUT / "val_results.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"\nSaved to {OUT / 'val_results.json'}. Share accuracy and macro-F1 with the team.")

    elif mode == "final":
        if (OUT / "test_results.json").exists() and "--again" not in sys.argv:
            sys.exit("The test set was already used (results/rf/test_results.json exists). "
                     "Re-scoring it would make the estimate optimistic. Add --again only if you must.")
        # a) one-time test estimate: train on train + validation, as in the proposal
        fe, model = fit(pd.concat([train, val], ignore_index=True))
        res = report("test", test["label"].to_numpy(), model.predict(fe.transform(test)))
        (OUT / "test_results.json").write_text(json.dumps(res, indent=1, default=str))

        # b) final model for submission: all 870 rows
        all_rows = pd.concat([train, val, test], ignore_index=True)
        fe_all, model_all = fit(all_rows)
        export(fe_all, model_all)

        # c) check pred.py on the original CSV text (validation + test students)
        raw = pd.read_csv(DATA_PATH)
        ids = set(val["id"]) | set(test["id"])
        check_pred(model_all, fe_all, raw[raw.iloc[:, 0].isin(ids)].reset_index(drop=True))
        print("\nSubmit these three files from submission/: pred.py, rf_preprocess.json, rf_forest.npz")
    else:
        sys.exit("usage: python rf_final.py [val|final]")
