"""CSC311 project prediction script: random forest, numpy/pandas/stdlib only.

Needs two files next to it (written by rf_final.py):
  rf_preprocess.json   preprocessing statistics learned from the training data
  rf_forest.npz        every tree of the trained forest as flat arrays

Usage:  from pred import predict_all;  predict_all("test.csv") -> list of labels
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cleaning rules: copied from the team's data.py and rf_features.py so this
# file has no project imports. Keep them in sync if those files change.
# ---------------------------------------------------------------------------
COLUMNS = ["id", "ease", "strategy", "injury", "individual", "where", "seasons",
           "music", "food", "colour", "song", "feel", "pay", "label"]
# Keywords to recognise each question in the CSV header (fallback: column position).
HEADER_KEYS = {
    "ease": "how easy", "strategy": "strategy and planning", "injury": "risk of injury",
    "individual": "individual brilliance", "where": "where might", "seasons": "which seasons",
    "music": "types of music", "food": "foods and drinks", "colour": "what colour",
    "song": "which song", "feel": "how do you feel", "pay": "willing to pay",
}
LIKERT_COLS = ["strategy", "injury", "individual"]
MULTI_COLS = ["where", "seasons", "music"]
TEXT_COLS = ["food", "colour", "song", "feel"]
LIKERT_MAP = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}
MISSING_MARKERS = {"", "n/a", "na", "nan", "none", "null", "-", "--", "#name?", "?"}
PLACEHOLDER_RE = re.compile(r"\[+[^\]]*\]+")
_ZERO_RE = re.compile(
    r"^none\b|\b(zero|nothing|unwilling|not willing|no willing|wouldn'?t|would not|not pay|"
    r"free|not really|pay me)\b")
SONG_NON_ANSWER = re.compile(
    r"no particular|don.?t know|idk|not sure|no song|nothing|none|n/a|no idea|can.?t think")
STOPWORDS = set("""
a an the and or of to in on for with at by from is are was were be been it its this that
i im i'm my me we our you your he she they them his her their as so but if not no very
just really like feel feeling feels some any all more most much too also than then there
when what who how which would will can could do does did am have has had get got s t
because colour color colours colors know
""".split())


def load_csv(path):
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        low = str(col).lower()
        for short, key in HEADER_KEYS.items():
            if key in low and short not in rename.values():
                rename[col] = short
                break
    if len(rename) == len(HEADER_KEYS):
        return df.rename(columns=rename)
    df.columns = COLUMNS[: len(df.columns)]  # same positional rule as data.load_raw
    return df


def normalize_missing(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).replace("\xa0", " ").strip()
    return np.nan if s.lower() in MISSING_MARKERS else s


def clean(df):
    out = df.copy()
    out["pay"] = out["pay"].map(
        lambda v: "0" if isinstance(v, str) and v.strip().lower() == "none" else v)
    for col in HEADER_KEYS:
        out[col] = out[col].map(normalize_missing)
    out["ease"] = pd.to_numeric(out["ease"], errors="coerce")
    return out


def split_multi(value):
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def likert_to_ordinal(value):
    if not isinstance(value, str):
        return np.nan
    return LIKERT_MAP.get(value.strip().lower(), np.nan)


def parse_pay(value):
    if not isinstance(value, str):
        return np.nan
    s = value.lower().replace(",", "")
    s = re.sub(r"(\d) (\d{3})\b", r"\1\2", s)
    if "scale" in s:
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return 0.0 if _ZERO_RE.search(s) else np.nan
    x = float(nums[0])
    if len(nums) > 1 and re.search(r"\d\s*\$?\s*(-|–|to)\s*\$?\d", s):
        x = (float(nums[0]) + float(nums[1])) / 2
    if re.search(r"\d\s*k\b", s):
        x *= 1000
    return x


def clean_text(value):
    if not isinstance(value, str):
        return ""
    s = PLACEHOLDER_RE.sub(" ", value.lower())
    return re.sub(r"[^a-z0-9']+", " ", s).strip()


def tokens(value):
    return [t for t in clean_text(value).split() if t not in STOPWORDS and len(t) > 1]


def song_missing(value):
    if not isinstance(value, str):
        return 1.0
    return 1.0 if SONG_NON_ANSWER.search(value.lower().strip()) else 0.0


# ---------------------------------------------------------------------------
# Features: same columns, same order as RFFeatures.transform in rf_features.py
# ---------------------------------------------------------------------------
def make_features(df, p):
    blocks = []
    num = np.column_stack(
        [df["ease"].astype(float).fillna(p["medians"]["ease"]).to_numpy()]
        + [df[c].map(likert_to_ordinal).astype(float).fillna(p["medians"][c]).to_numpy()
           for c in LIKERT_COLS])
    blocks.append(num)

    pay = df["pay"].map(parse_pay).astype(float)
    pay_missing = pay.isna().to_numpy(dtype=float)
    log_pay = np.log1p(pay.clip(upper=p["pay_clip"])).fillna(p["pay_median"]).to_numpy()
    song_miss = df["song"].map(song_missing).to_numpy(dtype=float)
    blocks.append(np.column_stack([log_pay, pay_missing, song_miss]))

    for c in MULTI_COLS:
        idx = {o: j for j, o in enumerate(p["options"][c])}
        m = np.zeros((len(df), len(idx)))
        for i, v in enumerate(df[c]):
            for o in split_multi(v):
                if o in idx:
                    m[i, idx[o]] = 1.0
        blocks.append(m)

    for c in TEXT_COLS:
        idx = {t: j for j, t in enumerate(p["vocab"][c])}
        m = np.zeros((len(df), len(idx)))
        for i, v in enumerate(df[c]):
            for t in tokens(v):  # unseen words are ignored
                if t in idx:
                    m[i, idx[t]] += 1.0
        m *= np.asarray(p["idf"][c])
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        blocks.append(np.divide(m, norms, out=np.zeros_like(m), where=norms > 0))
    return np.hstack(blocks)


# ---------------------------------------------------------------------------
# Random forest: walk every tree, average the leaf class probabilities
# ---------------------------------------------------------------------------
def forest_proba(X, f):
    # the trained trees compare float32 copies of the features with float64 thresholds
    X = X.astype(np.float32).astype(np.float64)
    rows = np.arange(len(X))
    proba = np.zeros((len(X), f["value"].shape[1]))
    for root in f["roots"]:
        node = np.full(len(X), root)
        while True:
            inner = f["left"][node] != -1          # -1 marks a leaf
            if not inner.any():
                break
            n = node[inner]
            go_left = X[rows[inner], f["feature"][n]] <= f["threshold"][n]
            node[inner] = np.where(go_left, f["left"][n], f["right"][n])
        proba += f["value"][node]
    return proba / len(f["roots"])


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        prep = json.loads((HERE / "rf_preprocess.json").read_text())
        forest = dict(np.load(HERE / "rf_forest.npz", allow_pickle=False))
        _MODEL = (prep, forest)
    return _MODEL


def predict_all(filename):
    """Return a list with one predicted sport per row of the CSV file."""
    prep, forest = _load_model()
    df = clean(load_csv(filename))
    proba = forest_proba(make_features(df, prep), forest)
    classes = prep["classes"]
    return [classes[k] for k in proba.argmax(axis=1)]


if __name__ == "__main__":
    for label in predict_all(sys.argv[1]):
        print(label)
