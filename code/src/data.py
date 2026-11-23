"""Shared data loading, cleaning helpers and the respondent-grouped split.

Every other script imports from here so that the split and the parsing rules
are identical across exploration, training and pred.py.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root (this file is code/src/data.py)
DATA_PATH = ROOT / "data" / "csc311_project_training_202609.csv"
SPLIT_PATH = ROOT / "data" / "split_team.json"

# Team split (matches the shared Colab notebook CSC311_Proposal.ipynb): students
# grouped by id, 70/15/15 train/val/test with sklearn train_test_split, seed 42.
SEED = 42
VAL_TEST_FRAC = 0.30
TEST_FRAC = 0.15
N_FOLDS = 5

# Short names for the 13 columns, in file order.
COLUMNS = [
    "id", "ease", "strategy", "injury", "individual", "where", "seasons",
    "music", "food", "colour", "song", "feel", "pay", "label",
]
LABELS = ["Baseball", "Basketball", "Soccer"]

NUMERIC_COLS = ["ease"]
LIKERT_COLS = ["strategy", "injury", "individual"]
MULTI_COLS = ["where", "seasons", "music"]
TEXT_COLS = ["food", "colour", "song", "feel"]
PAY_COL = "pay"

LIKERT_MAP = {
    "strongly disagree": 1, "disagree": 2, "neutral": 3,
    "agree": 4, "strongly agree": 5,
}

# Strings that mean "no answer"; compared after strip + lowercase.
MISSING_MARKERS = {"", "n/a", "na", "nan", "none", "null", "-", "--", "#name?", "?"}

# Survey masks names as "[this sport]", "[a competition]", "[a team in this sport]", ...;
# drop every bracketed placeholder so models can't key on the masking itself.
PLACEHOLDER_RE = re.compile(r"\[+[^\]]*\]+")


def load_raw(path=DATA_PATH):
    """Read the survey CSV and rename columns to short names.

    Works for the test CSV too (no label column): columns are renamed by position.
    """
    df = pd.read_csv(path)
    df.columns = COLUMNS[: len(df.columns)]
    return df


def normalize_missing(value):
    """Map every form of 'no answer' to np.nan; strip whitespace otherwise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).replace("\xa0", " ").strip()
    return np.nan if s.lower() in MISSING_MARKERS else s


def clean(df):
    """Return a copy with missing values normalized in every question column."""
    out = df.copy()
    # For pay, a bare "none" means "would pay nothing", not "no answer".
    if PAY_COL in out:
        out[PAY_COL] = out[PAY_COL].map(
            lambda v: "0" if isinstance(v, str) and v.strip().lower() == "none" else v)
    for col in COLUMNS[1:-1]:
        if col in out:
            out[col] = out[col].map(normalize_missing)
    out["ease"] = pd.to_numeric(out["ease"], errors="coerce")
    return out


def split_multi(value):
    """'Winter,Summer' -> ['Winter', 'Summer']; missing -> []."""
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def likert_to_ordinal(value):
    if not isinstance(value, str):
        return np.nan
    return LIKERT_MAP.get(value.strip().lower(), np.nan)


_ZERO_RE = re.compile(
    r"^none\b|\b(zero|nothing|unwilling|not willing|no willing|wouldn'?t|would not|not pay|"
    r"free|not really|pay me)\b")


def parse_pay(value):
    """Turn a willingness-to-pay answer into a float (CAD), or np.nan if unparseable.

    Rules: take the first number; a range 'a-b' / 'a to b' uses its midpoint;
    a trailing 'k' multiplies by 1000; refusals ('not willing', 'zero') are 0.
    """
    if not isinstance(value, str):
        return np.nan
    s = value.lower().replace(",", "")
    s = re.sub(r"(\d) (\d{3})\b", r"\1\2", s)  # "10 000" -> "10000"
    if "scale" in s:  # a 1-10 rating, not a dollar amount
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
    """Lowercase free text, drop the sport placeholders, keep letters/digits."""
    if not isinstance(value, str):
        return ""
    s = PLACEHOLDER_RE.sub(" ", value.lower())
    return re.sub(r"[^a-z0-9']+", " ", s).strip()


def make_split(df, seed=SEED):
    """Team split -> (train_ids, val_ids, test_ids).

    Reproduces the shared notebook exactly: ids in file order, 30% of students set
    aside, then int(15% of all students) of those become test and the rest val.
    """
    from sklearn.model_selection import train_test_split

    student_ids = df["id"].unique()
    train_ids, remaining_ids = train_test_split(
        student_ids, test_size=VAL_TEST_FRAC, random_state=seed)
    n_test = int(TEST_FRAC * len(student_ids))
    val_ids, test_ids = train_test_split(remaining_ids, test_size=n_test, random_state=seed)
    return [sorted(int(i) for i in x) for x in (train_ids, val_ids, test_ids)]


def get_split(df=None):
    """Load the saved split (creating it on first use) -> (train_df, val_df, test_df).

    train_df is for exploration, CV and tuning; val_df compares finalists;
    test_df is touched once, for the final estimate.
    """
    if df is None:
        df = clean(load_raw())
    if SPLIT_PATH.exists():
        saved = json.loads(SPLIT_PATH.read_text())
    else:
        train_ids, val_ids, test_ids = make_split(df)
        saved = {"seed": SEED, "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
        SPLIT_PATH.parent.mkdir(exist_ok=True)
        SPLIT_PATH.write_text(json.dumps(saved, indent=1))
    return tuple(df[df["id"].isin(saved[k])].reset_index(drop=True)
                 for k in ("train_ids", "val_ids", "test_ids"))


def group_folds(train_df, n_folds=N_FOLDS):
    """GroupKFold fold index (0..n_folds-1) per row of train_df, grouped by id."""
    from sklearn.model_selection import GroupKFold

    folds = np.empty(len(train_df), dtype=int)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=n_folds).split(train_df, groups=train_df["id"])):
        folds[idx] = k
    return folds


if __name__ == "__main__":
    for name, part in zip(["train", "val", "test"], get_split()):
        print(f"{name}: {len(part)} rows / {part.id.nunique()} ids")
