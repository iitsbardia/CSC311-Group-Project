"""Feature builder for the random forest, following the team proposal's preprocessing.

numpy/pandas/stdlib only (no sklearn), so pred.py can import it unchanged.
All statistics (medians, pay clip, vocabularies, IDF) are learned in `fit` from
training rows only and reused in `transform`.

Columns produced (in order):
  ease, strategy, injury, individual        ordinal / numeric, median-imputed
  log_pay, pay_missing                      parsed pay, clipped at train 99th pct, log1p
  song_missing                              blank OR non-answer ("idk", "no particular song", ...)
  where=*, seasons=*, music=*               multi-hot, minus Summer and Outdoor public space
  food:*, colour:*, song:*, feel:*          per-field TF-IDF (min_df=2, smooth idf, l2 norm)
"""

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from data import (LIKERT_COLS, MULTI_COLS, PAY_COL, TEXT_COLS, clean_text, likert_to_ordinal,
                  parse_pay, split_multi)

# Near-constant options dropped in the proposal (91-92% / 83-91% of every sport).
DROP_OPTIONS = {"seasons": {"Summer"}, "where": {"Outdoor public space"}}

# Same stopword list and token rule as the exploration notebook (section 12),
# which gives the proposal's 528-word field-prefixed vocabulary at min_df=2.
STOPWORDS = set("""
a an the and or of to in on for with at by from is are was were be been it its this that
i im i'm my me we our you your he she they them his her their as so but if not no very
just really like feel feeling feels some any all more most much too also than then there
when what who how which would will can could do does did am have has had get got s t
because colour color colours colors know
""".split())

# Song non-answers counted as missing (notebook section 17 / proposal).
SONG_NON_ANSWER = re.compile(
    r"no particular|don.?t know|idk|not sure|no song|nothing|none|n/a|no idea|can.?t think")

MIN_DF = 2
PAY_CLIP_Q = 0.99


def tokens(value):
    return [t for t in clean_text(value).split() if t not in STOPWORDS and len(t) > 1]


def song_missing(value):
    if not isinstance(value, str):
        return 1.0
    return 1.0 if SONG_NON_ANSWER.search(value.lower().strip()) else 0.0


def is_blank_row(df):
    """Rows where the student left the whole survey empty (dropped from training only)."""
    cols = ["ease", *LIKERT_COLS, *MULTI_COLS]
    return df[cols].isna().all(axis=1).to_numpy()


class RFFeatures:
    def fit(self, df):
        """Learn every preprocessing statistic from (cleaned) training rows."""
        p = {}
        p["medians"] = {"ease": float(df["ease"].median())}
        for c in LIKERT_COLS:
            p["medians"][c] = float(df[c].map(likert_to_ordinal).median())

        pay = df[PAY_COL].map(parse_pay)
        p["pay_clip"] = float(pay.quantile(PAY_CLIP_Q))
        p["pay_median"] = float(np.log1p(pay.clip(upper=p["pay_clip"])).median())

        p["options"] = {}
        for c in MULTI_COLS:
            seen = sorted({o for v in df[c] for o in split_multi(v)})
            p["options"][c] = [o for o in seen if o not in DROP_OPTIONS.get(c, set())]

        p["vocab"], p["idf"] = {}, {}
        n = len(df)
        for c in TEXT_COLS:
            dfreq = Counter(t for v in df[c] for t in set(tokens(v)))
            vocab = sorted(t for t, k in dfreq.items() if k >= MIN_DF)
            p["vocab"][c] = vocab
            # sklearn-style smooth idf: ln((1+n)/(1+df)) + 1
            p["idf"][c] = [float(np.log((1 + n) / (1 + dfreq[t])) + 1) for t in vocab]
        self.p = p
        self.feature_names = self._names()
        return self

    def _names(self):
        p = self.p
        names = ["ease", *LIKERT_COLS, "log_pay", "pay_missing", "song_missing"]
        for c in MULTI_COLS:
            names += [f"{c}={o}" for o in p["options"][c]]
        for c in TEXT_COLS:
            names += [f"{c}:{t}" for t in p["vocab"][c]]
        return names

    def transform(self, df):
        p = self.p
        blocks = []

        num = np.column_stack(
            [df["ease"].astype(float).fillna(p["medians"]["ease"]).to_numpy()]
            + [df[c].map(likert_to_ordinal).astype(float).fillna(p["medians"][c]).to_numpy()
               for c in LIKERT_COLS])
        blocks.append(num)

        pay = df[PAY_COL].map(parse_pay).astype(float)
        pay_missing = pay.isna().to_numpy(dtype=float)
        log_pay = np.log1p(pay.clip(upper=p["pay_clip"])).fillna(p["pay_median"]).to_numpy()
        song_miss = df["song"].map(song_missing).to_numpy(dtype=float)
        blocks.append(np.column_stack([log_pay, pay_missing, song_miss]))

        for c in MULTI_COLS:
            opts = p["options"][c]
            idx = {o: j for j, o in enumerate(opts)}
            m = np.zeros((len(df), len(opts)))
            for i, v in enumerate(df[c]):
                for o in split_multi(v):
                    if o in idx:
                        m[i, idx[o]] = 1.0
            blocks.append(m)

        for c in TEXT_COLS:
            vocab = p["vocab"][c]
            idx = {t: j for j, t in enumerate(vocab)}
            idf = np.asarray(p["idf"][c])
            m = np.zeros((len(df), len(vocab)))
            for i, v in enumerate(df[c]):
                for t in tokens(v):  # words never seen in training are ignored
                    if t in idx:
                        m[i, idx[t]] += 1.0
            m *= idf
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            blocks.append(np.divide(m, norms, out=np.zeros_like(m), where=norms > 0))

        return np.hstack(blocks)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.p, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path) as f:
            obj.p = json.load(f)
        obj.feature_names = obj._names()
        return obj
