"""A small, dependency-free Okapi BM25 index for the retrieval metrics.

Why BM25 and not embeddings
    The retrieval metric has to be reproducible by anyone without an
    embedding endpoint and has to change only when the *chunk text* changes.
    BM25 is lexical, deterministic and has no model behind it, so a gain in
    recall can be attributed to what the VLM added to ``retrieve_content``.
    It understates what an embedding retriever would find for paraphrased
    queries; the evaluation report says so.

Definition (Robertson/Sparck Jones idf with the "+1" used by Lucene, which keeps
idf positive for terms that occur in more than half of the chunks)::

    tokens(t)   = re.findall(r"\\w+", NFKC(t).casefold())
    idf(q)      = ln(1 + (N - df(q) + 0.5) / (df(q) + 0.5))
    score(D, Q) = sum over DISTINCT query terms q of
                  idf(q) * tf(q, D) * (k1 + 1) / (tf(q, D) + k1 * (1 - b + b * |D| / avgdl))

with ``k1 = 1.2`` and ``b = 0.75``. A repeated query term counts once.
Scores are returned in index order; ranking and tie handling belong to the
metric (:func:`vlmrag.evaluation.metrics.retrieval_metrics`).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Sequence

__all__ = ["BM25", "tokenize", "K1", "B"]

K1 = 1.2
B = 0.75
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """NFKC, casefold, then Unicode word-character runs (letters, digits, underscore)."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text or "").casefold())


class BM25:
    """An in-memory BM25 index over a fixed list of texts (one entry per chunk)."""

    def __init__(self, texts: Sequence[str], *, k1: float = K1, b: float = B) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: Counter = Counter()
        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._len.append(len(tokens))
            df.update(counts.keys())
        self.n = len(self._tf)
        self.avgdl = (sum(self._len) / self.n) if self.n else 0.0
        self.df: dict[str, int] = dict(df)

    def __len__(self) -> int:
        return self.n

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """BM25 score of every indexed text for ``query`` (0.0 when no query term occurs)."""
        out = [0.0] * self.n
        if not self.n or self.avgdl <= 0:
            return out
        for term in dict.fromkeys(tokenize(query)):  # distinct terms, first-seen order
            if term not in self.df:
                continue
            idf = self.idf(term)
            for i, counts in enumerate(self._tf):
                tf = counts.get(term, 0)
                if tf:
                    norm = self.k1 * (1.0 - self.b + self.b * self._len[i] / self.avgdl)
                    out[i] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out
