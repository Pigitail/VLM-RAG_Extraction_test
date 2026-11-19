"""Normalisation and BM25: hand-computed values."""

from __future__ import annotations

import math

from vlmrag.evaluation.bm25 import BM25, tokenize
from vlmrag.evaluation.textnorm import exact_match, mentions, normalize_value, normalized_match


def test_normalize_unifies_dashes_quotes_case_and_whitespace():
    assert normalize_value("ISO 2768–m") == "iso 2768-m"
    assert normalize_value("ISO 2768−M ") == "iso 2768-m"  # NBSP and minus sign
    assert normalize_value("“Draft”  ‘x’\n\tY") == "\"draft\" 'x' y"
    assert normalize_value(None) == ""
    # casefold, not lower: German sharp s folds to "ss"
    assert normalize_value("STRASSE") == normalize_value("straße")


def test_normalize_changes_nothing_else():
    assert normalize_value("0.42 kg") != normalize_value("0.42kg")
    assert normalize_value("A|B") == "a|b"


def test_exact_vs_normalized_match():
    assert exact_match("C", "C") and not exact_match("c", "C")
    assert exact_match(None, "")
    assert normalized_match("  matte—black ", "MATTE-BLACK")
    assert not normalized_match("", "x")


def test_mentions_is_normalised_substring():
    text = "The dialog asks “Release order PO-1050 to the vendor?”"
    assert mentions(text, "release order po‑1050 to the VENDOR?")
    assert not mentions(text, "PO-1051")
    assert not mentions(text, "   ")  # an empty fact is never mentioned
    assert mentions("total 2,971.50 USD", "2,971.50")


def test_tokenize_nfkc_casefold_word_runs():
    assert tokenize("EX-L300-1004 Ｌ３００") == ["ex", "l300", "1004", "l300"]
    assert tokenize("") == []
    assert tokenize("공급업체 V-2864") == ["공급업체", "v", "2864"]


def test_bm25_matches_the_formula_by_hand():
    index = BM25(["a b", "a a c"])
    assert index.n == 2 and index.avgdl == 2.5
    idf_a, idf_b = math.log(1.2), math.log(2.0)
    assert math.isclose(index.idf("a"), idf_a) and math.isclose(index.idf("b"), idf_b)
    # tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl)), k1 = 1.2, b = 0.75
    doc0 = 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.5))
    doc1 = 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.5))
    scores = index.scores("a")
    assert math.isclose(scores[0], idf_a * doc0) and math.isclose(scores[1], idf_a * doc1)
    # A repeated query term counts once.
    both = index.scores("a a B")
    assert math.isclose(both[0], idf_a * doc0 + idf_b * doc0)
    assert math.isclose(both[1], idf_a * doc1)


def test_bm25_edge_cases():
    assert BM25([]).scores("anything") == []
    assert BM25(["", ""]).scores("a") == [0.0, 0.0]
    assert BM25(["a b"]).scores("zzz") == [0.0]
    # idf stays positive for a term in every document (the "+1" variant)
    assert BM25(["a", "a", "a"]).scores("a")[0] > 0
