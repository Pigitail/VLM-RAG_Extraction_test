from __future__ import annotations

import hashlib
import io
import zipfile

from vlmrag.synthetic import GENERATORS, load_generator
from vlmrag.synthetic.slides import DECK_STEM, generate, presentation_skeleton
from vlmrag.techniques.slides.layout import read_deck


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_generator():
    assert GENERATORS["slides"] == "vlmrag.synthetic.slides:generate"
    assert load_generator("slides") is generate


def _written(directory):
    return [directory / f"{DECK_STEM}.pptx", directory / f"{DECK_STEM}.truth.json"]


def test_generation_is_deterministic(tmp_path):
    assert generate(tmp_path / "a", seed=0) == [tmp_path / "a" / f"{DECK_STEM}.pptx"]  # documents only
    generate(tmp_path / "b", seed=0)
    first, second = _written(tmp_path / "a"), _written(tmp_path / "b")
    assert all(p.is_file() for p in first)
    assert [_sha(p) for p in first] == [_sha(p) for p in second]
    generate(tmp_path / "c", seed=7)
    assert _sha(_written(tmp_path / "c")[1]) != _sha(first[1])  # the seed changes the screen values


def test_zip_entries_have_fixed_timestamps(tmp_path):
    [deck] = generate(tmp_path, seed=0)
    with zipfile.ZipFile(deck) as archive:
        assert {info.date_time for info in archive.infolist()} == {(2026, 1, 1, 0, 0, 0)}
    with zipfile.ZipFile(io.BytesIO(presentation_skeleton())) as archive:
        assert "ppt/presentation.xml" in archive.namelist()


def test_truth_structure(synthetic_deck):
    deck_path, truth = synthetic_deck
    assert truth["source"] == deck_path.name and truth["doc_type"] == "slides"
    assert truth["slide_count"] == 6 and [s["slide"] for s in truth["slides"]] == [1, 2, 3, 4, 5, 6]
    roles = [b["image_role"] for s in truth["slides"] for b in s["image_blocks"]]
    assert set(roles) == {"main_screenshot", "detail_crop", "inline_image", "ui_icon"}
    assert 3 <= len(truth["queries"]) <= 6
    picture_slides = {s["slide"] for s in truth["slides"] if s["image_blocks"]}
    for query in truth["queries"]:
        assert query["query"] and query["answer"]
        for relevant in query["relevant"]:
            assert relevant["source"] == deck_path.name and relevant["page"] in picture_slides
    for slide in truth["slides"]:
        assert slide["expects_visual_call"] == bool(slide["image_blocks"])
        for block in slide["image_blocks"]:
            assert block["description"] and block["description_en"] and block["description_ko"]


def test_visual_facts_exist_only_in_pixels(synthetic_deck):
    """Queries must be answerable only through the visual descriptions, never from slide XML text."""
    deck_path, truth = synthetic_deck
    deck = read_deck(deck_path)
    all_text = "\n".join(s.text_only + "\n" + s.text_and_table for s in deck.slides).lower()
    for query in truth["queries"]:
        assert query["answer"].lower() not in all_text, query["answer"]
    for slide_truth in truth["slides"]:
        slide = deck.slides[slide_truth["slide"] - 1]
        text = (slide.text_only + "\n" + slide.text_and_table).lower()
        for fact in slide_truth["visual_facts"] + slide_truth["must_mention"]:
            assert fact.lower() not in text, (slide_truth["slide"], fact)
