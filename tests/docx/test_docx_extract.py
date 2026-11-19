"""End-to-end ``extract()`` on the synthetic specification, plus failure paths.

Most tests replace the LibreOffice conversion with a reportlab PDF of the
right page count (``fake_pdf``): scripted answers are keyed by page number,
so pixels do not matter and the tests run everywhere. The ``soffice``-marked
test runs the real conversion.
"""

from __future__ import annotations

import json

import pytest

from vlmrag.config import Settings
from vlmrag.render.office import ConversionError
from vlmrag.techniques import load_technique
from vlmrag.techniques.docx import DocxExtractionError, extract, pipeline
from vlmrag.vlm.backends import NullBackend, RecordingBackend, ReplayBackend
from vlmrag.vlm.client import VLMClient

IMAGE = "docx.image_description"
TABLE = "docx.table_distribution"


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="null")


def by_page(chunks):
    return {c.page: c for c in chunks}


def test_registry_resolves_the_technique():
    assert load_technique("docx") is extract


def test_end_to_end_with_truth_answers(synthetic_docx, settings, scripted_client, truth_by_task, fake_pdf):
    path, truth = synthetic_docx
    conversions = fake_pdf(truth["page_count"])
    client = scripted_client(by_task=truth_by_task(truth))
    chunks = extract(path, client, settings)

    assert conversions == [path.name]
    assert [c.page for c in chunks] == list(range(1, truth["page_count"] + 1))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    pages = by_page(chunks)
    for chunk in chunks:
        assert chunk.doc_type == "docx" and chunk.source_name == path.name and len(chunk.doc_id) == 16
        assert chunk.filtering_content and not any(ch.isspace() for ch in chunk.filtering_content)
        assert chunk.llm_content.startswith(f"# {path.stem} - page {chunk.page}")
        assert chunk.provenance["fallbacks"] == []
        assert chunk.provenance["model"] == "scripted" and chunk.provenance["render"]["dpi"] == 220
        assert chunk.metadata["page_split"] == "rendered"

    # Visual facts land in retrieve_content of the page that shows them.
    for page in truth["pages"]:
        text = pages[page["page"]].retrieve_content.lower()
        for fact in page["must_mention"]:
            assert fact.lower() in text
        if page["figures"]:
            assert "[visual content]" in text
            assert pages[page["page"]].provenance["prompt_versions"][IMAGE] == "docx.image_description/1"
    for query in truth["queries"]:
        for relevant in query["relevant"]:
            assert "[Visual content]" in pages[relevant["page"]].retrieve_content

    # Every table row is on exactly the page the truth says, with the validated title.
    table = truth["tables"][0]
    placed = {}
    for chunk in chunks:
        for entry in chunk.metadata["multi_page_tables"]:
            assert entry["assignment"] == "vlm" and entry["title"] == table["title"]
            for index in entry["rows"]:
                placed[index] = chunk.page
    assert [placed[i] for i in range(table["row_count"])] == table["row_pages"]
    end = pages[table["end_page"]]
    assert f"(Table continued from page {table['start_page']}: {table['title']})" in end.retrieve_content
    assert table["title"] in end.metadata["table_titles"]

    calls = client.backend.calls
    image_pages = sorted({img["page"] for img in truth["images"]})
    assert sorted(c.page for c in calls if c.task == IMAGE) == image_pages
    assert [c.page for c in calls if c.task == TABLE] == [table["start_page"]]
    assert all(c.source_name == path.name for c in calls)


def test_valid_answer_that_differs_from_the_markers_wins(synthetic_docx, settings, scripted_client, truth_by_task,
                                                         truth_answers, fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    table = truth["tables"][0]
    moved = truth_answers["table"](truth)
    start, end = str(table["start_page"]), str(table["end_page"])
    moved["row_distribution"][end].insert(0, moved["row_distribution"][start].pop())  # last start-page row moves
    answers = truth_by_task(truth)
    answers[TABLE] = moved
    chunks = extract(path, scripted_client(by_task=answers), settings)
    end_rows = [e["rows"] for e in by_page(chunks)[table["end_page"]].metadata["multi_page_tables"]][0]
    assert end_rows[0] == moved["row_distribution"][end][0]
    assert by_page(chunks)[table["end_page"]].metadata["multi_page_tables"][0]["assignment"] == "vlm"


@pytest.mark.parametrize(
    "table_answer, reason",
    [
        ("Rows 0-20 are on the first page, the rest on the second.", "table_distribution_unparseable"),
        ("", "table_distribution_unparseable"),
        ({"has_title": False, "title_text": "", "title_source": "none", "title_confidence": "low",
          "row_distribution": {"1": [0]}}, "table_distribution_invalid:page_out_of_range"),
        ("missing_rows", "table_distribution_invalid:row_set_mismatch"),
        ("{}", "table_distribution_invalid:row_distribution_not_an_object"),
    ],
)
def test_bad_table_answers_fall_back_to_row_markers(synthetic_docx, settings, scripted_client, truth_by_task,
                                                    truth_answers, fake_pdf, table_answer, reason):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    table = truth["tables"][0]
    if table_answer == "missing_rows":
        table_answer = truth_answers["table"](truth)
        table_answer["row_distribution"][str(table["end_page"])].pop()
    answers = truth_by_task(truth)
    answers[TABLE] = table_answer
    client = scripted_client(by_task=answers)
    chunks = extract(path, client, settings)

    placed = {}
    for chunk in chunks:
        for entry in chunk.metadata["multi_page_tables"]:
            assert entry["assignment"] == "xml_row_markers" and entry["title"] == ""
            placed.update({i: chunk.page for i in entry["rows"]})
            assert any(f.startswith(f"table_rows_from_xml_markers:t1:{reason}") for f in chunk.provenance["fallbacks"])
    assert [placed[i] for i in range(table["row_count"])] == table["row_pages"]
    table_calls = [c for c in client.backend.calls if c.task == TABLE]
    expected_attempts = 2 if "unparseable" in reason else 1  # one re-ask for unparseable output
    assert len(table_calls) == expected_attempts
    # The call record uses the shared status vocabulary; the detail is in "reason".
    [record] = [c for c in by_page(chunks)[table["start_page"]].provenance["calls"] if c["task"] == TABLE]
    if "invalid:" in reason:
        assert record["status"] == "invalid" and record["reason"].startswith(reason.split("invalid:", 1)[1])
    else:
        assert record["status"] == "unparseable" and "reason" not in record


def test_hallucinated_title_is_dropped_but_rows_are_used(synthetic_docx, settings, scripted_client, truth_by_task,
                                                         truth_answers, fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    answer = truth_answers["table"](truth)
    answer["title_text"] = "Approved suppliers"
    answers = truth_by_task(truth)
    answers[TABLE] = answer
    chunks = extract(path, scripted_client(by_task=answers), settings)
    table = truth["tables"][0]
    start = by_page(chunks)[table["start_page"]]
    assert start.metadata["multi_page_tables"][0]["assignment"] == "vlm"
    assert start.metadata["table_titles"] == []
    assert "table_title_rejected:t1" in start.provenance["fallbacks"]


def test_null_backend_keeps_all_text_and_uses_fallbacks(synthetic_docx, settings, fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    chunks = extract(path, VLMClient(NullBackend()), settings)
    assert [c.page for c in chunks] == list(range(1, truth["page_count"] + 1))
    for chunk in chunks:
        assert "[Visual content]" not in chunk.retrieve_content
        assert chunk.metadata["visual_descriptions"] == []
    image_page = truth["images"][0]["page"]
    assert "image_description_invalid" in by_page(chunks)[image_page].provenance["fallbacks"]
    rows = sum(len(e["rows"]) for c in chunks for e in c.metadata["multi_page_tables"])
    assert rows == truth["tables"][0]["row_count"]


def test_empty_and_malformed_image_answers(synthetic_docx, settings, scripted_client, truth_by_task, fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    pages = sorted({img["page"] for img in truth["images"]})
    replies = {pages[0]: "", pages[1]: "```json\n{not json}\n```", pages[2]: {"image_descriptions": [], "notes": ""}}
    answers = truth_by_task(truth)
    answers[IMAGE] = lambda request: replies[request.page]
    chunks = by_page(extract(path, scripted_client(by_task=answers), settings))
    assert "image_description_unparseable" in chunks[pages[0]].provenance["fallbacks"]
    assert "image_description_unparseable" in chunks[pages[1]].provenance["fallbacks"]
    assert not any(f.startswith("image_description") for f in chunks[pages[2]].provenance["fallbacks"])
    assert all("[Visual content]" not in c.retrieve_content for c in chunks.values())


def test_backend_errors_degrade_gracefully(synthetic_docx, settings, scripted_client, fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])

    class BadRequest(Exception):
        status_code = 400

    def explode(request):
        raise BadRequest("content filter")

    client = scripted_client(explode)
    chunks = extract(path, client, settings)
    assert len(chunks) == truth["page_count"]
    image_page = truth["images"][0]["page"]
    assert "image_description_failed" in by_page(chunks)[image_page].provenance["fallbacks"]
    table = truth["tables"][0]
    assert any(
        f.startswith("table_rows_from_xml_markers:t1:table_distribution_failed")
        for f in by_page(chunks)[table["start_page"]].provenance["fallbacks"]
    )
    call = by_page(chunks)[image_page].provenance["calls"][0]
    assert call["status"] == "error" and call["usage"] is None and "error" in call


def test_conversion_failure_skips_the_model_entirely(synthetic_docx, settings, scripted_client, monkeypatch):
    path, truth = synthetic_docx

    def broken(*args, **kwargs):
        raise ConversionError("LibreOffice was not found")

    monkeypatch.setattr(pipeline, "convert_to_pdf", broken)
    client = scripted_client()
    chunks = extract(path, client, settings)
    assert client.backend.calls == []
    assert len(chunks) == truth["page_count"]
    assert all("pdf_unavailable" in c.provenance["fallbacks"] for c in chunks)
    rows = sum(len(e["rows"]) for c in chunks for e in c.metadata["multi_page_tables"])
    assert rows == truth["tables"][0]["row_count"]


def test_short_render_skips_pages_it_does_not_have(synthetic_docx, settings, scripted_client, truth_by_task,
                                                   fake_pdf):
    path, truth = synthetic_docx
    fake_pdf(1)  # LibreOffice paginated into a single page
    client = scripted_client(by_task=truth_by_task(truth))
    chunks = by_page(extract(path, client, settings))
    assert sorted(chunks) == list(range(1, truth["page_count"] + 1))  # text pages are never lost
    image_pages = sorted({img["page"] for img in truth["images"]})
    assert [c.page for c in client.backend.calls] == [1] == image_pages[:1]
    assert "[Visual content]" in chunks[1].retrieve_content
    for number in image_pages[1:]:
        assert "image_page_not_rendered" in chunks[number].provenance["fallbacks"]
        assert "[Visual content]" not in chunks[number].retrieve_content
    table = truth["tables"][0]
    for chunk in chunks.values():
        assert "pdf_page_count_mismatch" in chunk.provenance["fallbacks"]
        for entry in chunk.metadata["multi_page_tables"]:
            assert entry["assignment"] == "xml_row_markers"
    assert any(
        f == "table_rows_from_xml_markers:t1:table_pages_not_rendered"
        for f in chunks[table["start_page"]].provenance["fallbacks"]
    )


def test_text_only_document_never_converts(make_docx, settings, scripted_client, fake_pdf):
    conversions = fake_pdf(1)
    path = make_docx("<w:p><w:r><w:t>Only text.</w:t></w:r></w:p>", pages=1)
    client = scripted_client()
    [chunk] = extract(path, client, settings)
    assert conversions == [] and client.backend.calls == []
    assert chunk.retrieve_content == "doc - page 1\nOnly text."
    assert chunk.provenance["fallbacks"] == ["no_rendered_page_breaks:none"]


def test_app_xml_tail_page_is_recovered_from_the_render(make_docx, settings, scripted_client, fake_pdf):
    fake_pdf(3)
    path = make_docx("<w:p><w:r><w:t>Text page.</w:t></w:r></w:p>", pages=3)
    answers = {IMAGE: lambda r: {"image_descriptions": [{"location_hint": "full page", "description": "A wiring sketch."}]
                                 if r.page == 2 else [], "notes": ""}}
    chunks = extract(path, scripted_client(by_task=answers), settings)
    assert [c.page for c in chunks] == [1, 2]  # page 3 had nothing to describe: no chunk, no renumbering
    tail = chunks[1]
    assert tail.metadata["placeholder_page"] and "A wiring sketch." in tail.retrieve_content
    assert "app_xml_tail_page" in tail.provenance["fallbacks"]
    assert "app_xml_tail_padding" in tail.provenance["fallbacks"]


def test_empty_document_is_a_hard_failure(make_docx, settings, scripted_client):
    path = make_docx("<w:p/>")
    with pytest.raises(DocxExtractionError, match="no extractable content"):
        extract(path, scripted_client(), settings)


def test_record_and_replay_reproduce_the_same_chunks(synthetic_docx, settings, scripted_client, truth_by_task,
                                                     fake_pdf, tmp_path):
    path, truth = synthetic_docx
    fake_pdf(truth["page_count"])
    record = tmp_path / "calls.jsonl"
    live = scripted_client(by_task=truth_by_task(truth))
    recorded = extract(path, VLMClient(RecordingBackend(live.backend, record)), settings)
    replayed = extract(path, VLMClient(ReplayBackend(record)), settings)
    assert [c.retrieve_content for c in replayed] == [c.retrieve_content for c in recorded]
    lines = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
    assert {line["task"] for line in lines} == {IMAGE, TABLE}


@pytest.mark.soffice
def test_end_to_end_with_libreoffice(tmp_path, settings, scripted_client, truth_by_task):
    from vlmrag.synthetic.docx import generate

    [path] = generate(tmp_path, seed=0)
    truth = json.loads((tmp_path / f"{path.stem}.truth.json").read_text(encoding="utf-8"))
    client = scripted_client(by_task=truth_by_task(truth))
    chunks = extract(path, client, settings)
    assert [c.page for c in chunks] == list(range(1, truth["page_count"] + 1))
    pages = by_page(chunks)
    for chunk in chunks:
        assert chunk.provenance["fallbacks"] == []
        assert chunk.provenance["render"]["pdf_pages"] == truth["page_count"]
    for page in truth["pages"]:
        for fact in page["must_mention"]:
            assert fact.lower() in pages[page["page"]].retrieve_content.lower()
    images = [img for call in client.backend.calls for img in call.images()]
    assert images and all(img.png.startswith(b"\x89PNG") for img in images)
