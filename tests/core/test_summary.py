from __future__ import annotations

import pytest

from vlmrag.augment import summary as summary_module
from vlmrag.augment.summary import (
    MAX_OUTPUT_TOKENS,
    SOURCE_CHAR_LIMIT,
    apply_retrieval_summary,
    augment_chunks,
    summarize_for_retrieval,
)
from vlmrag.schema import Chunk

ANSWER = {
    "summary_ko": "  L-300 랜턴 하우징 도면의 표제란 정보.  ",
    "summary_en": "Title block of the L-300 lantern housing drawing.",
    "keywords_ko": ["랜턴 하우징", "표제란"],
    "keywords_en": ["lantern housing", "title block", "lantern housing"],
    "keywords_source": "L-300; Rev B",
}


def _chunk(**overrides):
    base = dict(
        doc_id="d1",
        source_name="lantern.pdf",
        doc_type="slides",
        page=3,
        chunk_index=2,
        retrieve_content="Drawing number L-300-02 Rev B",
        llm_content="| Drawing number | L-300-02 |",
        filtering_content="DrawingnumberL-300-02RevB",
    )
    base.update(overrides)
    return Chunk(**base)


def test_summarize_builds_request_and_normalises_answer(scripted_client):
    client = scripted_client(ANSWER)
    result = summarize_for_retrieval(client, "x" * 10_000, source_name="lantern.pdf", page=3)
    assert result == {
        "summary_ko": "L-300 랜턴 하우징 도면의 표제란 정보.",
        "summary_en": "Title block of the L-300 lantern housing drawing.",
        "keywords": ["랜턴 하우징", "표제란", "lantern housing", "title block", "L-300", "Rev B"],
    }
    request = client.backend.calls[0]
    assert request.task == "augment.summary"
    assert request.max_output_tokens == MAX_OUTPUT_TOKENS == 2400
    assert request.reasoning_effort == "low"
    assert request.json_schema["required"] == ["summary_ko", "summary_en", "keywords_ko", "keywords_en", "keywords_source"]
    assert request.page == 3 and request.source_name == "lantern.pdf"
    source_part = request.parts[1]
    assert source_part.startswith("[SOURCE_CHUNK]\nsource_name: lantern.pdf\npage: 3\n")
    assert len(source_part) - len("[SOURCE_CHUNK]\n") == SOURCE_CHAR_LIMIT == 7000


def test_summarize_skips_empty_input_without_calling(scripted_client):
    client = scripted_client(ANSWER)
    assert summarize_for_retrieval(client, "   ") is None
    assert client.backend.calls == []


def test_summarize_returns_none_for_empty_or_failed_answers(scripted_client):
    empty = {"summary_ko": "", "summary_en": "", "keywords_ko": [], "keywords_en": [], "keywords_source": []}
    assert summarize_for_retrieval(scripted_client(empty), "text") is None

    def boom(request):
        raise ValueError("non-retryable bug")

    assert summarize_for_retrieval(scripted_client(boom), "text") is None
    garbled = scripted_client("no json at all")
    assert summarize_for_retrieval(garbled, "text") is None
    assert len(garbled.backend.calls) == 3  # first try + 2 parse retries


def test_summarize_recovers_after_garbled_answer(scripted_client):
    answers = iter(["garbled", ANSWER])
    client = scripted_client(lambda request: next(answers))
    assert summarize_for_retrieval(client, "text")["summary_en"].startswith("Title block")


def test_apply_prefixes_retrieve_content_only():
    chunk = _chunk()
    summary = {"summary_ko": "요약", "summary_en": "Summary.", "keywords": ["L-300", "housing"]}
    updated = apply_retrieval_summary(chunk, summary)
    prefix, _, rest = updated.retrieve_content.partition("\n")
    assert rest == chunk.retrieve_content
    assert prefix == (
        "doc_type: slides | source_name: lantern.pdf | page: 3 | Korean retrieval summary: 요약 | "
        "English retrieval summary: Summary. | retrieval keywords: L-300 | housing"
    )
    assert updated.llm_content == chunk.llm_content
    assert updated.filtering_content == chunk.filtering_content
    assert updated.metadata["retrieval_summary"]["keywords"] == ["L-300", "housing"]
    assert updated.provenance["retrieval_summary"]["task"] == "augment.summary"
    assert updated.provenance["prompt_versions"]["augment.summary"] == "augment.summary/1"
    assert chunk.metadata == {} and "prompt_versions" not in chunk.provenance  # the input is not mutated
    assert apply_retrieval_summary(updated, summary) is updated  # idempotent


def test_apply_noops():
    chunk = _chunk()
    assert apply_retrieval_summary(chunk, None) is chunk
    assert apply_retrieval_summary(chunk, {"summary_ko": "", "summary_en": "", "keywords": []}) is chunk
    empty = _chunk(retrieve_content="")
    assert apply_retrieval_summary(empty, {"summary_en": "x"}) is empty


def test_augment_chunks_preserves_order(scripted_client):
    def responder(request):
        page = request.page
        return {"summary_ko": f"페이지 {page}", "summary_en": f"page {page}", "keywords_ko": [], "keywords_en": [], "keywords_source": []}

    client = scripted_client(responder)
    chunks = [_chunk(page=p, chunk_index=p - 1) for p in (1, 2, 3, 4)]
    out = augment_chunks(client, chunks, workers=3)
    assert [c.page for c in out] == [1, 2, 3, 4]
    assert all(f"page {c.page}" in c.retrieve_content.splitlines()[0] for c in out)
    assert augment_chunks(client, []) == []


def test_docstring_records_the_numeric_table_lesson():
    doc = summary_module.__doc__
    assert "8192" in doc and "numeric" in doc.lower()
    assert apply_retrieval_summary.__doc__ and "llm_content" in apply_retrieval_summary.__doc__


@pytest.mark.network
def test_summary_against_real_endpoint(network_settings):
    from vlmrag.vlm.client import VLMClient

    client = VLMClient.from_settings(network_settings)
    result = summarize_for_retrieval(client, "Outdoor Lantern L-300 housing, material: aluminium, finish: matte black")
    assert result is not None and result["summary_en"]
