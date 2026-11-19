from __future__ import annotations

import dataclasses
import threading

from vlmrag.vlm.cache import DiskCache, cache_key
from vlmrag.vlm.types import ImageInput, Usage, VLMRequest, VLMResponse
from vlmrag.vlm.usage import UsageLedger


def _request(**overrides):
    base = dict(
        task="unit.vision",
        prompt_version="unit.vision/1",
        parts=["prompt", ImageInput(b"pixels", label="FULL")],
        doc_id="doc-1",
        page=1,
        json_schema={"type": "object"},
        max_output_tokens=100,
        reasoning_effort="low",
    )
    base.update(overrides)
    return VLMRequest(**base)


def test_cache_key_covers_every_input():
    base = _request()
    key = cache_key(base, "model-a")
    assert key == cache_key(_request(), "model-a")
    assert key == cache_key(_request(timeout_s=99, source_name="renamed.pdf"), "model-a")  # not answer-relevant
    variants = [
        (base, "model-b"),
        (_request(task="unit.other"), "model-a"),
        (_request(prompt_version="unit.vision/2"), "model-a"),
        (_request(doc_id="doc-2"), "model-a"),
        (_request(page=2), "model-a"),
        (_request(parts=["other prompt", ImageInput(b"pixels", label="FULL")]), "model-a"),
        (_request(parts=["prompt", ImageInput(b"other pixels", label="FULL")]), "model-a"),
        (_request(parts=["prompt", ImageInput(b"pixels", label="TILE")]), "model-a"),
        (_request(json_schema={"type": "array"}), "model-a"),
        (_request(max_output_tokens=200), "model-a"),
        (_request(reasoning_effort="high"), "model-a"),
        (_request(instructions="be brief"), "model-a"),
    ]
    keys = {cache_key(request, model) for request, model in variants}
    assert key not in keys and len(keys) == len(variants)


def test_cache_put_get_and_corrupt_entry(tmp_path):
    cache = DiskCache(tmp_path / "c")
    request = _request()
    assert cache.get(request, "m") is None
    cache.put(request, "m", VLMResponse(text='{"a": 1}', model="m-2026", backend="openai", usage=Usage(1, 2, 0, 3), latency_s=4.0))
    hit = cache.get(request, "m")
    assert hit.text == '{"a": 1}' and hit.cached and hit.model == "m-2026" and hit.usage.total_tokens == 3
    path = next((tmp_path / "c").rglob("*.json"))
    path.write_text("{corrupt", encoding="utf-8")
    assert cache.get(request, "m") is None
    assert not list((tmp_path / "c").rglob("*.tmp"))


def test_usage_ledger_is_thread_safe():
    ledger = UsageLedger()

    def worker():
        for _ in range(500):
            ledger.add("t", Usage(1, 1, 0, 2), 0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    total = ledger.summary()["total"]
    assert total["calls"] == 4000 and total["total_tokens"] == 8000


def test_usage_ledger_merge_and_cached_rows():
    a, b = UsageLedger(), UsageLedger()
    a.add("x", Usage(10, 1, 0, 11), 1.0)
    b.add("x", Usage(5, 1, 0, 6), 2.0)
    b.add("x", Usage(5, 1, 0, 6), 0.0, cached=True)
    a.merge(b)
    row = a.summary()["tasks"]["x"]
    assert dataclasses.asdict(Usage(15, 2, 0, 17)) == {k: row[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")}
    assert row["calls"] == 3 and row["cached"] == 1 and row["latency_s"] == 3.0
