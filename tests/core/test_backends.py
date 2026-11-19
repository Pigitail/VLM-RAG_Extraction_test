from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vlmrag.config import Settings
from vlmrag.vlm.backends import (
    ChatCompletionsBackend,
    NullBackend,
    OpenAIResponsesBackend,
    RecordingBackend,
    ReplayBackend,
    ReplayMiss,
    ScriptedBackend,
    extract_response_text,
    make_backend,
    replay_key,
)
from vlmrag.vlm.types import ImageInput, VLMError, VLMRequest

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}
LABEL = "TILE r1c1 x=0.00-0.50 y=0.00-0.50"


class FakeResponsesClient:
    def __init__(self, response):
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class FakeChatClient:
    def __init__(self, response):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _request(**overrides):
    base = dict(
        task="unit.vision",
        prompt_version="unit.vision/1",
        parts=["Read the title block.", ImageInput(b"png-a", label=LABEL), ImageInput(b"png-b")],
        instructions="You are a careful reader.",
        json_schema=SCHEMA,
        schema_name="title_block",
        max_output_tokens=1234,
        reasoning_effort="low",
        timeout_s=42.0,
        source_name="sample.pdf",
        page=2,
    )
    base.update(overrides)
    return VLMRequest(**base)


def _responses_payload():
    return SimpleNamespace(
        model="model-x",
        status="completed",
        output=[
            SimpleNamespace(type="reasoning", content=None, summary=[]),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"title": '),
                    SimpleNamespace(type="output_text", text='"Bracket"}'),
                    SimpleNamespace(type="refusal", refusal="nope"),
                ],
            ),
        ],
        output_text='{"title": "Bracket"}',
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        ),
    )


def test_responses_request_construction():
    fake = FakeResponsesClient(_responses_payload())
    backend = OpenAIResponsesBackend(fake, model="deploy-1", name="azure")
    response = backend.generate(_request())
    kwargs = fake.calls[0]
    assert kwargs["model"] == "deploy-1"
    assert kwargs["instructions"] == "You are a careful reader."
    assert kwargs["input"][0]["role"] == "user"
    content = kwargs["input"][0]["content"]
    assert [c["type"] for c in content] == ["input_text", "input_text", "input_image", "input_image"]
    assert content[1]["text"] == LABEL  # the label comes right before its image
    assert content[2]["image_url"].startswith("data:image/png;base64,")
    assert "detail" not in content[2]
    assert kwargs["text"]["format"] == {"type": "json_schema", "name": "title_block", "schema": SCHEMA, "strict": True}
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["max_output_tokens"] == 1234
    assert kwargs["timeout"] == 42.0
    assert response.backend == "azure"
    assert response.model == "model-x"


def test_responses_optional_fields_are_omitted():
    fake = FakeResponsesClient(_responses_payload())
    backend = OpenAIResponsesBackend(fake, model="m", structured_output=False, image_detail="high")
    backend.generate(_request(instructions=None, reasoning_effort=None, max_output_tokens=None, timeout_s=None))
    kwargs = fake.calls[0]
    for key in ("instructions", "reasoning", "max_output_tokens", "timeout", "text"):
        assert key not in kwargs
    images = [c for c in kwargs["input"][0]["content"] if c["type"] == "input_image"]
    assert images and all(c["detail"] == "high" for c in images)


def test_responses_default_timeout_applies_when_request_has_none():
    fake = FakeResponsesClient(_responses_payload())
    OpenAIResponsesBackend(fake, model="m", default_timeout_s=7).generate(_request(timeout_s=None))
    assert fake.calls[0]["timeout"] == 7.0


def test_responses_text_and_usage_parsing():
    response = OpenAIResponsesBackend(FakeResponsesClient(_responses_payload()), model="m").generate(_request())
    # Pieces of one message are concatenated once; output_text (the same content) is not appended again.
    assert response.text == '{"title": "Bracket"}'
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 50
    assert response.usage.reasoning_tokens == 20
    assert response.usage.total_tokens == 150


def test_response_text_fallbacks_and_dedupe():
    assert extract_response_text(SimpleNamespace(output=[], output_text=" hello ")) == "hello"
    echoed = {
        "output": [
            {"content": [{"type": "output_text", "text": "same"}]},
            {"content": [{"type": "text", "text": "same"}]},
        ]
    }
    assert extract_response_text(echoed) == "same"
    assert extract_response_text(SimpleNamespace()) == ""
    assert extract_response_text("raw") == "raw"


def test_chat_completions_request_and_parsing():
    payload = SimpleNamespace(
        model="local-vlm",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"title": "Bracket"}'))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )
    fake = FakeChatClient(payload)
    response = ChatCompletionsBackend(fake, model="local-vlm").generate(_request())
    kwargs = fake.calls[0]
    assert kwargs["model"] == "local-vlm"
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a careful reader."}
    user = kwargs["messages"][1]
    assert user["role"] == "user"
    assert [c["type"] for c in user["content"]] == ["text", "text", "image_url", "image_url"]
    assert user["content"][1]["text"] == LABEL
    assert user["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == SCHEMA
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["max_tokens"] == 1234
    assert kwargs["timeout"] == 42.0
    assert "reasoning_effort" not in kwargs  # only forwarded when enabled
    assert response.text == '{"title": "Bracket"}'
    usage = response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens, usage.total_tokens) == (11, 7, 3, 18)
    assert response.backend == "compatible"


def test_chat_completions_options_and_list_content():
    payload = {"choices": [{"message": {"content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}}]}
    fake = FakeChatClient(payload)
    backend = ChatCompletionsBackend(fake, model="m", structured_output=False, send_reasoning_effort=True)
    response = backend.generate(_request(instructions=None))
    kwargs = fake.calls[0]
    assert "response_format" not in kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["messages"][0]["role"] == "user"
    assert response.text == "part1\npart2"
    assert response.usage.total_tokens == 0
    assert ChatCompletionsBackend(FakeChatClient({"choices": []}), model="m").generate(_request()).text == ""


def test_make_backend_azure_with_api_version_uses_azure_client():
    import openai

    settings = Settings(
        provider="azure",
        model="my-deployment",
        azure_api_key="test-placeholder-key",
        azure_endpoint="https://example-resource.example.com",
        azure_api_version="2025-04-01-preview",
    )
    backend = make_backend(settings)
    assert isinstance(backend, OpenAIResponsesBackend)
    assert isinstance(backend.client, openai.AzureOpenAI)
    assert backend.model == "my-deployment"
    assert backend.name == "azure"
    assert backend.client.max_retries == 0


def test_make_backend_azure_without_api_version_uses_v1_base_url():
    import openai

    for endpoint in ("https://example-resource.example.com/", "https://example-resource.example.com/openai/v1"):
        settings = Settings(
            provider="azure", model="my-deployment", azure_api_key="test-placeholder-key", azure_endpoint=endpoint
        )
        backend = make_backend(settings)
        assert type(backend.client) is openai.OpenAI
        assert not isinstance(backend.client, openai.AzureOpenAI)
        assert str(backend.client.base_url).rstrip("/") == "https://example-resource.example.com/openai/v1"


def test_make_backend_variants_and_errors():
    import openai

    assert isinstance(make_backend(Settings(provider="null")), NullBackend)
    compat = make_backend(Settings(provider="compatible", model="local", openai_base_url="http://localhost:8000/v1"))
    assert isinstance(compat, ChatCompletionsBackend)
    assert isinstance(compat.client, openai.OpenAI)
    responses_backend = make_backend(Settings(provider="openai", openai_api_key="test-placeholder-key", timeout_s=123))
    assert isinstance(responses_backend, OpenAIResponsesBackend)
    assert responses_backend.client.timeout == 123.0
    assert responses_backend.client.max_retries == 0
    with pytest.raises(VLMError):
        make_backend(Settings(provider="openai"))
    with pytest.raises(VLMError):
        make_backend(Settings(provider="azure", azure_endpoint="https://example-resource.example.com"))
    with pytest.raises(VLMError):
        make_backend(Settings(provider="compatible"))


def test_null_backend_returns_empty_object():
    assert NullBackend().generate(_request()).text == "{}"


def test_replay_key_is_model_and_pixel_independent():
    a = _request()
    b = _request(parts=["Read the title block.", ImageInput(b"other-bytes", label=LABEL), ImageInput(b"zzz")])
    c = _request(doc_id="different-doc-id", json_schema=None, max_output_tokens=1)
    assert replay_key(a) == replay_key(b) == replay_key(c)
    assert replay_key(a) != replay_key(_request(page=3))
    assert replay_key(a) != replay_key(_request(source_name="other.pdf"))
    assert replay_key(a) != replay_key(_request(prompt_version="unit.vision/2"))
    assert replay_key(a) != replay_key(_request(task="unit.other"))
    relabeled = _request(parts=["Read the title block.", ImageInput(b"png-a", label="other"), ImageInput(b"png-b")])
    assert replay_key(a) != replay_key(relabeled)
    assert replay_key(_request(parts=["line1\r\nline2"])) == replay_key(_request(parts=["line1\nline2"]))


def test_record_replay_round_trip(tmp_path):
    answers = iter(['{"n": 1}', '{"n": 2}', '{"n": 3}'])
    inner = ScriptedBackend(lambda request: next(answers), model="rec-model")
    path = tmp_path / "rec" / "calls.jsonl"
    recorder = RecordingBackend(inner, path)
    first = recorder.generate(_request(page=1))
    second = recorder.generate(_request(page=2))
    again = recorder.generate(_request(page=1))  # identical request recorded twice
    assert (recorder.name, recorder.model) == ("scripted", "rec-model")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [sorted(line) for line in lines] == [["key", "prompt_version", "response", "task"]] * 3
    assert lines[0]["key"] == replay_key(_request(page=1))
    assert lines[0]["task"] == "unit.vision"

    replay = ReplayBackend(path)
    assert len(replay) == 3
    other_pixels = _request(page=1, parts=["Read the title block.", ImageInput(b"x", label=LABEL), ImageInput(b"y")])
    assert replay.generate(other_pixels).text == first.text
    assert replay.generate(_request(page=2)).text == second.text
    assert replay.generate(_request(page=1)).text == again.text
    assert replay.generate(_request(page=1)).text == again.text  # the last answer repeats
    with pytest.raises(ReplayMiss) as info:
        replay.generate(_request(page=9))
    assert isinstance(info.value, KeyError)
    assert "page=9" in str(info.value)


def test_replay_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(VLMError):
        ReplayBackend(path)


def test_scripted_backend_encodes_dicts_and_records_calls():
    backend = ScriptedBackend(lambda request: {"ok": True, "page": request.page})
    response = backend.generate(_request())
    assert json.loads(response.text) == {"ok": True, "page": 2}
    assert backend.calls[0].task == "unit.vision"
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0


def test_image_input_helpers():
    image = ImageInput(b"\x89PNG-bytes", label="L")
    assert image.data_url().startswith("data:image/png;base64,")
    assert len(image.sha256()) == 64
    assert "bytes=" in repr(image) and "PNG-bytes" not in repr(image)
