# Common building blocks

Every technique in this repository (`drawing`, `slides`, `docx`) turns a document into
retrieval chunks with the help of a vision-language model (VLM). They share three things,
described here:

1. the **chunk format** with its dual (in fact triple) representation,
2. the optional **retrieval-summary prefix**,
3. the **VLM call infrastructure**: backends, concurrency limit, retries, JSON recovery,
   cache, usage accounting, provenance and record/replay,

plus the rendering and conversion helpers they all rely on. Each section explains *why* the
design looks the way it does; most choices come from failures observed in production
ingestion runs.

---

## 1. Dual representation

A chunk (`vlmrag.schema.Chunk`) stores the same content three times:

| field | consumer | optimised for |
|---|---|---|
| `retrieve_content` | embedding model and BM25 index | recall: plain text, labels spelled out, visual descriptions included |
| `llm_content` | the answering LLM | faithfulness: Markdown/HTML, tables stay tables |
| `filtering_content` | exact-match filters | robustness: the retrieval text with **all whitespace removed** |

**Why not one text?** The two readers want different things.

* Embeddings and BM25 are hurt by markup. A Markdown table spends many tokens on `|` and
  `---`, and a cell value ends up far away from its column header. For retrieval, a table is
  rendered as one line per row with cells joined by ` | ` (`vlmrag.text.plain_table`) and
  key/value pairs are written out as `label: value`.
* The answering LLM, on the other hand, needs structure to answer correctly ("which revision
  changed the material?"), so it gets a real Markdown table (`vlmrag.text.markdown_table`).
* Visual information (what a screenshot shows, what a diagram depicts) must be part of
  `retrieve_content`, not only of metadata. Metadata is not embedded, so a description that
  lives only there can never be found.

**Why a whitespace-free copy?** Part numbers, drawing numbers and codes are what users filter
on, and they are exactly the strings that text layers and OCR break apart (`L-300 -02`,
`L-300-\n02`). Removing all whitespace on both sides makes substring filters survive line
breaks and spacing differences without fuzzy matching.

**Page fidelity.** `Chunk.page` is always the 1-based page or slide number of the *source*
document. Document viewers jump straight to it, so a technique never emits chunks for virtual
or merged pages and never renumbers pages. `Chunk.__post_init__` rejects anything that is not
a positive integer. `chunk_index` is the 0-based position of the chunk among the chunks of its
document (emission order); pages without content emit no chunk, so it is not always
`page - 1`.

**Provenance.** `Chunk.provenance` records how a chunk was produced: model and backend,
prompt versions, render DPI and crop boxes, which fallbacks fired, and token usage. When a
chunk is wrong, provenance tells you whether to look at the prompt, the crop geometry or the
fallback logic. Every technique writes these keys, so one script can audit a mixed JSONL:

| key | content |
|---|---|
| `technique` | the doc type (`drawing`, `slides`, `docx`) |
| `model`, `backend` | the client's model and backend name (`openai`, `compatible`, `null`, `replay`, ...) |
| `prompt_versions` | `{task: prompt_version}` of the prompts that produced this chunk (empty when none ran) |
| `render` | render parameters (`dpi`, converter, crops); slides without pictures have none |
| `fallbacks` | codes of the fallbacks that fired; each technique documents its codes |
| `calls` | one record per model call behind the chunk: `task`, `prompt_version`, `status` (`ok`, `empty`, `unparseable`, `invalid`, `error`, or a more specific technique value), and when a response arrived `cached`, `latency_s`, `usage`; `error` / `reason` when the answer was not used |

Task names are `<technique>.<pass>` (`drawing.vision`, `slides.visual`,
`docx.table_distribution`, `augment.summary`) and prompt versions are `<task>/<n>`. The version
is part of the cache and replay keys, so bump it whenever a prompt changes. A call that
serves several chunks (a drawing title block reused on later sheets, a table spanning pages)
appears in each of them: sum token usage from the client's usage table, not from chunks. The
optional retrieval summary (section 2) adds `augment.summary` to `prompt_versions` and a
`retrieval_summary` entry; its call is counted in the usage table, not in `calls`.

Technique-specific provenance and metadata are described in [drawing.md](drawing.md),
[slides.md](slides.md) and [docx.md](docx.md).

## 2. Retrieval-summary prefix (optional)

`vlmrag.augment.summary` can prepend a short, source-grounded summary to `retrieve_content`:

```
doc_type: slides | source_name: manual.pptx | page: 4 | Korean retrieval summary: ... |
English retrieval summary: ... | retrieval keywords: ... | ...
<original retrieve_content>
```

The model receives the chunk text (capped at 7,000 characters) and must return
`summary_ko` (2-4 sentences), `summary_en` (1-3 sentences) and three keyword lists; it is told
to use only grounded information, to prefer distinctive terms (part numbers, screen names,
numeric conditions, defects, actions) over generic ones, and to return empty values when
the source has nothing searchable. The answer is limited to 2,400 output tokens.

**Why.** Users search in a different language, or with different words, than the document
uses. A bilingual summary plus keywords lets dense and sparse retrieval match queries that
share no surface form with the raw chunk.

**What it never touches.** `llm_content` stays source-only: the answering LLM must read the
document, not a model-written paraphrase. `filtering_content` also stays source-only, so an
exact-match filter can only match strings that really occur in the document.

**Why it is opt-in per document type** (`VLMRAG_RETRIEVAL_SUMMARY=slides,docx`, or `--summary`
on the CLI). On numeric tables the prefix did more harm than good:

* the model "helpfully" rounded or recombined numbers, so the prefix contained values that do
  not exist in the source and attracted wrong matches;
* the extra text consumed a large share of the 8,192-token embedding window, pushing real
  table rows out of what actually got embedded.

It was therefore disabled for table-heavy chunks and is enabled only where the source is
prose-like or visual. Treat it as an experiment you switch on per collection and evaluate
with the synthetic queries, not as a default.

The summary is an enhancement, never a requirement: any failure (empty input, unparseable
answer after two re-asks, a non-retryable API error) returns `None` and the chunk is kept
unchanged.

## 3. VLM call infrastructure

### Requests are provider-neutral

Techniques build a `VLMRequest`: an ordered list of parts (`str` or `ImageInput`), optional
`instructions`, an optional strict JSON schema, `max_output_tokens`, `reasoning_effort`, a
timeout, and identification fields (`task`, `prompt_version`, `doc_id`, `source_name`,
`page`). They never import a provider SDK. This is what makes every technique testable with
scripted answers and runnable against different providers.

**Position labels.** An `ImageInput` carries a label such as
`TILE r1c2 x=0.24-0.52 y=0.00-0.37 dpi=400`, which every backend sends as a text part right
before the image. A model cannot know where a crop came from; telling it lets prompts say
"values near the bottom-right belong to the title block" and lets the model reconcile
overlapping tiles. Labels use normalised, top-left-origin coordinates on the displayed page
(see section 4).

### Backends

| backend | use |
|---|---|
| `OpenAIResponsesBackend` | OpenAI and Azure OpenAI via the Responses API (`input_text` / `input_image` parts, `text.format` JSON schema, `reasoning.effort`, `max_output_tokens`) |
| `ChatCompletionsBackend` | any OpenAI-compatible `/chat/completions` server (vLLM, Ollama, ...); `response_format` can be disabled for servers that reject it; `reasoning_effort` is forwarded only on request |
| `ScriptedBackend` | tests: a function maps each request to an answer |
| `NullBackend` | dry runs: answers `{}` to everything, which exercises every fallback path |
| `RecordingBackend` / `ReplayBackend` | record real exchanges once, replay them anywhere without credentials |

`make_backend(settings)` picks one from `VLMRAG_PROVIDER`. For Azure, setting
`AZURE_OPENAI_API_VERSION` selects the classic `AzureOpenAI(api_version=...)` client; leaving
it empty uses the version-less v1 endpoint (`<endpoint>/openai/v1/`) through the plain OpenAI
client. SDK clients are created with `max_retries=0` because retrying is the client's job
(below); retrying in two layers multiplies attempts and hides rate limiting.

**Robust text extraction.** A Responses API result is a list of output items (reasoning
items, messages, tool calls). Only `output_text` pieces of messages carry the answer, so the
backend collects exactly those, concatenating the pieces of one message and joining messages
with newlines. The SDK's `output_text` convenience property is used only as a fallback:
appending it in addition would duplicate the answer whenever a message has more than one
piece. Identical messages are de-duplicated because some gateways echo the final message.

### The client: limiter, cache, retry, usage

`VLMClient.call(request)` runs a request through four stages:

1. **Cache** (`DiskCache`, optional). The key is the SHA-256 of the model, task, prompt
   version, document id, page, every text part, the SHA-256 of every image (with its label),
   the JSON schema and the generation parameters. Change any of them and the cache misses.
   Re-running after a crash or after changing post-processing code costs nothing. Entries are
   written atomically, so threads and processes can share a cache directory.
2. **Limiter.** The client owns a `threading.BoundedSemaphore(max_concurrency)`. VLM calls are
   I/O-bound, so techniques fan out with a `ThreadPoolExecutor` and the semaphore caps
   in-flight requests. A module-level `multiprocessing.Semaphore` is deliberately avoided: it
   is shared only between processes created by `fork`; with `spawn` or `forkserver` (the
   default on Windows and macOS, and on Linux since Python 3.14) every worker silently gets
   its own copy and the "global" limit is multiplied by the number of processes.
3. **Retry.** Rate limits (429), server errors (5xx), 408/409, timeouts and connection errors
   are retried with exponential backoff and jitter (`base * 2^(n-1)`, capped at 32 s, with
   half of it randomised so workers that failed together do not retry in lockstep). A
   `Retry-After` header wins when present. Other 4xx errors (bad request, authentication,
   content filter) are never retried: repeating them only burns quota and time. The
   semaphore is held only around the request itself, never while sleeping, so one call that
   is backing off does not block the others. `max_retries=3` means up to four attempts.
4. **Usage.** Every call is recorded in a thread-safe `UsageLedger` per task (calls, cached
   calls, input/output/reasoning tokens, latency). The CLI prints the table at the end of a
   run; it shows immediately which stage of a technique dominates cost.

### JSON recovery

Even with structured outputs, real answers arrive inside Markdown fences, after a sentence of
prose, followed by a remark, or (with servers that ignore `response_format`) next to
reasoning text. `call_json` parses the answer with `extract_json_value`:

1. strip code fences and try `json.loads` on the whole text;
2. otherwise scan for `{` / `[`, track nesting **while skipping string literals** (honouring
   escapes), and try each balanced top-level candidate in order until one parses.

A naive "first `{` to last `}`" slice fails as soon as a value contains a brace or the
trailing prose does; the string-aware scanner does not. Unparseable output returns `None`
instead of raising, so every technique can apply its own deterministic fallback.
Unparseable answers are never written to the cache (a garbled answer must not be replayed
forever), and `call_json(request, parse_retries=n)` can re-ask.

### Record and replay

`RecordingBackend(inner, path)` appends `{"key", "task", "prompt_version", "response"}` per
call; `ReplayBackend(path)` answers from that file and raises `ReplayMiss` (a `KeyError`)
for unknown requests. The key (`replay_key`) deliberately uses only model- and
platform-independent fields: task, prompt version, source file name, page, the ordered text
parts and image labels. Image bytes are excluded because PDFium, LibreOffice and font
versions change pixels between machines, and `doc_id` is excluded because regenerated
fixtures differ byte-wise even when their content is identical. A practical consequence:
never put absolute paths or temporary file names into text parts.

```
vlmrag run slides deck.pptx --out out.jsonl --record calls.jsonl   # once, with credentials
vlmrag run slides deck.pptx --out out.jsonl --replay calls.jsonl   # anywhere, offline
```

## 4. Rendering and conversion

**Displayed coordinates everywhere.** `PdfPages.render(index, dpi, bbox, upscale)` and
`PdfPages.text(index, bbox)` take normalised `(x0, y0, x1, y1)` boxes with a top-left origin
on the page *as displayed*, i.e. after `/Rotate` is applied. Drawings are often stored as
portrait pages with `/Rotate 90`; computing crops in unrotated coordinates would put "the
bottom-right title block" in a different corner for every rotation. Rendering lets PDFium
apply the rotation and crops after it, so only the requested region is rasterised (no
full-page bitmap at 650 dpi). The text layer API works in unrotated PDF space, so boxes are
mapped back through the inverse rotation. Both paths are tested on generated pages with
`/Rotate` 0, 90, 180 and 270. PDFium is not thread-safe, so calls into it are serialised.

**Tiles with overlap.** `tile_grid(rows, cols, overlap, region)` grows every tile by
`overlap` x tile size on each side (clamped to the region). Small text that straddles a cut
would otherwise be split into two unreadable halves; with the overlap band every glyph is
whole in at least one tile. `upscale` enlarges crops with LANCZOS because vision models
tokenise images in fixed-size patches: enlarging 4-6 pt text makes it span enough patches to
be read.

**LibreOffice conversion.** `convert(src, target_ext, out_dir, timeout_s=60)` runs headless
LibreOffice with a fresh, short, private profile directory per call
(`<tmp>/vlmrag_lo_<8 hex>`, removed afterwards; `VLMRAG_LO_PROFILE_ROOT` replaces `<tmp>`).
A private profile is required because LibreOffice refuses to run two instances on one
profile; the path is short because a long profile path makes LibreOffice crash at start-up
on Windows. A timeout no longer escapes as
`TimeoutExpired`: the process tree is killed, the next candidate binary is tried, and only
when every candidate failed is a `ConversionError` raised that lists each attempt and its
reason.

## 5. Running techniques in batches

`vlmrag run <technique> INPUT... --out out.jsonl` processes files (or directories) with an
optional thread pool. Every file is isolated: an exception is written to
`out.errors.jsonl` with its type, message and traceback, and the batch continues. A pipeline
that gathers results from a worker pool inside a single `try` loses every remaining result as
soon as one file fails; this runner never does. Chunks are written in input order as soon as
all earlier files are finished, so an interrupted run keeps its completed prefix. The exit
code is 1 when any file failed and 2 for configuration errors.

## 6. Configuration

All settings come from environment variables (see `.env.example`); credentials are never read
from files in the repository, never logged and excluded from `repr(Settings)`. The CLI loads a
local, git-ignored `.env` when present.

| variable | default | meaning |
|---|---|---|
| `VLMRAG_PROVIDER` | `openai` | `openai`, `azure`, `compatible` or `null` |
| `VLMRAG_MODEL` | `gpt-6-luna` | model name (deployment name on Azure) |
| `VLMRAG_MAX_CONCURRENCY` | `4` | in-flight requests per client |
| `VLMRAG_TIMEOUT_S` | `600` | default request timeout |
| `VLMRAG_MAX_RETRIES` | `3` | retries after the first attempt |
| `VLMRAG_REASONING_EFFORT` | `low` | empty disables the parameter |
| `VLMRAG_STRUCTURED_OUTPUT` | `1` | request strict JSON-schema output |
| `VLMRAG_CACHE_DIR` | unset | enables the response cache |
| `VLMRAG_SOFFICE` | auto | LibreOffice binary |
| `VLMRAG_LO_PROFILE_ROOT` | system temp dir | where the per-call LibreOffice profiles go (keep it short) |
| `VLMRAG_RETRIEVAL_SUMMARY` | empty | doc types that get the summary prefix |
