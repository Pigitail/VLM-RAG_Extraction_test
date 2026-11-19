# Evaluation

`python -m vlmrag.evaluation` runs every technique over documents with ground truth, runs a text-only
baseline over the same documents, and writes `results.json` (machine-readable) and `results.md` (tables
ready to paste into a README). With `--record-dir` it also stores every model exchange, so anyone can
reproduce the numbers later with `--replay-dir` and no API key.

The harness never decides what "good" means on its own: every number is defined below, the same
definitions are stored in `results.json` under `metric_definitions`, and every ratio in `results.md`
is printed with its count.

## 1. Running it

All commands are run from the repository root with the package installed (`pip install -e ".[dev]"`).
LibreOffice is needed for the slides and docx techniques to render the pages the model sees; without it
those techniques make no model calls and the report says so.

### Harness self-check (no model, no key)

```
python -m vlmrag.evaluation --oracle --out-dir out/eval-oracle
```

`--oracle` answers every model call from the truth files, like a flawless model. It should score at or
near 100 %; the documented gap is the drawing sheet that inherits another sheet's title block (see
`drawing.groups`). Anything else below 100 % is a harness or technique bug, not a model weakness.

### A model run

Credentials come from the environment (or a git-ignored `.env`, see `.env.example`), never from flags:

```
# Azure OpenAI: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (and optionally AZURE_OPENAI_API_VERSION)
python -m vlmrag.evaluation --provider azure --model <deployment name> --variants cap,nocap \
    --label "<what this run is>" --date <YYYY-MM-DD> --out-dir out/eval
```

### Recording a run for others

```
python -m vlmrag.evaluation --provider azure --model <deployment name> --variants cap,nocap \
    --record-dir examples/recordings/<model> --label "<what this run is>" --date <YYYY-MM-DD> \
    --out-dir out/eval
```

The recording directory receives one answer file per technique run, `manifest.json`, and a copy of
`results.json` / `results.md`. It contains only text: the documents themselves are regenerated from the
seed, never stored. Existing answer files of the runs being recorded are replaced. `--cache-dir` is
refused together with `--record-dir`, because a cached answer never reaches the model and would be
missing from the recording.

### Replaying without a key

```
python -m vlmrag.evaluation --replay-dir examples/recordings/<model> --out-dir out/eval-replay
```

The replay regenerates the synthetic documents exactly as recorded, answers every model call from the
recording and compares the resulting metrics and token counts with the recorded `results.json`. Exit
status 0 means every call was found, every recorded answer was used and every number matched; 1 means
a miss, an unused recorded answer or a difference (all are listed); 2 is a usage error. `--techniques`
and `--variants` select a subset of what was recorded. The replayed `results.json` / `results.md` carry
the recorded run's label, date and settings (reasoning effort, structured output) unless `--label` /
`--date` override them, and the requests are rebuilt with the recorded reasoning effort.

`pytest tests/replay` performs the same check for every directory under `examples/recordings/` (set
`VLMRAG_RECORDINGS_DIR` to check a recording elsewhere before committing it). It is skipped when there
are no recordings; the slides and docx cases need LibreOffice.

### Your own documents

`--docs DIR` evaluates the documents under `DIR` (by suffix: `.pdf` drawing, `.pptx` slides, `.docx`
docx). A document is scored when `<stem>.truth.json` sits next to it, in the format the generators in
`vlmrag.synthetic` write; documents without truth are still extracted and indexed, as retrieval
distractors. File names must be unique, because relevance is keyed by file name.

### Other options

| Option | Effect |
|---|---|
| `--techniques drawing,slides,docx` | subset of techniques (default: all) |
| `--variants cap,nocap,maxpx=N` | drawing per-image pixel cap variants, each a separate run (default: `cap`) |
| `--no-baseline` | skip the text-only baseline runs |
| `--seed N` | synthetic generator seed (default 0) |
| `--docx-layout auto\|estimated` | where the synthetic docx page breaks come from (see section 5) |
| `--label`, `--date` | stored in the results; default `$VLMRAG_EVAL_LABEL` / `$VLMRAG_EVAL_DATE` (on replay: the recorded values). The harness never reads the clock |
| `--doc-workers N` | documents processed in parallel within a run (default 1; the per-client concurrency limit applies per document) |

## 2. What runs

| Run | Backend | Documents |
|---|---|---|
| `vlm/drawing@<variant>` | the model (or recording) | every drawing, once per variant |
| `vlm/slides`, `vlm/docx` | the model (or recording) | every deck / specification |
| `baseline/<technique>` | `NullBackend` (every answer is `{}`, produced locally: no model is contacted, no tokens) | the same documents; drawing with the first variant |

Inside a run every document is isolated: an exception is recorded with its type (and a message with
absolute paths, URLs, host names, IP addresses and request ids removed) and the run continues. A failure
is a data point: its truth items are scored as wrong or not found.

Every run also records what it actually sent (`runs[].requests_sent`, per task): requests (every
attempt, retries included; cached answers never reach the backend), images, the largest and the total
image pixels, and the reasoning efforts. The cost table shows the largest image and the efforts per run,
so the pixel cap of a variant and the per-task efforts (the docx table pass uses `medium` whenever the
setting is not empty) are measured, not assumed.

Drawing variants differ only in `DrawingConfig.max_image_pixels`: `cap` is the packaged default
(4,000,000 pixels per image after enlargement), `nocap` disables the cap and sends every crop with its
configured enlargement, as the original pipeline did (the metadata-table overview of an A3 sheet then
reaches roughly 280 MP), and `maxpx=N` sets any other cap. A provider may reject very large requests;
such a call is recorded as a failure and the document is scored accordingly.

Retrieval is scored once per *retrieval run*: `vlm@<variant>` indexes the chunks of that drawing
variant together with the slides and docx runs; `baseline` indexes the baseline runs.

## 3. Metric definitions

**Normalisation.** Normalised comparisons unify dash variants (U+2010-U+2015, U+2212, U+FE58, U+FE63,
U+FF0D) to `-`, single-quote variants to `'` and double-quote variants to `"`, apply `str.casefold()`,
collapse every whitespace run to one space and strip. Nothing else changes: `0.42 kg` and `0.42kg` stay
different. Exact comparisons compare the raw strings. Ratios are rounded to 6 decimals; a ratio with an
empty denominator is `null` / `n/a`, never 0 or 1.

### Engineering drawings

| Metric | Definition |
|---|---|
| field accuracy (per field) | fraction of truth sheets whose emitted `metadata.title_block[field]` equals the truth value, exact and normalised. Fields with an empty truth value are not scored. |
| sheet accuracy | per sheet (source page), the fraction of its scored fields that match. Group rows report micro accuracy over all fields of the group, plus `sheet_mean_normalized`, the mean of per-sheet accuracies. |
| groups | `all` every truth sheet; `read` sheets whose chunk carries a title block read on that same sheet (conditional on success: sheets of failed documents are not in it, so always read it next to `all`); `inherited` sheets that received the title block of another page (the default `title_block_pages="first"`); `raster_title_block` sheets whose truth marks the title block as raster-only (no text layer, so only vision can read it); `text_layer_title_block` the others. `all`, `raster_title_block` and `text_layer_title_block` include the sheets of failed documents, scored as wrong. |
| gate pass rate | documents that returned chunks / documents attempted. `gate_failed` counts `TitleBlockGateError` (fewer than `min_pairs` pairs); `other_errors` counts failures before the gate (transport errors, empty answers, replay misses). |

### Software-manual slides

| Metric | Definition |
|---|---|
| image-role accuracy | over all truth image blocks: the first emitted description attached to the same shape (matched by shape name through the layout block) has `image_role` equal to the truth role. A block without a description counts as wrong. The confusion table maps truth role to predicted role (`missing` when undescribed). |
| must-mention recall | over all must-mention facts of all slides: the normalised fact is a substring of the normalised `retrieve_content` of the slide's chunk(s) - the text that is actually indexed. `slide_mean_recall` averages per-slide recall over slides with at least one fact. |
| hallucinated blocks | emitted descriptions that do not describe a distinct truth image block: not tied to any image block (`ungrounded`), tied to an image the truth does not list (`not_in_truth`), or a second description of an already described block (`duplicate`). |
| visual calls ok | slides that should get a model call (they contain pictures) whose call status is `ok`. |

### Specification documents (DOCX)

| Metric | Definition |
|---|---|
| page count correct | the page count of the technique's page split (`provenance.page_split`: `xml_pages + padded_pages`, else the highest chunk page) equals the truth `page_count`. |
| row->page accuracy | over all rows (header row included) of every truth table that spans pages: the set of pages whose chunk lists the row index in `metadata.multi_page_tables` is exactly `{truth page}`. Rows that are missing or listed on several pages count as wrong. `assignment` counts page entries by source: `vlm` (validated model answer) or `xml_row_markers` (deterministic fallback). |
| image must-mention recall | over all must-mention facts of pages with pictures: the normalised fact is a substring of the normalised `retrieve_content` of that page's chunk(s). |

### Retrieval

| Metric | Definition |
|---|---|
| index | one BM25 index per retrieval run over the `retrieve_content` of all chunks of all documents of that run (k1 = 1.2, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)), NFKC + casefold word tokens, each distinct query term counted once). Queries are the `queries` of the truth files of the run's documents; relevance is the listed (source file name, 1-based page) pairs. |
| ranking | chunk scores are collapsed to pages (a page scores its best chunk, so duplicate chunks of a page count once); pages scoring 0 are not retrieved. Ties are broken against relevant pages, so ranks are lower bounds and do not depend on chunk order or file names. |
| recall@k | relevant pages among the top k ranked pages / relevant pages, averaged over queries (k = 1, 3, 5). A query with several relevant pages scores partial recall. |
| MRR | mean over queries of 1 / rank of the first relevant page (0 when none is retrieved). |
| baseline | the text-only baseline indexes the chunks of the NullBackend runs. A drawing whose baseline run raised (the acceptance gate rejects an empty answer by design) is represented by the drawing technique's own chunks built with an empty title block: file name, `page N`, the text layer, structured fields parsed from that text and domain synonyms - everything a VLM chunk holds except what the model read (a test pins this to the technique's output for an empty answer with the gate disabled). Such documents are listed under `model_free_substitutes`. A drawing whose *VLM* run failed contributes no chunks to that VLM index, as it would not be indexed in production. |

The synthetic queries are written to be answerable only from visual content (each drawing query names
an `evidence` string printed only in raster regions), so the baseline is expected to do worse - but
lexical overlap with titles and body text still lets it find some pages.

### Cost and latency

Per technique run and drawing variant, from the VLM client's usage ledger: `calls` (every answer the
technique received: re-asks included, cached answers included), `cached` (the subset served from
`--cache-dir`, with no tokens and no latency; never add it to `calls`), input / output / reasoning
tokens as reported by the provider, `model_latency_s` (sum of per-call latencies; concurrent calls
overlap, so it can exceed wall time) and `wall_time_s` of the whole run, plus the same per document,
the largest image sent (`max_image_pixels`) and the reasoning efforts sent per task. A replayed run
reports the usage stored in the recording; its wall time is local. Baseline runs show their stub
answers as calls with zero tokens.

## 4. Outputs

`results.json` holds: `format`, `package_version`, `label`, `date`, `mode` (`live`, `record`, `replay`,
`oracle`), `model` (provider, model or deployment, backend), `settings` that affect results (reasoning
effort, structured output, retries, timeouts, concurrency, cache), `environment` (LibreOffice found),
`documents` (seed, docx layout source, file names and content hashes), `evaluation` (techniques,
variants, baseline, retrieval parameters, normalisation), `metrics` (per technique: `vlm` and
`baseline`; drawing additionally per variant; `retrieval` per retrieval run, with per-query ranks),
`cost`, `runs` (per document: chunks, pages, error, fallback codes; the drawing render configuration;
`requests_sent`), `replay` and `replay_check` for replays, `warnings`, `caveats` and
`metric_definitions`. No absolute paths are written. `results.md` opens with the headline caveats
(synthetic, single run, small N; an oracle run is labelled as not being model results).

`results.md` is rendered from `results.json` only, so the two never disagree.

## 5. Record / replay details

* **Answer files.** `<technique>.jsonl` or `drawing.<variant>.jsonl`, one line per model answer:
  `key`, `task`, `prompt_version`, `response`, plus `usage`, `latency_s`, `model` and the
  `reasoning_effort` the request carried. The core `ReplayBackend` (and `vlmrag run --replay`) reads the
  same files and ignores the extra fields.
* **Failures.** Every failed attempt while recording is written to `<name>.failures.jsonl` with its
  exception type and HTTP status only - provider messages can echo endpoints or request ids and are
  never stored. On replay, a request whose recorded answers are used up (or that never succeeded) and
  that failed while recording fails again (`RecordedFailure`, not retried), so the technique takes the
  same fallback instead of reporting a miss.
* **Keys.** The replay key covers the task, prompt version, source file name, page, the text parts and
  the image labels - not image bytes, not the document hash. Renderer and font differences therefore do
  not cause misses, while any change to prompts, hints, crops, labels or the page split does. Because
  image bytes are not in the key, the drawing variants are kept in separate answer files.
* **Misses.** Slides and docx turn a model error into a fallback, which would hide a replay miss; the
  replay backend counts misses itself, and any miss fails the replay. A request made more often than it
  was recorded is a miss too (`reason: "exhausted"`): unlike the core replay, the last answer is never
  repeated. A recorded answer that is never asked for (`unused`) also fails the replay.
* **Portable documents.** The docx generator places its rendered page-break markers where the local
  LibreOffice breaks pages, so a different LibreOffice (or none) could produce a different document and
  different requests. The manifest stores the recorded `anchor_pages` of the docx truth and the replay
  regenerates the document with exactly that layout (`vlmrag.synthetic.docx.generate(..., layout=...)`,
  `layout_source: "pinned"`). `--docx-layout estimated` (or `python -m vlmrag.synthetic
  --no-libreoffice`) gives a machine-independent layout without LibreOffice; do not use it for model
  runs, because LibreOffice still renders the pages the model sees and rows can then sit on other pages
  in the renders than in the truth.
* **Drawing settings.** `VLMRAG_DRAWING_*` values that differ from the defaults change crops and
  labels; the manifest stores them and the replay applies them. A replay ignores the replaying
  machine's own `VLMRAG_DRAWING_*` variables (and says so). A custom schema or domain-terms *file* is not
  stored, so a recording made with one cannot be replayed elsewhere; the replay warns about it.

## 6. Caveats

* **Synthetic data.** Three drawings (four sheets), one six-slide deck and one four-page specification
  with invented content. The numbers describe these fixtures, not real document collections.
* **Small N.** One field, image block or query moves a ratio by several points; read ratios with their
  counts.
* **Single run.** Model answers vary from run to run; no confidence intervals are computed.
* **By-design gaps.** With the default `title_block_pages="first"`, later drawing sheets carry the title
  block of sheet 1 (`inherited` group). The synthetic DOCX carries correct rendered-page markers, so
  the deterministic row fallback of the baseline is exact by construction: row->page accuracy shows
  whether the model path keeps rows on their pages, not whether it repairs stale markers.
* **Lexical matching.** BM25 and substring must-mention matching do not reward paraphrases; an
  embedding retriever and a human reader would.
