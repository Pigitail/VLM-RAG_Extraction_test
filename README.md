# vlm-rag-extraction

[한국어](README.ko.md)

Turning engineering drawings (PDF), software-manual slides (PPTX) and specification documents (DOCX)
into RAG chunks with a vision-language model (VLM).

These techniques come from an in-house RAG ingestion pipeline. This repo re-implements them with
generic names and synthetic documents so they can be run, tested and measured without the original data.

## Techniques

| Document | Problem | What the VLM does |
|---|---|---|
| Engineering drawing PDF ([docs](docs/techniques/drawing.md)) | Title-block values are tiny, sometimes raster-only, and the text layer is out of order | Reads the title block from overlapping tiles and close-ups; text-layer words are passed as hints only |
| Manual slides PPTX ([docs](docs/techniques/slides.md)) | Screenshots carry the meaning, but a text extractor sees only the text around them | Describes each picture (role + description + short en/ko captions) on a whole-slide render. No full OCR |
| Specification DOCX ([docs](docs/techniques/docx.md)) | Chunks need real page numbers, and long tables cross pages | Describes pictures per page; for tables that cross pages, answers only "which row is on which page" |
| Common ([docs](docs/techniques/common.md)) | Same plumbing for every document type | - |

Common parts: chunk format with `retrieve_content` / `llm_content` / `filtering_content`, optional
retrieval-summary prefix, VLM backends (OpenAI, Azure OpenAI, OpenAI-compatible servers such as vLLM),
concurrency limit, retries, JSON recovery, response cache, token accounting, record/replay.

## Rules the code follows

- Text the file already has (PDF text layer, DOCX/PPTX XML) goes into the chunk as is. Model output
  is added next to it, never in place of it.
- The VLM is asked for what that text cannot give: picture content, drawing title blocks (text layer
  missing or out of order), and which table row is on which page.
- VLM answers are checked before use. A bad answer falls back to a fixed rule and the chunk records
  which one (`provenance.fallbacks`). On drawings a title block that cannot be read (no vision answer,
  or fewer than 3 pairs) fails the file instead.
- `Chunk.page` is the page (slide) number of the source file. No merged or virtual pages, no renumbering.
- Visual descriptions go into `retrieve_content`. Metadata is not embedded, so a description kept
  only there is never found.
- A failing file is written to `<out>.errors.jsonl`; the rest of the batch keeps going.

## Results

Synthetic set (3 drawings / 4 sheets, 1 deck / 6 slides, 1 DOCX / 4 pages), Azure OpenAI `gpt-6-luna`,
reasoning effort `low` (`medium` for the DOCX table pass), single run on 2026-11-19.
Baseline = same pipeline with every VLM call answered by an empty result (no model).

| Metric | VLM | Baseline |
|---|---|---|
| Drawing title-block fields | 65/68 | 0/68 |
| Drawing sheets whose title block is raster-only (fields) | 17/17 | 0/17 |
| Slide picture role (main screenshot / detail crop / icon ...) | 7/7 | 0/7 |
| DOCX table rows on the correct page | 44/44 | 44/44 |
| DOCX picture facts found in the chunk text | 6/8 | 2/8 |
| Retrieval Recall@1 (BM25, 18 queries, page-level) | 83.3% | 38.9% |
| Retrieval Recall@5 | 100% | 83.3% |
| Retrieval MRR | 0.917 | 0.563 |

Tokens: drawings 333,491 in / 6,900 out (3 files, 9 calls), slides 22,046 / 1,289 (4 calls),
DOCX 31,456 / 746 (4 calls).

How to read these numbers:

- Small synthetic set, one run. One field or one query moves a ratio by several points.
- 65/68 is the ceiling for this set: the second sheet of one drawing inherits sheet 1's title block
  (default `title_block_pages="first"`), so 3 of its fields differ by design.
- Baseline drawings fail the title-block gate (empty answer). For retrieval they are indexed with the
  drawing chunk minus the model output (file name, text layer, fields parsed from it), so the
  retrieval gap is the model output alone.
- Slides "must-mention" facts scored 1/6. Those facts are exact strings inside screenshots (`V-2864`,
  `Overdue orders: 4`). The slide prompt asks what a screen shows, not for a transcription (no full
  OCR), so most of them are not written out. A cost of that rule, not misreads.
- The two DOCX misses are paraphrases ("hanging loop" for "carry hook", "green collar" for
  "green status ring"); facts are matched as substrings.
- The synthetic DOCX has correct page-break markers, so the baseline already places every row.
  The 44/44 shows the VLM path does not break it, not that it repairs bad markers.

Full tables and metric definitions: [examples/recordings/gpt-6-luna/results.md](examples/recordings/gpt-6-luna/results.md),
[docs/evaluation.md](docs/evaluation.md).

## Quick start

Python 3.10+. LibreOffice is needed for slide and DOCX page renders (also when replaying them);
drawings do not need it.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# synthetic documents + ground truth
python -m vlmrag.synthetic --out work/synthetic

# reproduce the results above without an API key (recorded model answers)
python -m vlmrag.evaluation --replay-dir examples/recordings/gpt-6-luna --out-dir out/eval-replay
```

Replay matches requests by their text and image labels, not pixels. A different pypdfium2 or
LibreOffice version can still change a request; the replay then reports misses instead of guessing.

With a model:

```bash
cp .env.example .env            # set provider, model, key
vlmrag run drawing work/synthetic/drawing --out out/drawing.jsonl
vlmrag run slides  work/synthetic/slides  --out out/slides.jsonl
vlmrag run docx    work/synthetic/docx    --out out/docx.jsonl
```

`--record FILE` saves every model answer, `--replay FILE` runs from such a file without network,
`--provider null` runs the whole pipeline with empty answers (fallback paths only; drawings then fail
the title-block gate by design).

Demos: `python examples/drawing_demo.py --provider null` (same for `slides_demo.py`, `docx_demo.py`).
Without `--provider null` they call the configured model. `drawing_demo.py --simulate` answers from the
ground truth instead.

## Lessons

- **Image size cap.** The original sent close-ups without a cap (an A3 metadata-table overview reaches
  ~280 MP). `gpt-6-luna` rejects those requests (image patch limit). In a separate run on the same
  3 drawings, 4 MP read as many fields as 8 MP and 2 more than 16 MP, with 28% of the 16 MP input
  tokens, so 4 MP is the default (`VLMRAG_DRAWING_MAX_IMAGE_PIXELS`, 0 = no cap):

  | cap | fields | input tokens (3 drawings) |
  |---|---|---|
  | none | rejected | - |
  | 16 MP | 63/68 | 1,189,152 |
  | 8 MP | 65/68 | 643,750 |
  | 4 MP | 65/68 | 333,334 |

- **Summary prefix is not free.** On numeric tables a model-written summary in front of the chunk text
  added numbers that are not in the source (rounded or recombined) and took a large share of the
  embedding model's 8,192-token window, pushing real table rows out. It is opt-in per document type
  (`VLMRAG_RETRIEVAL_SUMMARY`) and not used for the results above.
- **Let the model assign, not copy.** For tables that cross pages the model answers row indexes per
  page, never cell text. A wrong answer is easy to check (every row exactly once, pages inside the
  span, rows in page order) and is rejected as a whole; nothing it says can change the table content.
- **`multiprocessing.Semaphore` at module level** limits concurrency only under `fork`. Under
  `spawn` / `forkserver` every process gets its own semaphore. The limit lives in `VLMClient`, and VLM
  calls run in threads.
- **LibreOffice** needs its own profile per call to run in parallel, a timeout that kills the whole
  process tree, and on Windows a short profile path (a long one crashes it).

## Layout

```
src/vlmrag/
  techniques/{drawing,slides,docx}/   the three techniques
  vlm/                                request types, backends, client, JSON recovery, cache, usage
  render/                             PDF render (pypdfium2), tiling, LibreOffice conversion
  augment/                            optional retrieval-summary prefix
  synthetic/                          synthetic documents + ground truth
  evaluation/                         metrics, BM25, record/replay runner
examples/                             demos and recorded model answers
configs/                              example title-block schema and domain terms
docs/                                 technique notes and evaluation
scripts/release_check.py              pre-publish scan (keys, private addresses, e-mails, file types)
```

## License

MIT
