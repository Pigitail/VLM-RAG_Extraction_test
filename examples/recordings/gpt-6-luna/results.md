# Evaluation results

|  |  |
|---|---|
| Label | synthetic set seed 0, azure gpt-6-luna |
| Date | 2026-11-19 |
| Model | azure / gpt-6-luna |
| Model reported by the responses | gpt-6-luna |
| Mode | record (backend: azure) |
| Package | vlmrag 0.1.0 |
| Documents | 3 drawing, 1 slides, 1 docx (synthetic, seed 0) |
| Reasoning effort setting | low |

Synthetic documents (5), a single run, small N: read every ratio with its count and see the caveats at the end.

## Engineering drawings: title-block fields

| Run | Sheets | Fields exact | Fields normalised | Read on own sheet (norm., successful docs only) | Raster-only title block (norm.) | Gate passed |
|---|---|---|---|---|---|---|
| VLM, cap (4,000,000 px) | 4 | 95.6% (65/68) | 95.6% (65/68) | 100.0% (51/51) | 100.0% (17/17) | 3/3 |
| Text-only baseline | 4 | 0.0% (0/68) | 0.0% (0/68) | n/a | 0.0% (0/17) | 0/3 |

Normalised accuracy per field (all sheets):

| Field | VLM, cap (4,000,000 px) | Text-only baseline |
|---|---|---|
| company | 100.0% | 0.0% |
| department | 100.0% | 0.0% |
| product | 100.0% | 0.0% |
| sheet_size | 100.0% | 0.0% |
| projection | 100.0% | 0.0% |
| document_type | 100.0% | 0.0% |
| title | 75.0% | 0.0% |
| drawing_number | 100.0% | 0.0% |
| part_number | 100.0% | 0.0% |
| revision | 100.0% | 0.0% |
| date_of_issue | 100.0% | 0.0% |
| scale | 75.0% | 0.0% |
| sheet | 75.0% | 0.0% |
| material | 100.0% | 0.0% |
| general_tolerance | 100.0% | 0.0% |
| mass | 100.0% | 0.0% |
| surface_finish | 100.0% | 0.0% |

Per sheet (normalised field accuracy):

| Sheet | Raster-only regions | VLM, cap (4,000,000 px) | Text-only baseline |
|---|---|---|---|
| lantern_bracket.pdf p1 | material_table | 100.0% | 0.0% (TitleBlockGateError) |
| lantern_housing.pdf p1 | revision_table | 100.0% | 0.0% (TitleBlockGateError) |
| lantern_housing.pdf p2 | none | 82.4% (inherited) | 0.0% (TitleBlockGateError) |
| lantern_reflector_scan.pdf p1 | revision_table, title_block | 100.0% | 0.0% (TitleBlockGateError) |

- Text-only baseline: failed documents: 3 gate

## Software-manual slides

| Run | Image-role accuracy | Must-mention recall | Hallucinated blocks | Visual calls ok |
|---|---|---|---|---|
| VLM | 100.0% (7/7) | 16.7% (1/6) | 0 | 4/4 |
| Text-only baseline | 0.0% (0/7) | 0.0% (0/6) | 0 | 0/4 |

Per slide:

| Slide | Image blocks | Roles correct (VLM) | Must-mention (VLM) | Roles correct (Text-only baseline) | Must-mention (Text-only baseline) |
|---|---|---|---|---|---|
| sample_erp_manual.pptx s1 | 0 | - | - | - | - |
| sample_erp_manual.pptx s2 | 1 | 1/1 | 0.0% (0/2) | 0/1 | 0.0% (0/2) |
| sample_erp_manual.pptx s3 | 2 | 2/2 | 50.0% (1/2) | 0/2 | 0.0% (0/2) |
| sample_erp_manual.pptx s4 | 3 | 3/3 | 0.0% (0/1) | 0/3 | 0.0% (0/1) |
| sample_erp_manual.pptx s5 | 1 | 1/1 | 0.0% (0/1) | 0/1 | 0.0% (0/1) |
| sample_erp_manual.pptx s6 | 0 | - | - | - | - |

## Specification documents (DOCX)

| Run | Page count correct | Row->page accuracy | Row placement | Image must-mention recall |
|---|---|---|---|---|
| VLM | 100.0% (1/1) | 100.0% (44/44) | vlm: 2 page(s) | 75.0% (6/8) |
| Text-only baseline | 100.0% (1/1) | 100.0% (44/44) | xml_row_markers: 2 page(s) | 25.0% (2/8) |

Image must-mention recall per page:

| Page | Figures | VLM | Text-only baseline |
|---|---|---|---|
| lantern_spec.docx p1 | fig1 | 100.0% (3/3) | 0.0% (0/3) |
| lantern_spec.docx p2 | fig2 | 100.0% (3/3) | 66.7% (2/3) |
| lantern_spec.docx p4 | fig3 | 0.0% (0/2) | 0.0% (0/2) |

## Retrieval (BM25 over all chunks, page-level relevance)

| Index | Queries | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| VLM, drawing cap (4,000,000 px) | 18 | 83.3% | 100.0% | 100.0% | 0.917 |
| Text-only baseline | 18 | 38.9% | 66.7% | 83.3% | 0.563 |

Baseline drawings that fail the title-block gate are indexed with the technique's own chunks minus the model output: lantern_bracket.pdf, lantern_housing.pdf, lantern_reflector_scan.pdf.

Per technique:

| Index | Queries | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| VLM, drawing cap (4,000,000 px) - docx | 4 | 100.0% | 100.0% | 100.0% | 1.000 |
| VLM, drawing cap (4,000,000 px) - drawing | 9 | 77.8% | 100.0% | 100.0% | 0.889 |
| VLM, drawing cap (4,000,000 px) - slides | 5 | 80.0% | 100.0% | 100.0% | 0.900 |
| Text-only baseline - docx | 4 | 75.0% | 100.0% | 100.0% | 0.875 |
| Text-only baseline - drawing | 9 | 11.1% | 44.4% | 77.8% | 0.350 |
| Text-only baseline - slides | 5 | 60.0% | 80.0% | 80.0% | 0.695 |

## Cost and latency

| Run | Docs (failed) | Answers (incl. cached) | Cached | Input tokens | Output tokens | Reasoning tokens | Largest image sent (MP) | Reasoning effort sent | Model latency s | Wall time s |
|---|---|---|---|---|---|---|---|---|---|---|
| vlm/drawing@cap | 3 (0) | 9 | 0 | 333,491 | 6,900 | 693 | 4.0 | low | 71.7 | 60.7 |
| vlm/slides | 1 (0) | 4 | 0 | 22,046 | 1,289 | 0 | 3.2 | low | 19.0 | 10.5 |
| vlm/docx | 1 (0) | 4 | 0 | 31,456 | 746 | 163 | 4.7 | low, medium | 20.1 | 13.6 |
| baseline/drawing@cap (stub, no model) | 3 (3) | 9 | 0 | 0 | 0 | 0 | 4.0 | low | 0.0 | 15.0 |
| baseline/slides (stub, no model) | 1 (0) | 4 | 0 | 0 | 0 | 0 | 3.2 | low | 0.0 | 5.2 |
| baseline/docx (stub, no model) | 1 (0) | 4 | 0 | 0 | 0 | 0 | 4.7 | low, medium | 0.0 | 5.4 |

Baseline runs answer every request locally with '{}' (NullBackend): no model is contacted and no tokens are spent.

## Caveats

- Synthetic documents only (3 drawing, 1 slides, 1 docx); invented content generated by vlmrag.synthetic. Numbers describe these fixtures, not real document collections.
- Small N: every ratio comes with its count; one field, block or query moves a ratio by several points.
- Single run: model answers vary between runs and no confidence intervals are reported.
- Retrieval is lexical BM25 over retrieve_content; an embedding retriever would also match paraphrases. Must-mention facts are matched as normalised substrings, so a correct paraphrase counts as not found.
- The text-only baseline never contacts a model: every request it builds is answered locally with '{}' (NullBackend), so its calls are stub answers with zero tokens.
- Drawings use the technique default title_block_pages='first': later sheets carry the title block read on sheet 1 ('inherited' group), so sheet-specific fields (title, scale, sheet) differ there by design.
- The text-only drawing baseline fails the title-block gate by design (an empty answer yields no pairs); for retrieval it is represented by the technique's own chunks with an empty title block (text layer, fields parsed from it, domain synonyms), so the retrieval difference is the model output alone. A drawing whose VLM run fails contributes nothing to the VLM index.
- The synthetic DOCX carries correct rendered-page markers, so the deterministic row fallback is exact by construction: row->page accuracy shows whether the model path keeps rows on their pages, not whether it repairs stale markers.

## Metric definitions

- **normalization**: Normalised comparisons first unify dash and quote variants, casefold, collapse whitespace runs to one space, strip. Exact comparisons compare the raw strings.
- **failures**: A document whose extraction raised is scored, not skipped: every truth item of it counts as wrong or not found. Ratios with an empty denominator are reported as n/a.
- **drawing.field_accuracy**: Per title-block field: fraction of truth sheets whose emitted metadata.title_block[field] equals the truth value (exact and normalised). Fields with an empty truth value are not scored.
- **drawing.sheet_accuracy**: Per sheet (source page): fraction of that sheet's scored fields that match. Group rows report micro accuracy over all fields of the group and sheet_mean_normalized, the mean of per-sheet accuracies.
- **drawing.groups**: all = every truth sheet; read = sheets whose chunk carries a title block read on that same sheet (conditional on success: sheets of failed documents are not in it, so compare it with 'all'); inherited = sheets that received the title block of another page (title_block_pages='first'); raster_title_block = sheets whose truth marks the title block as raster-only (no text layer); text_layer_title_block = the other sheets. all, raster_title_block and text_layer_title_block include the sheets of failed documents, scored as wrong.
- **drawing.gate_pass_rate**: Documents that returned chunks / documents attempted. gate_failed counts TitleBlockGateError (fewer than min_pairs pairs); other_errors counts failures before the gate (transport, empty answer, replay miss).
- **drawing.variants**: Each drawing variant is a separate run of the same documents that differs only in DrawingConfig.max_image_pixels (cap = packaged default, nocap = 0 = no cap, the original behaviour).
- **slides.image_role_accuracy**: Over all truth image blocks: the first emitted description attached to the same shape (matched by shape name) has image_role equal to the truth role. A block without a description counts as wrong.
- **slides.must_mention_recall**: Over all must-mention facts of all slides: the normalised fact is a substring of the normalised retrieve_content of that slide's chunk(s). slide_mean_recall averages per-slide recall over slides with at least one fact.
- **slides.hallucinated_blocks**: Emitted descriptions that do not describe a distinct truth image block: not tied to any image block (ungrounded), tied to an image not listed in the truth (not_in_truth), or a second description of a block already described (duplicate).
- **docx.page_count_correct**: The page count of the technique's page split (provenance.page_split xml_pages + padded_pages, else the highest chunk page) equals truth page_count.
- **docx.row_page_accuracy**: Over all rows (header included) of every truth table that spans pages: the set of pages whose chunk metadata.multi_page_tables lists the row index is exactly {truth page}. Rows missing or listed on several pages count as wrong. assignment counts page entries by source (vlm or xml_row_markers).
- **docx.image_must_mention_recall**: Over all must-mention facts of pages with pictures: the normalised fact is a substring of the normalised retrieve_content of that page's chunk(s).
- **retrieval.index**: One BM25 index (k1=1.2, b=0.75, NFKC+casefold word tokens, distinct query terms) per run over the retrieve_content of ALL chunks of ALL documents of that run. Queries are the truth-file queries of the run's documents; relevance is the (source file name, 1-based page) pairs listed in the truth.
- **retrieval.ranking**: Chunk scores are collapsed to pages (a page scores its best chunk; duplicate chunks of a page count once); pages scoring 0 are not retrieved. Ties are broken against relevant pages, so ranks are lower bounds.
- **retrieval.recall_at_k**: |relevant pages among the top k ranked pages| / |relevant pages|, averaged over queries.
- **retrieval.mrr**: Mean over queries of 1 / rank of the first relevant page (0 when none is retrieved).
- **retrieval.baseline**: The text-only baseline indexes the chunks of the NullBackend runs. A drawing whose baseline run raised (the acceptance gate rejects an empty answer by design) is represented by the drawing technique's own chunks built with an empty title block - file name, page, text layer, fields parsed from the text layer, domain synonyms - so it differs from a VLM chunk only by what the model read (listed under model_free_substitutes). A drawing whose VLM run failed contributes no chunks to the VLM index, as it would not be indexed in production.
- **cost**: Per technique run (and drawing variant), from the VLM client's usage ledger: calls = every answer the technique received (re-asks included, cached answers included), cached = the subset served from --cache-dir with no tokens or latency, input/output/reasoning tokens as reported by the provider, model latency (sum of per-call latencies, so concurrent calls overlap) and wall time of the whole run. requests_sent (per run) and max_image_pixels count what was actually sent to the backend, every attempt included; reasoning_effort lists the efforts sent per task. Replayed runs report the usage stored in the recording. Baseline runs answer locally with NullBackend: zero tokens.
