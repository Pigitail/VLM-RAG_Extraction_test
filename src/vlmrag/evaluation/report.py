"""Write ``results.json`` and ``results.md`` (tables ready to paste into a README).

The Markdown is generated only from ``results.json``, so the two can never
disagree and a results file can be re-rendered later
(:func:`render_markdown`). Every ratio is shown with its count, because the
synthetic set is small and "75 %" means little without "(3/4)".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

__all__ = ["write_results", "render_markdown", "pct"]


def pct(value: Optional[float], numerator: Optional[int] = None, denominator: Optional[int] = None) -> str:
    """``0.925`` -> ``"92.5%"``; with counts ``"92.5% (37/40)"``; ``None`` -> ``"n/a"``."""
    if value is None:
        return "n/a"
    text = f"{100.0 * value:.1f}%"
    if numerator is not None and denominator is not None:
        text += f" ({numerator}/{denominator})"
    return text


def _num(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def _variant_label(results: dict, name: Optional[str]) -> str:
    for variant in results.get("evaluation", {}).get("drawing_variants", []):
        if variant["name"] == name:
            pixels = variant["max_image_pixels"]
            return f"{name} ({pixels:,} px)" if pixels else f"{name} (no cap)"
    return str(name)


def _count_ratio(found: int, total: int) -> str:
    return pct(found / total if total else None, found, total)


# --------------------------------------------------------------------------- sections


def _header(results: dict) -> list[str]:
    model = results.get("model", {})
    docs = results.get("documents", {})
    counts: dict[str, int] = {}
    for item in docs.get("items", []):
        counts[item["technique"]] = counts.get(item["technique"], 0) + 1
    origin = f"synthetic, seed {docs.get('seed')}" if docs.get("generated") else "user-supplied directory"
    reported = model.get("reported_by_responses") or []
    rows = [
        ("Label", results.get("label") or "-"),
        ("Date", results.get("date") or "-"),
        ("Model", f"{model.get('provider', '-')} / {model.get('model', '-')}"),
        # On Azure the configured name is a deployment; this is what the responses said answered.
        ("Model reported by the responses", ", ".join(reported) or "-"),
        ("Mode", f"{results.get('mode', '-')} (backend: {model.get('backend', '-')})"),
        ("Package", f"vlmrag {results.get('package_version', '-')}"),
        ("Documents", ", ".join(f"{n} {t}" for t, n in counts.items()) + f" ({origin})"),
        ("Reasoning effort setting", results.get("settings", {}).get("reasoning_effort") or "none"),
    ]
    out = ["# Evaluation results", ""] + _table(["", ""], rows)
    # The headline caveats sit next to the numbers, so a table copied out of this file is not read without them.
    notes = []
    if results.get("mode") == "oracle":
        notes.append("**Oracle run: every answer comes from the ground truth. These numbers check the harness; "
                     "they are not model results.**")
    if docs.get("generated"):
        notes.append(f"Synthetic documents ({sum(counts.values())}), a single run, small N: read every ratio with "
                     "its count and see the caveats at the end.")
    else:
        notes.append("A single run with small N: read every ratio with its count and see the caveats at the end.")
    if results.get("mode") == "replay":
        notes.append("Replayed from a recording: the answers, token counts, label, date and settings are the "
                     "recorded run's.")
    for note in notes:
        out += ["", note]  # one paragraph each
    return out


def _drawing(results: dict) -> list[str]:
    section = results["metrics"].get("drawing")
    if not section:
        return []
    out = ["", "## Engineering drawings: title-block fields", ""]
    runs: list[tuple[str, dict]] = []
    for variant, data in (section.get("vlm") or {}).items():
        runs.append((f"VLM, {_variant_label(results, variant)}", data))
    for variant, data in (section.get("baseline") or {}).items():
        runs.append(("Text-only baseline", data))

    def group(data: dict, name: str) -> str:
        g = data["groups"][name]
        return pct(g["normalized"], g["normalized_correct"], g["fields"]) if g["fields"] else "n/a"

    rows = []
    for label, data in runs:
        all_group = data["groups"]["all"]
        gate = data["gate"]
        rows.append((
            label,
            all_group["sheets"],
            pct(all_group["exact"], all_group["exact_correct"], all_group["fields"]),
            pct(all_group["normalized"], all_group["normalized_correct"], all_group["fields"]),
            group(data, "read"),
            group(data, "raster_title_block"),
            f"{gate['passed']}/{gate['attempted']}",
        ))
    out += _table(
        ["Run", "Sheets", "Fields exact", "Fields normalised", "Read on own sheet (norm., successful docs only)",
         "Raster-only title block (norm.)", "Gate passed"],
        rows,
    )
    keys: list[str] = []
    for _, data in runs:
        for key in data["per_field"]:
            if key not in keys:
                keys.append(key)
    if keys:
        out += ["", "Normalised accuracy per field (all sheets):", ""]
        out += _table(
            ["Field"] + [label for label, _ in runs],
            [[key] + [pct((data["per_field"].get(key) or {}).get("normalized")) for _, data in runs] for key in keys],
        )
    sheets: dict[tuple[str, int], dict] = {}
    for _, data in runs:
        for sheet in data["sheets"]:
            sheets.setdefault((sheet["source"], sheet["page"]), sheet)
    if sheets:
        out += ["", "Per sheet (normalised field accuracy):", ""]
        rows = []
        for (source, page), first in sorted(sheets.items()):
            row = [f"{source} p{page}", ", ".join(first["raster_regions"]) or "none"]
            for _, data in runs:
                match = next((s for s in data["sheets"] if s["source"] == source and s["page"] == page), None)
                if match is None:
                    row.append("-")
                else:
                    note = " (inherited)" if match["inherited"] else ""
                    note = f" ({match['error_type']})" if match["status"] == "error" else note
                    row.append(pct(match["normalized"]) + note)
            rows.append(row)
        out += _table(["Sheet", "Raster-only regions"] + [label for label, _ in runs], rows)
    errors = []
    for label, data in runs:
        other = data["gate"]["other_errors"]
        if data["gate"]["gate_failed"] or other:
            parts = [f"{data['gate']['gate_failed']} gate"] + [f"{n} {k}" for k, n in other.items()]
            errors.append(f"- {label}: failed documents: {', '.join(parts)}")
    if errors:
        out += [""] + errors
    return out


def _slides(results: dict) -> list[str]:
    section = results["metrics"].get("slides")
    if not section:
        return []
    out = ["", "## Software-manual slides", ""]
    runs = [(label, section[key]) for key, label in (("vlm", "VLM"), ("baseline", "Text-only baseline")) if key in section]
    rows = []
    for label, data in runs:
        role, must, calls = data["image_role"], data["must_mention"], data["visual_calls"]
        rows.append((
            label,
            pct(role["accuracy"], role["correct"], role["blocks"]),
            pct(must["recall"], must["found"], must["facts"]),
            data["hallucinated_blocks"]["total"],
            f"{calls['ok']}/{calls['expected']}",
        ))
    out += _table(["Run", "Image-role accuracy", "Must-mention recall", "Hallucinated blocks", "Visual calls ok"], rows)
    if runs:
        per_slide = {}
        for label, data in runs:
            for row in data["per_slide"]:
                per_slide.setdefault((row["source"], row["slide"]), {})[label] = row
        out += ["", "Per slide:", ""]
        header = ["Slide", "Image blocks"]
        for label, _ in runs:
            header += [f"Roles correct ({label})", f"Must-mention ({label})"]
        table_rows = []
        for (source, slide), by_label in sorted(per_slide.items()):
            any_row = next(iter(by_label.values()))
            row = [f"{source} s{slide}", len(any_row["image_blocks"])]
            for label, _ in runs:
                r = by_label.get(label)
                if r is None:
                    row += ["-", "-"]
                    continue
                facts = r["must_mention"]
                row += [
                    f"{r['role_correct']}/{len(r['image_blocks'])}" if r["image_blocks"] else "-",
                    _count_ratio(sum(f["found"] for f in facts), len(facts)) if facts else "-",
                ]
            table_rows.append(row)
        out += _table(header, table_rows)
    return out


def _docx(results: dict) -> list[str]:
    section = results["metrics"].get("docx")
    if not section:
        return []
    out = ["", "## Specification documents (DOCX)", ""]
    runs = [(label, section[key]) for key, label in (("vlm", "VLM"), ("baseline", "Text-only baseline")) if key in section]
    rows = []
    for label, data in runs:
        pages, rp, must = data["page_count"], data["row_page"], data["image_must_mention"]
        sources: dict[str, int] = {}
        for table in data["per_table"]:
            for name, n in table["assignment"].items():
                sources[name] = sources.get(name, 0) + n
        rows.append((
            label,
            pct(pages["accuracy"], pages["correct"], pages["documents"]),
            pct(rp["accuracy"], rp["correct"], rp["rows"]),
            ", ".join(f"{k}: {v} page(s)" for k, v in sorted(sources.items())) or "-",
            pct(must["recall"], must["found"], must["facts"]),
        ))
    out += _table(["Run", "Page count correct", "Row->page accuracy", "Row placement", "Image must-mention recall"], rows)
    pages_seen: dict[tuple[str, int], dict] = {}
    for label, data in runs:
        for page in data["per_page"]:
            pages_seen.setdefault((page["source"], page["page"]), {})[label] = page
    if pages_seen:
        out += ["", "Image must-mention recall per page:", ""]
        table_rows = []
        for (source, page), by_label in sorted(pages_seen.items()):
            any_page = next(iter(by_label.values()))
            row = [f"{source} p{page}", ", ".join(any_page["figures"]) or "-"]
            for label, _ in runs:
                p = by_label.get(label)
                row.append(_count_ratio(sum(f["found"] for f in p["must_mention"]), len(p["must_mention"])) if p else "-")
            table_rows.append(row)
        out += _table(["Page", "Figures"] + [label for label, _ in runs], table_rows)
    return out


def _retrieval_label(results: dict, name: str) -> str:
    if name == "baseline":
        return "Text-only baseline"
    if name.startswith("vlm@"):
        return f"VLM, drawing {_variant_label(results, name[4:])}"
    return "VLM"


def _retrieval(results: dict) -> list[str]:
    section = results["metrics"].get("retrieval") or {}
    if not section:
        return []
    ks = results.get("evaluation", {}).get("retrieval", {}).get("ks", [1, 3, 5])
    out = ["", "## Retrieval (BM25 over all chunks, page-level relevance)", ""]
    header = ["Index", "Queries"] + [f"Recall@{k}" for k in ks] + ["MRR"]

    def row(label: str, summary: dict) -> list:
        return [label, summary["queries"]] + [pct(summary.get(f"recall@{k}")) for k in ks] + [_num(summary.get("mrr"))]

    out += _table(header, [row(_retrieval_label(results, n), d["overall"]) for n, d in section.items()])
    substitutes = (section.get("baseline") or {}).get("model_free_substitutes") or []
    if substitutes:
        out += ["", "Baseline drawings that fail the title-block gate are indexed with the technique's own chunks "
                "minus the model output: " + ", ".join(substitutes) + "."]
    by_type_rows = []
    for name, data in section.items():
        for doc_type, summary in data["by_doc_type"].items():
            by_type_rows.append(row(f"{_retrieval_label(results, name)} - {doc_type}", summary))
    if by_type_rows:
        out += ["", "Per technique:", ""] + _table(header, by_type_rows)
    return out


def _efforts(row: dict) -> str:
    efforts = sorted({e for values in (row.get("reasoning_effort") or {}).values() for e in values})
    return ", ".join(efforts) or "-"


def _cost(results: dict) -> list[str]:
    rows = []
    baseline = False
    for row in results.get("cost", []):
        label = row["run"]
        if row.get("mode") == "baseline":
            label += " (stub, no model)"
            baseline = True
        pixels = row.get("max_image_pixels") or 0
        rows.append((
            label,
            f"{row['documents']} ({row['failed']})",
            row["calls"],
            row["cached"],
            f"{row['input_tokens']:,}",
            f"{row['output_tokens']:,}",
            f"{row['reasoning_tokens']:,}",
            f"{pixels / 1e6:.1f}" if pixels else "-",
            _efforts(row),
            f"{row['model_latency_s']:.1f}",
            f"{row['wall_time_s']:.1f}",
        ))
    if not rows:
        return []
    out = ["", "## Cost and latency", ""]
    out += _table(
        ["Run", "Docs (failed)", "Answers (incl. cached)", "Cached", "Input tokens", "Output tokens",
         "Reasoning tokens", "Largest image sent (MP)", "Reasoning effort sent", "Model latency s", "Wall time s"],
        rows,
    )
    if baseline:
        out += ["", "Baseline runs answer every request locally with '{}' (NullBackend): no model is contacted "
                "and no tokens are spent."]
    if results.get("mode") == "replay":
        out += ["", "Replayed run: tokens and model latency are the recorded values; wall time is local."]
    return out


def _footer(results: dict) -> list[str]:
    out = ["", "## Caveats", ""] + [f"- {c}" for c in results.get("caveats", [])]
    if results.get("warnings"):
        out += ["", "## Warnings", ""] + [f"- {w}" for w in results["warnings"]]
    replay = results.get("replay")
    if replay:
        out += [
            "",
            "## Replay",
            "",
            f"- served {replay['served']} recorded answer(s), {replay['misses']} miss(es), "
            f"{replay['unused']} unused, {replay['recorded_failures_served']} recorded failure(s) replayed",
        ]
        check = results.get("replay_check")
        if check is not None:
            out.append(f"- matches the recorded results.json: {'yes' if check.get('matches_recording') else 'no'}")
    out += ["", "## Metric definitions", ""]
    out += [f"- **{name}**: {text}" for name, text in results.get("metric_definitions", {}).items()]
    return out


def render_markdown(results: dict) -> str:
    lines = _header(results) + _drawing(results) + _slides(results) + _docx(results) + _retrieval(results)
    lines += _cost(results) + _footer(results)
    return "\n".join(lines).rstrip() + "\n"


def write_results(results: dict, out_dir: Union[str, Path]) -> tuple[Path, Path]:
    """Write ``results.json`` and ``results.md`` into ``out_dir``; returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    md_path = out / "results.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(render_markdown(results), encoding="utf-8", newline="\n")
    return json_path, md_path
