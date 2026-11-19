"""Field entries, tags, parts, synonyms and layout columns."""

from __future__ import annotations

from vlmrag.techniques.drawing.config import DomainTerms, TitleBlockSchema
from vlmrag.techniques.drawing.entries import (
    build_field_entries,
    domain_synonym_text,
    entries_from_text,
    entries_text,
    entry_part,
    entry_sub_part,
    entry_tags,
    field_key_for_topic,
    field_values,
    is_markup_fragment,
    layout_columns,
    layout_columns_text,
    make_entry,
    merge_entries,
)

SCHEMA = TitleBlockSchema.default()
DOMAIN = DomainTerms.default()


def test_field_key_for_topic():
    assert field_key_for_topic("DWG. NO.", SCHEMA, DOMAIN) == "drawing_number"
    assert field_key_for_topic("Housing vent holes", SCHEMA, DOMAIN) == "vent_hole"
    assert field_key_for_topic("Paint / Code", SCHEMA, DOMAIN) == "paint___code"
    assert field_key_for_topic("x" * 200, SCHEMA, DOMAIN) == "x" * 80
    assert field_key_for_topic("Housing vent holes", SCHEMA, DomainTerms.empty()) == "housing_vent_holes"


def test_tags_parts_and_sub_parts():
    assert entry_tags("Finish LH", "HSG matte black coating", DOMAIN) == ["surface_finish"]
    assert entry_tags("Breather hole", "dia 3.2", DOMAIN) == ["vent_hole"]
    assert entry_part("Finish LH", "HSG matte", DOMAIN) == "housing"
    assert entry_part("Top cap", "", DOMAIN) == "cap"
    assert entry_part("Rev", "C", DOMAIN) == ""
    assert entry_sub_part("Finish lh", DOMAIN) == "LH"
    assert entry_sub_part("Finish", DOMAIN) == ""
    assert entry_sub_part("Finish LH", DomainTerms.empty()) == ""


def test_make_entry():
    entry = make_entry("", "", "raw_chunk", schema=SCHEMA, domain=DOMAIN, source_text="VENT HOLE DIA 3.2 " + "x" * 600)
    assert entry["value"].startswith("VENT HOLE") and len(entry["source_text"]) == 500
    assert entry["field_key"] == "" or entry["topic_raw"] == ""
    assert make_entry("", "", "raw_chunk", schema=SCHEMA, domain=DOMAIN) is None
    entry = make_entry("Material", "PC", "title_block", schema=SCHEMA, domain=DOMAIN)
    assert entry["field_key"] == "material" and entry["source_text"] == "Material: PC"
    assert entry["topic_norm"] == "material" and entry["value_norm"] == "pc"


def test_merge_entries_dedupes_and_limits():
    a = make_entry("Material", "PC", "title_block", schema=SCHEMA, domain=DOMAIN)
    b = make_entry("material:", "pc", "title_block_table", schema=SCHEMA, domain=DOMAIN)
    c = make_entry("Mass", "0.4 kg", "title_block", schema=SCHEMA, domain=DOMAIN)
    assert len(merge_entries([a, b], [c, "junk"])) == 2
    assert len(merge_entries([a, c], limit=1)) == 1


def test_build_field_entries_order_and_sources():
    entries = build_field_entries(
        fields={"drawing_number": "EX-1", "mass": ""},
        pairs=[{"topic": "Drawing No.", "value": "EX-1"}],
        table_pairs=[{"topic": "Finish", "value": "HOUSING: MATTE"}],
        schema=SCHEMA,
        domain=DOMAIN,
    )
    assert [e["source"] for e in entries] == ["title_block", "title_block_table", "title_block_field"]
    assert entries[2]["topic_raw"] == "Drawing Number" and entries[2]["field_key"] == "drawing_number"
    assert entries[1]["intent_tags"] == ["surface_finish"] and entries[1]["part"] == "housing"


def test_entries_from_text_only_takes_signal_lines():
    text = "\n".join(
        [
            "1. UNLESS OTHERWISE SPECIFIED ALL DIMENSIONS ARE IN MILLIMETRES.",
            "9. VENT HOLE DIA 3.2 THRU, 6 PLACES EQUALLY SPACED.",
            "4. SURFACE FINISH: SEE TITLE BLOCK.",
            "<td>coating</td>",
            "GASKET = EPDM 70",
        ]
    )
    entries = entries_from_text(text, "raw_chunk", schema=SCHEMA, domain=DOMAIN, chunk="page 1")
    # The topic/value splitter needs at least one character before the term (a documented
    # limitation), so a line that starts with the term is kept whole as topic and value.
    assert [e["topic_raw"] for e in entries] == [
        "9. VENT HOLE DIA 3.2 THRU, 6 PLACES EQUALLY SPACED.",
        "4. SURFACE FINISH",
        "GASKET = EPDM 70",
    ]
    assert entries[1]["value"] == "SEE TITLE BLOCK." and entries[2]["value"] == "GASKET = EPDM 70"
    split = entries_from_text("Lid GASKET = EPDM 70", "raw_chunk", schema=SCHEMA, domain=DOMAIN)
    assert (split[0]["topic_raw"], split[0]["value"]) == ("Lid GASKET", "EPDM 70")
    assert entries[0]["value"] == entries[0]["source_text"] and entries[0]["chunk"] == "page 1"
    assert entries_from_text(text, "raw_chunk", schema=SCHEMA, domain=DomainTerms.empty()) == []
    assert is_markup_fragment("<td>coating</td>")
    # A configured pattern with its own capturing group must not shift the topic/value split.
    grouped = DomainTerms.from_dict({"groups": [{"tag": "vent", "terms": ["Vent"], "patterns": ["(vent|breather)\\s*hole"]}]})
    split = entries_from_text("Lid vent hole: 3.2 THRU", "raw_chunk", schema=SCHEMA, domain=grouped)
    assert (split[0]["topic_raw"], split[0]["value"]) == ("Lid vent hole", "3.2 THRU")


def test_entries_text_and_values():
    entries = [
        make_entry("Finish LH", "HSG: MATTE", "title_block", schema=SCHEMA, domain=DOMAIN),
        make_entry("Finish RH", "HSG: MATTE", "title_block", schema=SCHEMA, domain=DOMAIN),
        make_entry("Rev", "C", "title_block", schema=SCHEMA, domain=DOMAIN),
    ]
    text = entries_text(entries + [None])
    assert text.splitlines()[0] == "- Finish LH: HSG: MATTE (field=surface_finish / part=housing / sub_part=LH / tags=surface_finish)"
    assert text.splitlines()[2] == "- Rev: C (field=revision)"
    assert field_values(entries) == {"surface_finish": ["HSG: MATTE"], "revision": ["C"]}


def test_layout_columns_and_synonyms():
    pairs = [{"topic": "Rev", "value": "C"}, {"topic": "Rev", "value": "D"}, {"topic": "Mass", "value": ""}, {"topic": "Scale", "value": "1:2"}]
    assert layout_columns(pairs) == ["Rev", "Scale"]
    assert layout_columns_text(["Rev", "Scale"]) == "Title block layout columns: Rev | Scale"
    assert layout_columns_text([]) == ""
    text = domain_synonym_text("VENT HOLE DIA 3.2 and a gasket", DOMAIN)
    assert text.splitlines() == ["Domain synonym: Vent hole = Venthole = Breather hole", "Domain synonym: Gasket = Seal = O-ring"]
    assert domain_synonym_text("nothing here", DOMAIN) == ""
