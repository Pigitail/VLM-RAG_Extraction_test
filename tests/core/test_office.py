from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterator

import pytest

from vlmrag.render import office
from vlmrag.render.office import ConversionError, convert, find_soffice, soffice_candidates
from vlmrag.render.pdf import PdfPages

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/></Relationships>'
)
_DOCUMENT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Outdoor Lantern L-300 conversion check</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def _minimal_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _RELS)
        archive.writestr("word/document.xml", _DOCUMENT)
    return path


def _profile_dirs(root: Path) -> set[str]:
    return {p.name for p in root.glob("vlmrag_lo_*")}


@pytest.fixture
def private_profile_root(monkeypatch) -> Iterator[Path]:
    """A private, short profile root for the real-LibreOffice tests.

    Globbing the shared temp directory would also see the ``vlmrag_lo_*``
    profiles of other processes that run LibreOffice at the same time, which
    made the cleanup assertions flaky. ``tmp_path`` is not used because it is
    long, and a long profile path makes LibreOffice crash on Windows.
    """
    root = Path(tempfile.mkdtemp(prefix="lo"))
    monkeypatch.setenv("VLMRAG_LO_PROFILE_ROOT", str(root))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _script(directory: Path, name: str, windows_body: str, posix_body: str) -> str:
    """Create a fake 'soffice' launcher script for the current platform."""
    if sys.platform == "win32":
        path = directory / f"{name}.cmd"
        path.write_text("@echo off\r\n" + windows_body.replace("\n", "\r\n") + "\r\n", encoding="ascii")
    else:
        path = directory / name
        path.write_text("#!/bin/sh\n" + posix_body + "\n", encoding="ascii")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


_FAKE_CONVERTER = '''
import sys
from pathlib import Path
args = sys.argv[1:]
ext = args[args.index("--convert-to") + 1].split(":")[0]
out_dir = Path(args[args.index("--outdir") + 1])
src = Path(args[-1])
(out_dir / (src.stem + "." + ext)).write_text("converted", encoding="utf-8")
'''


def test_find_soffice_honours_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "my-soffice.exe"
    fake.write_text("", encoding="ascii")
    monkeypatch.setenv("VLMRAG_SOFFICE", str(fake))
    assert find_soffice() == str(fake)
    assert soffice_candidates()[0] == str(fake)
    preferred = tmp_path / "preferred-soffice.exe"
    preferred.write_text("", encoding="ascii")
    candidates = soffice_candidates(str(preferred))
    assert candidates[:2] == [str(preferred), str(fake)]
    assert soffice_candidates(str(fake)).count(str(fake)) == 1  # no duplicates
    assert str(tmp_path / "missing") not in soffice_candidates(str(tmp_path / "missing"))


def test_no_binary_gives_clear_error(tmp_path):
    src = tmp_path / "a.docx"
    src.write_bytes(b"x")
    with pytest.raises(ConversionError, match="not found"):
        convert(src, "pdf", tmp_path / "out", candidates=[])


def test_missing_binary_falls_through_and_reports_every_attempt(tmp_path):
    src = tmp_path / "a.docx"
    src.write_bytes(b"x")
    missing = [str(tmp_path / "missing-1" / "soffice"), str(tmp_path / "missing-2" / "soffice")]
    with pytest.raises(ConversionError) as info:
        convert(src, "pdf", tmp_path / "out", candidates=missing)
    assert [binary for binary, _ in info.value.attempts] == missing
    assert all("could not start" in reason for _, reason in info.value.attempts)


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path / "nope.docx", "pdf", tmp_path)


def test_timeout_kills_and_falls_back_to_next_candidate(tmp_path):
    """The internal bug: a TimeoutExpired escaped and the fallback binary was never tried."""
    marker = tmp_path / "slow-started.txt"
    converter = tmp_path / "fake_convert.py"
    converter.write_text(_FAKE_CONVERTER, encoding="utf-8")
    slow = _script(
        tmp_path,
        "slow-soffice",
        f'echo started> "{marker}"\nping -n 30 127.0.0.1 > nul',
        f'echo started > "{marker}"\nsleep 30',
    )
    fast = _script(
        tmp_path,
        "fast-soffice",
        f'"{sys.executable}" "{converter}" %*',
        f'exec "{sys.executable}" "{converter}" "$@"',
    )
    src = tmp_path / "Spec Sheet.docx"
    src.write_bytes(b"not really a docx")
    started = time.monotonic()
    output = convert(src, "pdf", tmp_path / "out", timeout_s=3, candidates=[slow, fast])
    assert time.monotonic() - started < 25  # the slow launcher was killed, not waited for
    assert marker.exists()  # the first candidate really ran
    assert output == (tmp_path / "out" / "Spec Sheet.pdf").resolve()
    assert output.read_text(encoding="utf-8") == "converted"


def test_failing_candidate_then_success(tmp_path):
    converter = tmp_path / "fake_convert.py"
    converter.write_text(_FAKE_CONVERTER, encoding="utf-8")
    failing = _script(tmp_path, "failing-soffice", "exit /b 3", "exit 3")
    fast = _script(
        tmp_path, "fast-soffice", f'"{sys.executable}" "{converter}" %*', f'exec "{sys.executable}" "{converter}" "$@"'
    )
    src = tmp_path / "a.pptx"
    src.write_bytes(b"x")
    assert convert(src, ".PNG", tmp_path / "out", candidates=[failing, fast]).name == "a.png"
    with pytest.raises(ConversionError) as info:
        convert(src, "pdf", tmp_path / "out2", candidates=[failing])
    assert "exit code 3" in info.value.attempts[0][1]


@pytest.mark.soffice
def test_real_conversion_with_short_private_profile(tmp_path, soffice_path, private_profile_root):
    src = _minimal_docx(tmp_path / "sample spec.docx")
    assert _profile_dirs(private_profile_root) == set()
    pdf = convert(src, "pdf", tmp_path / "out", timeout_s=120)
    assert pdf.name == "sample spec.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    with PdfPages(pdf) as pages:
        assert len(pages) == 1
        assert "Outdoor Lantern L-300" in pages.text(0)
    assert _profile_dirs(private_profile_root) == set()  # the per-call profile was removed


@pytest.mark.soffice
def test_real_soffice_timeout_raises_conversion_error_and_tries_next(
    tmp_path, soffice_path, monkeypatch, private_profile_root
):
    src = _minimal_docx(tmp_path / "timeout.docx")
    candidates = soffice_candidates()
    if len(candidates) < 2:
        candidates = [soffice_path, soffice_path]
    candidates = candidates[:2]
    calls = []
    real_popen = office.subprocess.Popen

    def recording_popen(command, *args, **kwargs):
        if "--convert-to" in command:  # ignore the process-tree kill helper
            calls.append(command[0])
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(office.subprocess, "Popen", recording_popen)
    assert _profile_dirs(private_profile_root) == set()
    with pytest.raises(ConversionError) as info:
        convert(src, "pdf", tmp_path / "out", timeout_s=0.05, candidates=candidates)
    assert not isinstance(info.value, office.subprocess.TimeoutExpired)
    assert calls == candidates  # the second binary was tried after the first timed out
    assert [binary for binary, _ in info.value.attempts] == candidates
    assert all("timed out" in reason for _, reason in info.value.attempts)
    assert "timed out" in str(info.value)
    time.sleep(0.5)
    assert _profile_dirs(private_profile_root) == set()


def test_soffice_marker_handling_is_registered(pytestconfig):
    markers = " ".join(pytestconfig.getini("markers"))
    assert "soffice" in markers and "network" in markers
    assert os.environ.get("VLMRAG_RUN_NETWORK_TESTS", "") in ("", "0", "1")
