"""Headless LibreOffice conversion (DOCX/PPTX -> PDF, WMF/EMF -> PNG, ...).

Lessons baked into :func:`convert`:

* **One short, private profile per call.** LibreOffice refuses to run two
  instances on one user profile, and the default profile is shared with the
  desktop app. Each call therefore passes ``-env:UserInstallation`` pointing to
  a fresh ``<tmp>/vlmrag_lo_<8 hex>`` directory that is deleted afterwards
  (``VLMRAG_LO_PROFILE_ROOT`` replaces ``<tmp>``). The path is kept short on
  purpose: on Windows a long profile path makes LibreOffice crash at start-up
  (exit code ``0xC0000409``).
* **Timeouts fall through to the next binary.** A hung conversion raises
  ``subprocess.TimeoutExpired``; it is caught, the whole process tree is killed
  (LibreOffice spawns ``soffice.bin`` under a launcher, so killing the launcher
  alone leaves an orphan holding the profile), and the next candidate binary is
  tried. Only when every candidate failed is a :class:`ConversionError` raised,
  listing what was tried and why each attempt failed.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = ["find_soffice", "soffice_candidates", "ConversionError", "convert", "to_pdf"]

log = logging.getLogger(__name__)

_WINDOWS_DEFAULTS = (
    r"C:\Program Files\LibreOffice\program\soffice.com",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)
_MAC_DEFAULTS = ("/Applications/LibreOffice.app/Contents/MacOS/soffice",)
_POSIX_DEFAULTS = (
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/usr/lib/libreoffice/program/soffice",
    "/opt/libreoffice/program/soffice",
    "/snap/bin/libreoffice",
)


class ConversionError(RuntimeError):
    """LibreOffice could not produce the requested output.

    ``attempts`` lists ``(binary, reason)`` for every candidate that was tried.
    """

    def __init__(self, message: str, attempts: Optional[list[tuple[str, str]]] = None) -> None:
        super().__init__(message)
        self.attempts = list(attempts or [])


def soffice_candidates(preferred: Optional[str] = None) -> list[str]:
    """All usable LibreOffice binaries, best first, without duplicates.

    Order: ``preferred`` (e.g. ``Settings.soffice``), ``VLMRAG_SOFFICE``,
    ``soffice``/``libreoffice`` on ``PATH``, then the platform's default install
    locations. On Windows the console launcher ``soffice.com`` is preferred over
    ``soffice.exe`` because it waits for the conversion and forwards its output.
    """
    found: list[str] = []

    def add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        path = candidate if os.path.isfile(candidate) else shutil.which(candidate)
        if path:
            norm = os.path.normcase(os.path.abspath(path))
            if norm not in {os.path.normcase(os.path.abspath(p)) for p in found}:
                found.append(path)

    add(preferred)
    add(os.environ.get("VLMRAG_SOFFICE", "").strip() or None)
    if sys.platform == "win32":
        for name in ("soffice.com", "soffice.exe", "soffice"):
            add(shutil.which(name))
        for default in _WINDOWS_DEFAULTS:
            add(default)
    else:
        for name in ("soffice", "libreoffice"):
            add(shutil.which(name))
        for default in _MAC_DEFAULTS + _POSIX_DEFAULTS:
            add(default)
    return found


def find_soffice() -> Optional[str]:
    """Path of the preferred LibreOffice binary, or ``None`` when not installed."""
    candidates = soffice_candidates()
    return candidates[0] if candidates else None


def _profile_root() -> Path:
    override = os.environ.get("VLMRAG_LO_PROFILE_ROOT", "").strip()
    return Path(override) if override else Path(tempfile.gettempdir())


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and its children; best effort, never raises."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.communicate(timeout=10)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass


def _remove_profile(profile: Path) -> None:
    # A freshly killed soffice may hold files for a moment on Windows.
    for _ in range(5):
        shutil.rmtree(profile, ignore_errors=True)
        if not profile.exists():
            return
        time.sleep(0.2)
    log.debug("could not remove LibreOffice profile %s", profile)


def _find_output(src: Path, out_dir: Path, ext: str) -> Optional[Path]:
    expected = out_dir / f"{src.stem}.{ext}"
    if expected.is_file():
        return expected
    matches = sorted(out_dir.glob(f"{src.stem}*.{ext}"))
    return matches[0] if matches else None


def convert(
    src: Union[str, os.PathLike],
    target_ext: str,
    out_dir: Union[str, os.PathLike],
    *,
    timeout_s: float = 60,
    filter_name: Optional[str] = None,
    candidates: Optional[Sequence[str]] = None,
) -> Path:
    """Convert ``src`` with headless LibreOffice and return the output path.

    ``target_ext`` is the output extension (``"pdf"``, ``"png"``, ...);
    ``filter_name`` optionally selects an export filter
    (``--convert-to pdf:writer_pdf_Export``). ``candidates`` overrides the
    binaries to try (defaults to :func:`soffice_candidates`).
    """
    src_path = Path(src).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    ext = target_ext.lstrip(".").lower()
    if not ext:
        raise ValueError("target_ext must not be empty")
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    binaries = list(candidates) if candidates is not None else soffice_candidates()
    if not binaries:
        raise ConversionError(
            "LibreOffice was not found; install it or set VLMRAG_SOFFICE to the soffice binary"
        )
    convert_to = f"{ext}:{filter_name}" if filter_name else ext
    attempts: list[tuple[str, str]] = []

    for binary in binaries:
        profile = _profile_root() / f"vlmrag_lo_{secrets.token_hex(4)}"
        command = [
            binary,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--norestore",
            "--nologo",
            "--convert-to",
            convert_to,
            "--outdir",
            str(out_path),
            str(src_path),
        ]
        popen_kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True
        started = time.monotonic()
        log.info("converting %s -> %s via %s", src_path.name, ext, Path(binary).name)
        try:
            try:
                proc = subprocess.Popen(command, **popen_kwargs)
            except OSError as exc:
                attempts.append((binary, f"could not start: {exc}"))
                continue
            try:
                stdout, stderr = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                attempts.append((binary, f"timed out after {timeout_s:g}s"))
                log.warning("%s: conversion via %s timed out after %gs", src_path.name, Path(binary).name, timeout_s)
                continue
            elapsed = time.monotonic() - started
            output = _find_output(src_path, out_path, ext)
            if proc.returncode == 0 and output is not None:
                log.info("converted %s in %.1fs", src_path.name, elapsed)
                return output
            detail = (stderr or b"").decode("utf-8", "replace").strip() or (stdout or b"").decode(
                "utf-8", "replace"
            ).strip()
            reason = f"exit code {proc.returncode}" if proc.returncode != 0 else "no output file was produced"
            if detail:
                reason += f" ({detail[-300:]})"
            attempts.append((binary, reason))
            log.warning("%s: conversion via %s failed: %s", src_path.name, Path(binary).name, reason)
        finally:
            _remove_profile(profile)

    tried = "; ".join(f"{Path(b).name}: {why}" for b, why in attempts)
    raise ConversionError(f"LibreOffice could not convert {src_path.name} to {ext} ({tried})", attempts)


def to_pdf(src: Union[str, os.PathLike], out_dir: Union[str, os.PathLike], *, timeout_s: float = 60) -> Path:
    """Convenience wrapper: convert an office document to PDF."""
    return convert(src, "pdf", out_dir, timeout_s=timeout_s)
