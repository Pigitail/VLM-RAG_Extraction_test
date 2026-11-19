#!/usr/bin/env python3
"""Pre-publication check for this repository.

Scans every file that would be published for:
  * credentials (API keys, tokens, private keys, passwords, URLs with credentials)
  * private IP addresses, cloud endpoints and e-mail addresses
  * entries of an optional denylist kept OUTSIDE the repository
  * document/binary formats that must not be committed, and image files whose
    header does not match their extension (e.g. encrypted by an endpoint agent)
  * text files that are not valid UTF-8

Usage:
    python scripts/release_check.py [--root .] [--denylist PATH]

Denylist format (one entry per line, '#' starts a comment):
    re:<regular expression>     matched case-insensitively
    lit:<literal text>          matched case-insensitively as a substring

Findings are reported as category, path, line and denylist entry number only;
matched text is never printed. Add "release-check: ignore" to a line to
suppress a known false positive on that line.

Exit status: 0 when clean, 1 when there are findings, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", "build", "dist", "node_modules", "work", "out",
}
FORBIDDEN_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".docm", ".dot", ".dotx", ".ppt", ".pptx", ".pptm",
    ".pot", ".potx", ".xls", ".xlsx", ".xlsm", ".msg", ".eml", ".hwp", ".hwpx",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".exe", ".dll", ".pyc", ".pyd", ".so",
    ".db", ".sqlite", ".pkl", ".pickle", ".npy", ".npz", ".pt", ".pth", ".bin",
    ".onnx", ".safetensors", ".pem", ".key", ".pfx", ".p12",
}
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", "credentials.json"}
IMAGE_MAGIC = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".webp": (b"RIFF",),
    ".ico": (b"\x00\x00\x01\x00",),
}
MAX_FILE_BYTES = 3 * 1024 * 1024
IGNORE_MARK = "release-check: ignore"
PLACEHOLDER_HINTS = (
    "example", "dummy", "placeholder", "your-", "your_", "<", "${", "xxx",
    "changeme", "test", "fake", "sample", "redacted", "****",
)

SECRET_PATTERNS = [
    ("openai-key", r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    ("google-api-key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    ("slack-token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    ("vendor-key", r"\bup_[A-Za-z0-9]{20,}\b"),
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("hex32-key", r"(?<![0-9A-Fa-f])[0-9a-fA-F]{32}(?![0-9A-Fa-f])"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    (
        "credential-assignment",
        r"\b(?:api[_-]?key|secret|passw(?:or)?d|pwd|token|access[_-]?key)\b[\"']?\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']",
    ),
    ("url-with-credentials", r"\b[a-z][a-z0-9+.\-]*://[^/\s:@\"']+:[^/\s@\"']+@"),
    ("cloud-endpoint", r"\b[a-z0-9][a-z0-9\-]*\.(?:openai|cognitiveservices)\.azure\.com\b"),
    (
        "private-ip",
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b",
    ),
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
]
EMAIL_ALLOW = ("@example.com", "@example.org", "@users.noreply.github.com", "noreply@")
COMPILED = [(name, re.compile(rx, re.IGNORECASE if name != "hex32-key" else 0)) for name, rx in SECRET_PATTERNS]


def load_denylist(path: Path) -> list[tuple[int, str, object]]:
    entries = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            entries.append((number, "re", re.compile(line[3:], re.IGNORECASE)))
        elif line.startswith("lit:"):
            entries.append((number, "lit", line[4:].lower()))
        else:
            raise ValueError(f"denylist line {number}: expected 're:' or 'lit:' prefix")
    return entries


def list_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                check=True, capture_output=True,
            ).stdout
            return sorted(root / p for p in out.decode("utf-8").split("\0") if p)
        except (OSError, subprocess.CalledProcessError):
            pass
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        files.extend(Path(dirpath) / name for name in filenames)
    return sorted(files)


def is_placeholder(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in PLACEHOLDER_HINTS)


def deny_hits(text: str, denylist) -> list[int]:
    low = text.lower()
    hits = []
    for number, kind, matcher in denylist:
        if (kind == "lit" and matcher in low) or (kind == "re" and matcher.search(text)):
            hits.append(number)
    return hits


def check(root: Path, denylist) -> list[str]:
    findings: list[str] = []
    files = list_files(root)
    for path in files:
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        for number in deny_hits(rel, denylist):
            findings.append(f"[denylist#{number}] {rel} (file path)")
        if path.name.lower() in FORBIDDEN_NAMES:
            findings.append(f"[forbidden-file] {rel}")
            continue
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"[forbidden-format] {rel}")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            findings.append(f"[unreadable] {rel} ({type(exc).__name__})")
            continue
        if len(data) > MAX_FILE_BYTES:
            findings.append(f"[too-large] {rel} ({len(data)} bytes)")
        if suffix in IMAGE_MAGIC:
            if not data.startswith(IMAGE_MAGIC[suffix]):
                findings.append(f"[bad-image-header] {rel} (encrypted or mislabeled?)")
            continue
        if b"\x00" in data:
            findings.append(f"[binary-file] {rel}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"[not-utf8] {rel} (encrypted or wrong encoding?)")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if IGNORE_MARK in line:
                continue
            for name, rx in COMPILED:
                for match in rx.finditer(line):
                    value = match.group(0)
                    if name == "email" and any(allow in value.lower() for allow in EMAIL_ALLOW):
                        continue
                    if name in ("credential-assignment", "cloud-endpoint", "email") and is_placeholder(value):
                        continue
                    findings.append(f"[{name}] {rel}:{lineno}")
                    break
            for number in deny_hits(line, denylist):
                findings.append(f"[denylist#{number}] {rel}:{lineno}")
    return findings, len(files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--denylist", type=Path, default=os.environ.get("RELEASE_CHECK_DENYLIST") or None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    denylist = []
    if args.denylist:
        denylist_path = Path(args.denylist).resolve()
        if denylist_path.is_relative_to(root):
            print("error: the denylist must live outside the repository", file=sys.stderr)
            return 2
        denylist = load_denylist(denylist_path)
    findings, scanned = check(root, denylist)
    for finding in findings:
        print(finding)
    print(f"scanned {scanned} files, {len(denylist)} denylist entries, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
