#!/usr/bin/env python3
"""Generate the release SHA-256 manifest in deterministic path order."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", "work", "environment_capture"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT or not path.is_file():
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix in {".pyc", ".pyo", ".zip"}:
        return False
    return True


def main() -> int:
    records = []
    for path in sorted((value for value in ROOT.rglob("*") if included(value))):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(records) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records to {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

