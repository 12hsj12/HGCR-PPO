"""Shared helpers for experiment output paths and CSV aggregation."""

from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


def make_run_id(run_id: str | None = None) -> str:
    return sanitize_token(run_id) if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return safe.strip("_") or "untagged"


def make_result_path(
    result_dir: str | Path,
    prefix: str,
    tokens: Sequence[object],
    run_id: str | None,
    extension: str = ".csv",
    overwrite: bool = False,
) -> Path:
    path_tokens = [*tokens]
    if run_id:
        path_tokens.append(run_id)
    suffix = "_".join(sanitize_token(token) for token in path_tokens)
    path = Path(result_dir) / f"{prefix}_{suffix}{extension}"
    return path if overwrite else unique_path(path)


def unique_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def make_run_dir(base_dir: str | Path, tokens: Sequence[object], run_id: str, overwrite: bool = False) -> Path:
    dirname = "_".join(sanitize_token(token) for token in [*tokens, run_id])
    path = Path(base_dir) / dirname
    return path if overwrite else unique_path(path)


def save_csv_no_overwrite(
    rows: Iterable[dict],
    path: str | Path,
    fieldnames: Sequence[str],
    overwrite: bool = False,
) -> Path:
    out_path = Path(path) if overwrite else unique_path(path)
    write_csv(rows, out_path, fieldnames)
    return out_path


def write_csv(rows: Iterable[dict], path: str | Path, fieldnames: Sequence[str]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def update_latest_file(rows: Iterable[dict], path: str | Path, fieldnames: Sequence[str]) -> Path:
    return write_csv(rows, path, fieldnames)


def update_latest_dir(run_dir: str | Path, latest_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    latest_dir = Path(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for file_path in run_dir.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, latest_dir / file_path.name)
    return latest_dir


def rebuild_all_summary(
    result_dir: str | Path,
    pattern: str,
    output_path: str | Path,
    fieldnames: Sequence[str],
    exclude_names: set[str] | None = None,
    exclude_prefixes: Sequence[str] = (),
    required_substrings: Sequence[str] = (),
) -> Path:
    result_dir = Path(result_dir)
    expected_fields = list(fieldnames)
    exclude_names = exclude_names or set()
    combined_rows: list[dict] = []

    for path in sorted(result_dir.glob(pattern)):
        if path.name in exclude_names:
            continue
        if any(path.name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if any(substring not in path.name for substring in required_substrings):
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != expected_fields:
                print(f"Warning: skipped {path} because CSV fields do not match the current schema.")
                continue
            combined_rows.extend(reader)

    return write_csv(combined_rows, output_path, expected_fields)
