"""Rename unreadable WSI files after validating with OpenSlide and tiffslide.

The script is dry-run by default. Add ``--apply`` to actually rename files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import openslide
except ImportError:  # pragma: no cover - depends on environment
    openslide = None

try:
    import tiffslide
except ImportError:  # pragma: no cover - depends on environment
    tiffslide = None


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class CheckResult:
    path: Path
    readable: bool
    backend: str
    error: str
    new_path: Path | None = None
    renamed: bool = False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename WSI files that cannot be opened by OpenSlide or tiffslide."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing WSI .tif/.tiff files.")
    parser.add_argument(
        "--batch-summary",
        help="Optional batch_summary.csv. If provided, only rows with status Failed are checked.",
    )
    parser.add_argument("--suffix", default="_CORRUPT", help="Suffix inserted before the file extension.")
    parser.add_argument(
        "--report",
        default="rename_unreadable_wsi_report.csv",
        help="CSV report path. Relative paths are written under input-dir.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename unreadable files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    candidates = discover_candidates(input_dir, Path(args.batch_summary).expanduser() if args.batch_summary else None)
    if not candidates:
        raise SystemExit(f"No candidate .tif/.tiff files found in {input_dir}")

    results: list[CheckResult] = []
    for path in candidates:
        result = check_slide(path)
        if not result.readable:
            result.new_path = unique_renamed_path(path, args.suffix)
            if args.apply:
                path.rename(result.new_path)
                result.renamed = True
        results.append(result)

    report_path = Path(args.report).expanduser()
    if not report_path.is_absolute():
        report_path = input_dir / report_path
    write_report(results, report_path)
    print_summary(results, report_path, applied=args.apply)
    return 0


def discover_candidates(input_dir: Path, batch_summary: Path | None) -> list[Path]:
    if batch_summary is None:
        return sorted(
            [
                path
                for path in input_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not path.name.startswith(".")
                and not path.name.startswith("._")
            ],
            key=lambda path: path.name.lower(),
        )

    if not batch_summary.exists():
        raise SystemExit(f"Batch summary does not exist: {batch_summary}")

    names: list[str] = []
    with batch_summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status", "")).lower()
            if status == "failed":
                names.append(str(row.get("WSI filename", "")).strip())

    paths = [input_dir / name for name in names if name]
    return sorted([path for path in paths if path.exists()], key=lambda path: path.name.lower())


def check_slide(path: Path) -> CheckResult:
    errors: list[str] = []

    if openslide is not None:
        try:
            slide = openslide.OpenSlide(str(path))
            slide.close()
            return CheckResult(path=path, readable=True, backend="openslide", error="")
        except Exception as exc:  # noqa: BLE001 - diagnostic utility
            errors.append(f"openslide: {exc!r}")
    else:
        errors.append("openslide: not installed")

    if tiffslide is not None:
        try:
            slide = tiffslide.TiffSlide(str(path))
            slide.close()
            return CheckResult(path=path, readable=True, backend="tiffslide", error="")
        except Exception as exc:  # noqa: BLE001 - diagnostic utility
            errors.append(f"tiffslide: {exc!r}")
    else:
        errors.append("tiffslide: not installed")

    return CheckResult(path=path, readable=False, backend="", error="; ".join(errors))


def unique_renamed_path(path: Path, suffix: str) -> Path:
    suffix = suffix.strip() or "_CORRUPT"
    if path.stem.endswith(suffix):
        return path

    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}{suffix}_{counter}{path.suffix}")
        counter += 1
    return candidate


def write_report(results: Iterable[CheckResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "readable",
                "backend",
                "renamed",
                "new_filename",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "filename": result.path.name,
                    "readable": result.readable,
                    "backend": result.backend,
                    "renamed": result.renamed,
                    "new_filename": result.new_path.name if result.new_path else "",
                    "error": result.error,
                }
            )


def print_summary(results: list[CheckResult], report_path: Path, applied: bool) -> None:
    readable = sum(1 for result in results if result.readable)
    unreadable = len(results) - readable
    renamed = sum(1 for result in results if result.renamed)
    mode = "APPLY" if applied else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Checked: {len(results)}")
    print(f"Readable: {readable}")
    print(f"Unreadable: {unreadable}")
    print(f"Renamed: {renamed}")
    print(f"Report: {report_path}")
    if not applied and unreadable:
        print("Dry run only. Re-run with --apply to rename unreadable files.")


if __name__ == "__main__":
    raise SystemExit(main())
