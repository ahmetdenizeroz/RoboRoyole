#!/usr/bin/env python3
"""
trim_tracking_outputs.py

Keep only the first N frame rows in single-bee tracking coordinate TXT files.

Default: creates trimmed copies and does not overwrite originals.
Use --in-place to overwrite originals. Backups .bak are created by default.

Trims:
    *_coordinates_raw.txt
    *_coordinates_filtered.txt

Optionally updates:
    *_info.txt
    *_tracking_progress.txt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def seconds_to_hms(seconds: float) -> str:
    seconds_i = int(round(seconds))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def trim_coordinate_file(path: Path, frames: int, *, in_place: bool, backup: bool, suffix: str) -> Path:
    output_path = path if in_place else path.with_name(path.stem + suffix + path.suffix)

    if in_place and backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    kept_data_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fin, tmp_path.open("w", encoding="utf-8", newline="") as fout:
        header = fin.readline()
        if header:
            fout.write(header)

        for line in fin:
            if kept_data_rows >= frames:
                break
            fout.write(line)
            kept_data_rows += 1

    tmp_path.replace(output_path)
    return output_path


def update_key_value_file(path: Path, replacements: dict[str, str], *, in_place: bool, backup: bool, suffix: str) -> Path:
    output_path = path if in_place else path.with_name(path.stem + suffix + path.suffix)

    if in_place and backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fin, tmp_path.open("w", encoding="utf-8", newline="") as fout:
        for line in fin:
            stripped = line.rstrip("\r\n")
            newline = "\n"
            if line.endswith("\r\n"):
                newline = "\r\n"
            elif line.endswith("\r"):
                newline = "\r"

            if "\t" in stripped:
                key, _old = stripped.split("\t", 1)
                if key in replacements:
                    fout.write(f"{key}\t{replacements[key]}{newline}")
                    continue

            fout.write(line)

    tmp_path.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim tracking coordinate TXT files to the first N frame rows.")
    parser.add_argument("--folder", required=True, help="Tracking output folder containing coordinate TXT files")
    parser.add_argument("--frames", type=int, default=864000, help="Number of frame rows to keep, excluding header. Default: 864000")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used to update metadata. Default: 30.0")
    parser.add_argument("--in-place", action="store_true", help="Overwrite original files. Without this, trimmed copies are created.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when using --in-place.")
    parser.add_argument("--no-metadata-update", action="store_true", help="Only trim coordinate files; do not update info/progress TXT files.")
    parser.add_argument("--suffix", default="_first_864000", help="Suffix for output copies when not using --in-place.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    frames = int(args.frames)
    if frames <= 0:
        raise SystemExit("--frames must be positive")

    backup = not args.no_backup

    coordinate_files = sorted(folder.glob("*_coordinates_raw.txt")) + sorted(folder.glob("*_coordinates_filtered.txt"))
    if not coordinate_files:
        raise SystemExit("No coordinate files found. Expected *_coordinates_raw.txt and *_coordinates_filtered.txt")

    print(f"Folder: {folder}")
    print(f"Keeping first {frames} frame rows from each coordinate file.")
    print(f"Mode: {'overwrite originals' if args.in_place else 'create trimmed copies'}")

    for path in coordinate_files:
        before_lines = count_lines(path)
        before_data_rows = max(0, before_lines - 1)
        out = trim_coordinate_file(path, frames, in_place=args.in_place, backup=backup, suffix=args.suffix)
        after_lines = count_lines(out)
        after_data_rows = max(0, after_lines - 1)
        print(f"{path.name}: {before_data_rows} -> {after_data_rows} frame rows | output: {out.name}")

    if not args.no_metadata_update:
        end_time_s = frames / float(args.fps)
        end_time_str = seconds_to_hms(end_time_s)

        info_replacements = {
            "end_time_str": end_time_str,
            "end_time_s": f"{end_time_s:.6f}",
            "end_frame": str(frames),
            "planned_total_frames": str(frames),
            "tracking_status": "trimmed",
        }
        progress_replacements = {
            "status": "trimmed",
            "processed_frames": str(frames),
            "total_frames": str(frames),
            "percent": "100.0000",
            "last_absolute_frame": str(frames - 1),
        }

        for path in sorted(folder.glob("*_info.txt")):
            out = update_key_value_file(path, info_replacements, in_place=args.in_place, backup=backup, suffix=args.suffix)
            print(f"metadata updated: {out.name}")

        for path in sorted(folder.glob("*_tracking_progress.txt")):
            out = update_key_value_file(path, progress_replacements, in_place=args.in_place, backup=backup, suffix=args.suffix)
            print(f"progress updated: {out.name}")

    print("Done.")


if __name__ == "__main__":
    main()
