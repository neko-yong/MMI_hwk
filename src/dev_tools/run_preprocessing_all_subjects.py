"""Run stable preprocessing smoke validation for all DEAP original BDF files.

This script does not generate figures and does not change the preprocessing
logic. It simply checks whether preprocess_subject can process every sXX.bdf
file under data/DEAP/original/ with the stable default flow: use_ica=False.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from src.preprocess import (
    DEFAULT_ORIGINAL_DIR,
    EEG_CHANNEL_COUNT,
    EXPECTED_DEAP_TRIAL_COUNT,
    PROJECT_ROOT,
    TARGET_STIMULUS_SAMPLES,
    preprocess_subject,
)


RESULTS_DIR = Path("results/preprocessing_all_subjects")
SUMMARY_CSV_PATH = RESULTS_DIR / "all_subjects_preprocessing_summary.csv"
REPORT_PATH = RESULTS_DIR / "all_subjects_preprocessing_report.txt"
EXPECTED_FINAL_SHAPE = (
    EXPECTED_DEAP_TRIAL_COUNT,
    EEG_CHANNEL_COUNT,
    TARGET_STIMULUS_SAMPLES,
)


def discover_bdf_files(original_dir: Path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR) -> list[Path]:
    """Find all DEAP subject BDF files named sXX.bdf."""
    if not original_dir.exists():
        raise FileNotFoundError(f"DEAP original directory not found: {original_dir}")

    return sorted(
        path
        for path in original_dir.glob("s*.bdf")
        if re.fullmatch(r"s\d{2}\.bdf", path.name)
    )


def subject_id_from_path(path: Path) -> int:
    """Extract integer subject id from sXX.bdf."""
    return int(path.stem[1:])


def shape_to_text(shape) -> str:
    return "x".join(str(item) for item in shape)


def process_subject_file(path: Path) -> dict:
    """Run preprocess_subject for one file and return a summary row."""
    subject_id = subject_id_from_path(path)
    row = {
        "subject_id": f"s{subject_id:02d}",
        "file_path": str(path),
        "status": "failed",
        "error_message": "",
        "baseline_shape": "",
        "filtered_stimulus_shape": "",
        "baseline_corrected_stimulus_shape": "",
        "n_trials": "",
        "n_channels": "",
        "n_samples": "",
    }

    try:
        result = preprocess_subject(subject_id=subject_id, use_ica=False)
        final_shape = result["baseline_corrected_stimulus"].shape
        row.update(
            {
                "status": "success",
                "baseline_shape": shape_to_text(result["raw_fixed_baseline"].shape),
                "filtered_stimulus_shape": shape_to_text(
                    result["filtered_stimulus"].shape
                ),
                "baseline_corrected_stimulus_shape": shape_to_text(final_shape),
                "n_trials": final_shape[0],
                "n_channels": final_shape[1],
                "n_samples": final_shape[2],
            }
        )
        print(f"success: shape {final_shape}")
    except Exception as exc:
        row["error_message"] = str(exc)
        print(f"failed: {exc}")

    return row


def save_summary_csv(rows: list[dict], output_path: Path = SUMMARY_CSV_PATH) -> Path:
    """Save per-subject preprocessing status to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "file_path",
        "status",
        "error_message",
        "baseline_shape",
        "filtered_stimulus_shape",
        "baseline_corrected_stimulus_shape",
        "n_trials",
        "n_channels",
        "n_samples",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_report(rows: list[dict]) -> str:
    """Build a compact all-subject preprocessing report."""
    success_rows = [row for row in rows if row["status"] == "success"]
    failed_rows = [row for row in rows if row["status"] != "success"]
    abnormal_rows = [
        row
        for row in success_rows
        if (
            row["n_trials"],
            row["n_channels"],
            row["n_samples"],
        )
        != EXPECTED_FINAL_SHAPE
    ]
    abnormal_subjects = [row["subject_id"] for row in abnormal_rows + failed_rows]

    return "\n".join(
        [
            "All-subject DEAP preprocessing validation report",
            "",
            f"Total subjects: {len(rows)}",
            f"Success count: {len(success_rows)}",
            f"Failure count: {len(failed_rows)}",
            f"Expected final shape: {EXPECTED_FINAL_SHAPE}",
            "All successful subjects match expected shape: "
            f"{len(abnormal_rows) == 0}",
            "Abnormal subjects: "
            + (", ".join(abnormal_subjects) if abnormal_subjects else "none"),
            "",
            "Stable flow used:",
            "preprocess_subject(subject_id=..., use_ica=False)",
            "ICA is available for debugging, but no ICA components are deleted "
            "in this all-subject stability validation.",
        ]
    )


def save_report(rows: list[dict], output_path: Path = REPORT_PATH) -> Path:
    """Save all-subject report to text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(rows), encoding="utf-8")
    return output_path


def main() -> None:
    """Run preprocessing validation for every original DEAP BDF subject."""
    bdf_files = discover_bdf_files()
    rows = []

    for path in bdf_files:
        subject_id = subject_id_from_path(path)
        print(f"processing s{subject_id:02d}...")
        rows.append(process_subject_file(path))

    summary_path = save_summary_csv(rows)
    report_path = save_report(rows)
    print(f"saved summary to {summary_path}")
    print(f"saved report to {report_path}")


if __name__ == "__main__":
    main()
