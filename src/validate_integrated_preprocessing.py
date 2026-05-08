"""Validate integrated stable + official-like preprocessing outputs.

This script keeps preprocess_subject() as the single entry point and checks
that the backward-compatible output plus optional official-like fields work
for all DEAP subjects.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.preprocess import (
    EEG_CHANNEL_COUNT,
    EXPECTED_DEAP_TRIAL_COUNT,
    PROJECT_ROOT,
    preprocess_subject,
)


RESULTS_DIR = PROJECT_ROOT / "results/integrated_preprocessing_validation"
SUMMARY_PATH = RESULTS_DIR / "integrated_preprocessing_summary.csv"
REPORT_PATH = RESULTS_DIR / "integrated_preprocessing_report.txt"

EXPECTED_BASELINE_CORRECTED_SHAPE = (40, 32, 30720)
EXPECTED_OFFICIAL_LIKE_EEG_SHAPE = (40, 32, 7680)
EXPECTED_OFFICIAL_LIKE_DATA_SHAPE = (40, 40, 8064)
EXPECTED_LABELS_SHAPE = (40, 4)


def shape_text(value) -> str:
    """Format an array shape or None for CSV/report output."""
    if value is None:
        return "None"
    return str(tuple(value.shape))


def validate_subject(subject_id: int) -> dict:
    """Run one subject and return a validation row."""
    subject_label = f"s{subject_id:02d}"
    print(f"validating subject {subject_label}...")
    row = {
        "subject_id": subject_label,
        "success": False,
        "baseline_corrected_stimulus_shape": "",
        "official_like_baseline_corrected_eeg_shape": "",
        "official_like_labels_shape": "",
        "official_like_data_shape": "",
        "official_like_data_available": False,
        "n_trials": "",
        "n_channels": "",
        "n_samples": "",
        "notes": "",
    }

    try:
        result = preprocess_subject(
            subject_id=subject_id,
            output_official_like=True,
        )
        baseline_corrected = result["baseline_corrected_stimulus"]
        official_like_eeg = result["official_like_baseline_corrected_eeg"]
        labels = result["official_like_labels"]
        official_like_data = result["official_like_data"]
        info = result["preprocessing_info"]

        checks = [
            (
                "baseline_corrected_stimulus",
                baseline_corrected.shape,
                EXPECTED_BASELINE_CORRECTED_SHAPE,
            ),
            (
                "official_like_baseline_corrected_eeg",
                official_like_eeg.shape,
                EXPECTED_OFFICIAL_LIKE_EEG_SHAPE,
            ),
            ("official_like_labels", labels.shape, EXPECTED_LABELS_SHAPE),
        ]
        for name, actual, expected in checks:
            if actual != expected:
                raise ValueError(f"{name} shape {actual} != expected {expected}")

        if official_like_data is not None:
            if official_like_data.shape != EXPECTED_OFFICIAL_LIKE_DATA_SHAPE:
                raise ValueError(
                    "official_like_data shape "
                    f"{official_like_data.shape} != expected "
                    f"{EXPECTED_OFFICIAL_LIKE_DATA_SHAPE}"
                )
            official_like_data_available = True
        else:
            official_like_data_available = False

        row.update(
            {
                "success": True,
                "baseline_corrected_stimulus_shape": shape_text(baseline_corrected),
                "official_like_baseline_corrected_eeg_shape": shape_text(official_like_eeg),
                "official_like_labels_shape": shape_text(labels),
                "official_like_data_shape": shape_text(official_like_data),
                "official_like_data_available": official_like_data_available,
                "n_trials": baseline_corrected.shape[0],
                "n_channels": baseline_corrected.shape[1],
                "n_samples": baseline_corrected.shape[2],
                "notes": info.get("official_like_data_reason", ""),
            }
        )
        print(
            f"success: baseline_corrected={baseline_corrected.shape}, "
            f"official_like_eeg={official_like_eeg.shape}, "
            f"labels={labels.shape}, official_like_data={shape_text(official_like_data)}"
        )
    except Exception as exc:
        row["notes"] = str(exc)
        print(f"failed: {subject_label}: {exc}")

    return row


def write_summary(rows: list[dict], path: Path) -> None:
    """Write validation rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict]) -> str:
    """Build a compact text report for course handoff."""
    success_rows = [row for row in rows if row["success"]]
    failed_rows = [row for row in rows if not row["success"]]
    official_data_rows = [
        row for row in success_rows if row["official_like_data_available"]
    ]
    expected_baseline_ok = all(
        row["baseline_corrected_stimulus_shape"]
        == str(EXPECTED_BASELINE_CORRECTED_SHAPE)
        for row in success_rows
    )
    expected_official_eeg_ok = all(
        row["official_like_baseline_corrected_eeg_shape"]
        == str(EXPECTED_OFFICIAL_LIKE_EEG_SHAPE)
        for row in success_rows
    )
    expected_labels_ok = all(
        row["official_like_labels_shape"] == str(EXPECTED_LABELS_SHAPE)
        for row in success_rows
    )
    official_data_ok = all(
        row["official_like_data_shape"] == str(EXPECTED_OFFICIAL_LIKE_DATA_SHAPE)
        for row in official_data_rows
    )

    lines = [
        "Integrated preprocessing validation report",
        "",
        "Purpose:",
        "Validate preprocess_subject() as the unified entry point for both the stable 512 Hz preprocessing output and optional official .dat-style outputs.",
        "",
        "Expected shapes:",
        f"baseline_corrected_stimulus: {EXPECTED_BASELINE_CORRECTED_SHAPE}",
        f"official_like_baseline_corrected_eeg: {EXPECTED_OFFICIAL_LIKE_EEG_SHAPE}",
        f"official_like_labels: {EXPECTED_LABELS_SHAPE}",
        f"official_like_data, if available: {EXPECTED_OFFICIAL_LIKE_DATA_SHAPE}",
        "",
        "Summary:",
        f"total subjects: {len(rows)}",
        f"success: {len(success_rows)}",
        f"failed: {len(failed_rows)}",
        f"official_like_data available: {len(official_data_rows)}",
        f"stable baseline shape all OK: {expected_baseline_ok}",
        f"official-like EEG shape all OK: {expected_official_eeg_ok}",
        f"official-like label shape all OK: {expected_labels_ok}",
        f"available official_like_data shape all OK: {official_data_ok}",
        "",
        "Conservative ICA policy:",
        "Default preprocess_subject() behavior remains unchanged; no ICA component is automatically deleted unless explicitly requested.",
        "",
        "Notes:",
        "official_like_data is project-generated official-style data, not an exact point-by-point copy of DEAP official .dat.",
    ]

    if failed_rows:
        lines.extend(["", "Failed subjects:"])
        lines.extend(
            f"{row['subject_id']}: {row['notes']}"
            for row in failed_rows
        )

    unavailable = [
        row for row in success_rows if not row["official_like_data_available"]
    ]
    if unavailable:
        lines.extend(["", "Subjects without official_like_data:"])
        lines.extend(
            f"{row['subject_id']}: {row['notes']}"
            for row in unavailable
        )

    return "\n".join(lines)


def main() -> None:
    """Validate s01-s32 and save summary/report files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [validate_subject(subject_id) for subject_id in range(1, 33)]
    write_summary(rows, SUMMARY_PATH)
    REPORT_PATH.write_text(build_report(rows), encoding="utf-8")
    print(f"saved summary to {SUMMARY_PATH}")
    print(f"saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
