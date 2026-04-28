"""Validate preprocessing stability across multiple DEAP subjects.

This script extends the s01 preprocessing sanity checks to s01-s03. It does not
rewrite the preprocessing pipeline; it reuses preprocess_subject and small
helpers from preprocess.py, then writes course-report friendly summaries.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from src.preprocess import (
    DEFAULT_ICA_COMPONENTS,
    DEFAULT_ICA_RANDOM_STATE,
    EXPECTED_DEAP_TRIAL_COUNT,
    TARGET_STIMULUS_SAMPLES,
    EEG_CHANNEL_COUNT,
    bandpass_and_notch_filter,
    baseline_correction,
    compute_ica_quantitative_summary,
    fix_trial_length,
    load_bdf_subject,
    preprocess_subject,
    run_ica_artifact_removal,
    _get_matplotlib_pyplot,
    _segments_to_numpy,
    extract_raw_eeg_trials_from_bdf,
)


SUBJECT_IDS = (1, 2, 3)
EXCLUDE_SETS = ((), (0,), (1,), (0, 1))
RESULTS_DIR = Path("results/preprocessing_multisubject")


def subject_label(subject_id: int) -> str:
    return f"s{subject_id:02d}"


def shape_to_text(shape) -> str:
    return "x".join(str(item) for item in shape)


def ensure_plot_env() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))


def save_signal_plot(
    before,
    after,
    output_path: Path,
    title: str,
    before_label: str,
    after_label: str,
    sampling_rate: float,
    max_seconds: float = 5.0,
) -> Path:
    """Save a simple first-trial/channel-0 before-after signal plot."""
    ensure_plot_env()
    plt = _get_matplotlib_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_samples = min(int(max_seconds * sampling_rate), before.shape[-1], after.shape[-1])
    time_axis = np.arange(max_samples) / sampling_rate

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time_axis, before[0, 0, :max_samples], label=before_label, linewidth=1.0)
    axis.plot(time_axis, after[0, 0, :max_samples], label=after_label, linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_single_signal_plot(data, output_path: Path, title: str, sampling_rate: float) -> Path:
    """Save one first-trial/channel-0 filtered stimulus waveform."""
    ensure_plot_env()
    plt = _get_matplotlib_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_samples = min(int(5 * sampling_rate), data.shape[-1])
    time_axis = np.arange(max_samples) / sampling_rate

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(time_axis, data[0, 0, :max_samples], linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_basic_visualizations(subject_id: int, preprocessed: dict) -> list[Path]:
    """Save basic preprocessing visualizations for one subject."""
    label = subject_label(subject_id)
    output_dir = RESULTS_DIR / label
    sampling_rate = preprocessed["sampling_rate"]

    return [
        save_signal_plot(
            preprocessed["raw_fixed_stimulus"],
            preprocessed["filtered_stimulus"],
            output_dir / f"{label}_raw_vs_filtered.png",
            f"{label} Raw vs Filtered Stimulus",
            "raw",
            "bandpass + notch",
            sampling_rate,
        ),
        save_signal_plot(
            preprocessed["ica_stimulus"],
            preprocessed["baseline_corrected_stimulus"],
            output_dir / f"{label}_baseline_correction_before_after.png",
            f"{label} Baseline Correction Before vs After",
            "before correction",
            "after correction",
            sampling_rate,
        ),
        save_single_signal_plot(
            preprocessed["filtered_stimulus"],
            output_dir / f"{label}_first_trial_ch0_filtered_stimulus.png",
            f"{label} First Trial Channel 0 Filtered Stimulus",
            sampling_rate,
        ),
    ]


def run_basic_smoke_test(subject_id: int) -> dict:
    """Run use_ica=False preprocessing and collect shape information."""
    result = {
        "subject_id": subject_label(subject_id),
        "success": False,
        "baseline_shape": "",
        "filtered_stimulus_shape": "",
        "baseline_corrected_stimulus_shape": "",
        "n_trials": "",
        "n_channels": "",
        "n_samples": "",
        "notes/error": "",
    }

    try:
        preprocessed = preprocess_subject(subject_id=subject_id, use_ica=False)
        final_shape = preprocessed["baseline_corrected_stimulus"].shape
        result.update(
            {
                "success": True,
                "baseline_shape": shape_to_text(preprocessed["raw_fixed_baseline"].shape),
                "filtered_stimulus_shape": shape_to_text(
                    preprocessed["filtered_stimulus"].shape
                ),
                "baseline_corrected_stimulus_shape": shape_to_text(final_shape),
                "n_trials": final_shape[0],
                "n_channels": final_shape[1],
                "n_samples": final_shape[2],
                "notes/error": "ok",
                "preprocessed": preprocessed,
            }
        )
        save_basic_visualizations(subject_id, preprocessed)
    except Exception as exc:
        result["notes/error"] = str(exc)

    return result


def prepare_filtered_subject_data(subject_id: int) -> dict:
    """Read once, fix trial length, and filter data before ICA experiments."""
    subject = load_bdf_subject(subject_id)
    extracted = extract_raw_eeg_trials_from_bdf(subject["bdf_path"])
    fixed = fix_trial_length(extracted)
    raw_baseline = _segments_to_numpy(fixed["trials"], "baseline")
    raw_stimulus = _segments_to_numpy(fixed["trials"], "stimulus")
    filtered = bandpass_and_notch_filter(
        raw_baseline,
        raw_stimulus,
        sampling_rate=fixed["sampling_rate"],
    )
    return {
        "sampling_rate": fixed["sampling_rate"],
        "filtered_baseline": filtered["filtered_baseline"],
        "filtered_stimulus": filtered["filtered_stimulus"],
    }


def interpret_ica_change(metrics: dict, exclude_setting: str) -> str:
    """Give a conservative text interpretation for one ICA setting."""
    hf_change = metrics["high_freq_ratio_change_pct"]
    std_change = metrics["std_change_pct"]
    var_change = metrics["var_change_pct"]

    if exclude_setting == "[]":
        return "reference ICA fit without component deletion"

    if hf_change < -5 and -15 <= std_change <= -1 and -30 <= var_change <= -2:
        return "possible candidate, but requires visual inspection"

    if hf_change >= 0:
        return "not convincing; high-frequency ratio did not decrease"

    if std_change < -20 or var_change < -40:
        return "risky; signal scale changed too much"

    return "weak evidence; do not default-delete"


def run_light_ica_validation(subject_id: int) -> list[dict]:
    """Run lightweight ICA metrics for one subject and save per-subject CSV."""
    label = subject_label(subject_id)
    output_dir = RESULTS_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        prepared = prepare_filtered_subject_data(subject_id)
        for exclude_components in EXCLUDE_SETS:
            ica_result = run_ica_artifact_removal(
                prepared["filtered_baseline"],
                prepared["filtered_stimulus"],
                enable_ica=True,
                n_components=DEFAULT_ICA_COMPONENTS,
                random_state=DEFAULT_ICA_RANDOM_STATE,
                exclude_components=exclude_components,
            )
            _ = baseline_correction(ica_result["baseline"], ica_result["stimulus"])
            summary = compute_ica_quantitative_summary(
                prepared["filtered_stimulus"],
                ica_result["stimulus"],
                prepared["sampling_rate"],
            )
            std_change_pct = (
                (summary["overall_std_after"] - summary["overall_std_before"])
                / summary["overall_std_before"]
                * 100
            )
            var_change_pct = (
                (summary["overall_var_after"] - summary["overall_var_before"])
                / summary["overall_var_before"]
                * 100
            )
            high_freq_ratio_change_pct = (
                (
                    summary["high_freq_ratio_30_45_after"]
                    - summary["high_freq_ratio_30_45_before"]
                )
                / summary["high_freq_ratio_30_45_before"]
                * 100
            )
            exclude_setting = str(list(exclude_components))
            rows.append(
                {
                    "subject_id": label,
                    "exclude_setting": exclude_setting,
                    "removed_components": str(ica_result["removed_components"]),
                    "overall_std_before": summary["overall_std_before"],
                    "overall_std_after": summary["overall_std_after"],
                    "overall_var_before": summary["overall_var_before"],
                    "overall_var_after": summary["overall_var_after"],
                    "high_freq_ratio_before": summary["high_freq_ratio_30_45_before"],
                    "high_freq_ratio_after": summary["high_freq_ratio_30_45_after"],
                    "std_change_pct": std_change_pct,
                    "var_change_pct": var_change_pct,
                    "high_freq_ratio_change_pct": high_freq_ratio_change_pct,
                    "suggested_interpretation": interpret_ica_change(
                        {
                            "std_change_pct": std_change_pct,
                            "var_change_pct": var_change_pct,
                            "high_freq_ratio_change_pct": high_freq_ratio_change_pct,
                        },
                        exclude_setting,
                    ),
                }
            )
    except Exception as exc:
        rows.append(
            {
                "subject_id": label,
                "exclude_setting": "error",
                "removed_components": "",
                "overall_std_before": "",
                "overall_std_after": "",
                "overall_var_before": "",
                "overall_var_after": "",
                "high_freq_ratio_before": "",
                "high_freq_ratio_after": "",
                "std_change_pct": "",
                "var_change_pct": "",
                "high_freq_ratio_change_pct": "",
                "suggested_interpretation": f"ICA validation failed: {exc}",
            }
        )

    save_csv(output_dir / "ica_light_summary.csv", rows)
    return rows


def save_csv(path: Path, rows: list[dict]) -> Path:
    """Save list-of-dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def generate_report(basic_rows: list[dict], ica_rows: list[dict]) -> str:
    """Build a conservative multi-subject validation report."""
    successful = [row for row in basic_rows if row["success"]]
    all_success = len(successful) == len(basic_rows)
    expected_shape = (
        EXPECTED_DEAP_TRIAL_COUNT,
        EEG_CHANNEL_COUNT,
        TARGET_STIMULUS_SAMPLES,
    )
    shape_consistent = all(
        (
            row["n_trials"],
            row["n_channels"],
            row["n_samples"],
        )
        == expected_shape
        for row in successful
    )
    abnormal_subjects = [
        row["subject_id"]
        for row in basic_rows
        if not row["success"]
        or (
            row["n_trials"],
            row["n_channels"],
            row["n_samples"],
        )
        != expected_shape
    ]
    deletion_rows = [
        row for row in ica_rows if row["exclude_setting"] not in {"[]", "error"}
    ]
    stable_hf_improvement = [
        row for row in deletion_rows
        if isinstance(row["high_freq_ratio_change_pct"], float)
        and row["high_freq_ratio_change_pct"] < -5
    ]

    lines = [
        "Multi-subject preprocessing validation report",
        "",
        f"Subjects checked: {', '.join(row['subject_id'] for row in basic_rows)}",
        "",
        "1. Stable preprocessing run:",
        f"s01/s02/s03 all succeeded: {all_success}",
        "",
        "2. Trial/channel/sample consistency:",
        f"expected final stimulus shape: {expected_shape}",
        f"all successful subjects match expected shape: {shape_consistent}",
        "",
        "3. Abnormal subjects:",
        ", ".join(abnormal_subjects) if abnormal_subjects else "none detected",
        "",
        "4. ICA evidence across subjects:",
        "Current evidence is insufficient to default-delete ICA components.",
        f"component-deletion rows with >5% high-frequency ratio decrease: {len(stable_hf_improvement)}",
        "A decrease in std/var alone is not treated as sufficient evidence.",
        "",
        "5. Current recommendation:",
        "Keep use_ica=True available for debugging, but keep "
        "ica_exclude_components=[] by default.",
        "",
        "6. Next checks:",
        "Inspect waveform/PSD/component diagnostics per subject before choosing "
        "manual component removal. Multi-component deletion should be treated as "
        "higher risk unless visual evidence is strong.",
    ]

    return "\n".join(lines)


def main() -> None:
    """Run s01-s03 preprocessing and lightweight ICA validation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    basic_rows = []
    all_ica_rows = []

    for subject_id in SUBJECT_IDS:
        print(f"validating subject {subject_label(subject_id)}...")
        basic_rows.append(run_basic_smoke_test(subject_id))
        all_ica_rows.extend(run_light_ica_validation(subject_id))

    basic_summary_path = save_csv(
        RESULTS_DIR / "multisubject_preprocessing_summary.csv",
        [
            {key: value for key, value in row.items() if key != "preprocessed"}
            for row in basic_rows
        ],
    )
    ica_summary_path = save_csv(
        RESULTS_DIR / "multisubject_ica_light_summary.csv",
        all_ica_rows,
    )
    report_path = RESULTS_DIR / "multisubject_validation_report.txt"
    report_path.write_text(generate_report(basic_rows, all_ica_rows), encoding="utf-8")

    print(f"saved summary to {basic_summary_path}")
    print(f"saved ICA summary to {ica_summary_path}")
    print(f"saved report to {report_path}")


if __name__ == "__main__":
    main()
