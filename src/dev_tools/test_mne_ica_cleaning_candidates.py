"""Test subject-specific ICA cleaning candidates from MNE review.

This script compares the stable default preprocessing output against a
candidate ICA-cleaned output for selected subjects. It is diagnostic only:
preprocess_subject defaults are not modified, and component IDs are not treated
as global deletion rules.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from scipy.signal import welch

from src.preprocess import (
    DEFAULT_ICA_COMPONENTS,
    DEFAULT_ICA_RANDOM_STATE,
    DEAP_SAMPLING_RATE,
    EEG_CHANNEL_COUNT,
    preprocess_subject,
)
from src.dev_tools.review_ica_components import (
    find_auxiliary_channels,
    flatten_trials,
    load_auxiliary_subset,
    read_bdf_header,
    zscore_columns,
)
from src.preprocess import DEFAULT_ORIGINAL_DIR, PROJECT_ROOT


CANDIDATE_EXCLUDES = {
    10: [0],
    24: [0, 7],
}
RESULTS_DIR = Path("results/mne_ica_cleaning_test")
SUMMARY_CSV = RESULTS_DIR / "mne_ica_cleaning_test_summary.csv"
CHANNEL_STD_CSV = RESULTS_DIR / "mne_ica_cleaning_test_channel_std.csv"
REPORT_PATH = RESULTS_DIR / "mne_ica_cleaning_test_report.txt"


def configure_plot_cache() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))


def get_pyplot():
    configure_plot_cache()
    import matplotlib.pyplot as plt

    return plt


def subject_label(subject_id: int) -> str:
    return f"s{subject_id:02d}"


def high_freq_ratio(data: np.ndarray) -> float:
    """Return 30-45 Hz power ratio within 4-45 Hz for trial EEG data."""
    flattened = data.reshape(-1)
    freqs, psd = welch(
        flattened,
        fs=DEAP_SAMPLING_RATE,
        nperseg=min(4096, flattened.shape[0]),
    )
    high_mask = (freqs >= 30) & (freqs <= 45)
    total_mask = (freqs >= 4) & (freqs <= 45)
    total_power = float(psd[total_mask].sum())
    return float(psd[high_mask].sum()) / total_power if total_power > 0 else 0.0


def overall_stats(data: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "high_freq_ratio_30_45": high_freq_ratio(data),
    }


def pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def run_default_preprocessing(subject_id: int) -> dict:
    return preprocess_subject(subject_id=subject_id, use_ica=False)


def run_candidate_preprocessing(subject_id: int, exclude_components: list[int]) -> dict:
    return preprocess_subject(
        subject_id=subject_id,
        use_ica=True,
        ica_n_components=DEFAULT_ICA_COMPONENTS,
        ica_random_state=DEFAULT_ICA_RANDOM_STATE,
        ica_exclude_components=exclude_components,
    )


def plot_waveform(subject: str, before: np.ndarray, after: np.ndarray, output_dir: Path) -> Path:
    plt = get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = min(5 * DEAP_SAMPLING_RATE, before.shape[-1], after.shape[-1])
    time_axis = np.arange(max_samples) / DEAP_SAMPLING_RATE
    path = output_dir / f"{subject}_waveform_before_after_cleaning.png"

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(time_axis, before[0, 0, :max_samples], label="default/no ICA", linewidth=1.0)
    axis.plot(time_axis, after[0, 0, :max_samples], label="candidate ICA cleaned", linewidth=1.0)
    axis.set_title(f"{subject} trial 0 channel 0 waveform before/after candidate ICA cleaning")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_psd(subject: str, before: np.ndarray, after: np.ndarray, output_dir: Path) -> Path:
    plt = get_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject}_psd_before_after_cleaning.png"
    freqs_before, psd_before = welch(before[0, 0], fs=DEAP_SAMPLING_RATE, nperseg=2048)
    freqs_after, psd_after = welch(after[0, 0], fs=DEAP_SAMPLING_RATE, nperseg=2048)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.semilogy(freqs_before, psd_before, label="default/no ICA")
    axis.semilogy(freqs_after, psd_after, label="candidate ICA cleaned")
    axis.set_title(f"{subject} trial 0 channel 0 PSD before/after candidate ICA cleaning")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD")
    axis.set_xlim(0, 60)
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def max_abs_eog_corr_by_eeg(data: np.ndarray, eog_data: np.ndarray | None) -> tuple[float | None, float | None]:
    """Return mean/max per-EEG-channel max abs corr with EOG-like channels."""
    if eog_data is None or eog_data.size == 0:
        return None, None

    eeg_flat = zscore_columns(flatten_trials(data[:, :EEG_CHANNEL_COUNT, :]))
    eog_flat = zscore_columns(flatten_trials(eog_data))
    corr = np.corrcoef(eeg_flat.T, eog_flat.T)[: eeg_flat.shape[1], eeg_flat.shape[1] :]
    abs_corr = np.nan_to_num(np.abs(corr), nan=0.0)
    channel_max = abs_corr.max(axis=1)
    return float(channel_max.mean()), float(channel_max.max())


def load_eog_data(subject_id: int, n_trials: int, n_samples: int) -> tuple[np.ndarray | None, str]:
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"{subject_label(subject_id)}.bdf"
    header = read_bdf_header(bdf_path)
    aux = find_auxiliary_channels(header)
    eog_data = load_auxiliary_subset(subject_id, aux["eog_indices"], n_trials, n_samples)
    return eog_data, ", ".join(aux["eog_names"]) or "unavailable"


def per_channel_std_rows(subject: str, before: np.ndarray, after: np.ndarray) -> list[dict]:
    before_std = before.std(axis=(0, 2))
    after_std = after.std(axis=(0, 2))
    rows = []
    for channel in range(min(EEG_CHANNEL_COUNT, before.shape[1])):
        rows.append(
            {
                "subject_id": subject,
                "channel": channel,
                "std_before": float(before_std[channel]),
                "std_after": float(after_std[channel]),
                "std_change_pct": pct_change(float(before_std[channel]), float(after_std[channel])),
            }
        )
    return rows


def interpretation(summary: dict) -> tuple[str, str, str]:
    high_freq_drop = -summary["high_freq_ratio_change_pct"]
    std_drop = -summary["std_change_pct"]
    var_drop = -summary["var_change_pct"]
    eog_drop = None
    if summary["eog_corr_mean_before"] != "":
        eog_drop = -summary["eog_corr_mean_change_pct"]

    if std_drop > 30 or var_drop > 50 or abs(summary["max_abs_channel_std_change_pct"]) > 40:
        overclean = "possible over-cleaning risk"
    else:
        overclean = "no obvious over-cleaning from simple statistics"

    if eog_drop is not None and eog_drop > 10:
        eog_effect = "EOG-like correlation decreased"
    elif eog_drop is None:
        eog_effect = "EOG-like correlation unavailable"
    else:
        eog_effect = "EOG-like correlation did not clearly decrease"

    if high_freq_drop > 5 and "no obvious" in overclean:
        optional = "can be considered as subject-specific optional enhancement after visual review"
    else:
        optional = "do not promote yet; keep as manual-review candidate"

    return eog_effect, overclean, optional


def analyze_subject(subject_id: int, exclude_components: list[int]) -> tuple[dict, list[dict]]:
    subject = subject_label(subject_id)
    output_dir = RESULTS_DIR / subject
    print(f"testing candidate ICA cleaning for {subject}: exclude={exclude_components}")

    default = run_default_preprocessing(subject_id)
    cleaned = run_candidate_preprocessing(subject_id, exclude_components)
    before = default["baseline_corrected_stimulus"]
    after = cleaned["baseline_corrected_stimulus"]
    before_stats = overall_stats(before)
    after_stats = overall_stats(after)
    channel_rows = per_channel_std_rows(subject, before, after)

    eog_data, eog_names = load_eog_data(subject_id, before.shape[0], before.shape[-1])
    eog_before_mean, eog_before_max = max_abs_eog_corr_by_eeg(before, eog_data)
    eog_after_mean, eog_after_max = max_abs_eog_corr_by_eeg(after, eog_data)

    plot_waveform(subject, before, after, output_dir)
    plot_psd(subject, before, after, output_dir)

    std_changes = [row["std_change_pct"] for row in channel_rows]
    summary = {
        "subject_id": subject,
        "candidate_exclude_components": str(exclude_components),
        "removed_components": str(cleaned["ica_info"]["removed_components"]),
        "before_shape": str(tuple(before.shape)),
        "after_shape": str(tuple(after.shape)),
        "shape_ok": before.shape == after.shape == (40, EEG_CHANNEL_COUNT, before.shape[-1]),
        "mean_before": before_stats["mean"],
        "mean_after": after_stats["mean"],
        "mean_change_pct": pct_change(before_stats["mean"], after_stats["mean"]),
        "std_before": before_stats["std"],
        "std_after": after_stats["std"],
        "std_change_pct": pct_change(before_stats["std"], after_stats["std"]),
        "var_before": before_stats["var"],
        "var_after": after_stats["var"],
        "var_change_pct": pct_change(before_stats["var"], after_stats["var"]),
        "high_freq_ratio_before": before_stats["high_freq_ratio_30_45"],
        "high_freq_ratio_after": after_stats["high_freq_ratio_30_45"],
        "high_freq_ratio_change_pct": pct_change(
            before_stats["high_freq_ratio_30_45"],
            after_stats["high_freq_ratio_30_45"],
        ),
        "mean_channel_std_change_pct": float(np.mean(std_changes)),
        "max_abs_channel_std_change_pct": float(np.max(np.abs(std_changes))),
        "eog_like_channels": eog_names,
        "eog_corr_mean_before": "" if eog_before_mean is None else eog_before_mean,
        "eog_corr_mean_after": "" if eog_after_mean is None else eog_after_mean,
        "eog_corr_mean_change_pct": "" if eog_before_mean is None else pct_change(eog_before_mean, eog_after_mean),
        "eog_corr_max_before": "" if eog_before_max is None else eog_before_max,
        "eog_corr_max_after": "" if eog_after_max is None else eog_after_max,
        "eog_corr_max_change_pct": "" if eog_before_max is None else pct_change(eog_before_max, eog_after_max),
    }
    eog_effect, overclean, optional = interpretation(summary)
    summary["eog_effect_judgment"] = eog_effect
    summary["overcleaning_risk_judgment"] = overclean
    summary["optional_enhancement_judgment"] = optional
    return summary, channel_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary_rows: list[dict]) -> str:
    lines = [
        "MNE ICA candidate cleaning test report",
        "",
        "Purpose:",
        "Compare default preprocessing against subject-specific candidate ICA cleaning. "
        "This script does not modify preprocess_subject defaults.",
        "",
        "Candidate settings:",
        "s10: exclude [0]",
        "s24: exclude [0, 7]",
        "",
        "Important limitation:",
        "Candidates came from MNE ICA review. This test applies the existing "
        "preprocess.py FastICA-based cleaning interface, so component numbering "
        "should be treated as subject-specific evidence to verify, not a global rule.",
        "",
        "Subject results:",
    ]
    for row in summary_rows:
        lines.extend(
            [
                "",
                f"{row['subject_id']} exclude={row['candidate_exclude_components']}",
                f"shape check: before={row['before_shape']}, after={row['after_shape']}, ok={row['shape_ok']}",
                f"std change: {row['std_change_pct']:.2f}%",
                f"var change: {row['var_change_pct']:.2f}%",
                f"30-45 Hz ratio change: {row['high_freq_ratio_change_pct']:.2f}%",
                f"mean per-channel std change: {row['mean_channel_std_change_pct']:.2f}%",
                f"max abs per-channel std change: {row['max_abs_channel_std_change_pct']:.2f}%",
                f"EOG-like channels: {row['eog_like_channels']}",
                f"EOG judgment: {row['eog_effect_judgment']}",
                f"over-cleaning judgment: {row['overcleaning_risk_judgment']}",
                f"optional enhancement judgment: {row['optional_enhancement_judgment']}",
            ]
        )
        if row["eog_corr_mean_before"] != "":
            lines.append(
                "EOG-like mean corr change: "
                f"{row['eog_corr_mean_before']:.4f} -> {row['eog_corr_mean_after']:.4f} "
                f"({row['eog_corr_mean_change_pct']:.2f}%)"
            )
            lines.append(
                "EOG-like max corr change: "
                f"{row['eog_corr_max_before']:.4f} -> {row['eog_corr_max_after']:.4f} "
                f"({row['eog_corr_max_change_pct']:.2f}%)"
            )

    lines.extend(
        [
            "",
            "Final conservative conclusion:",
            "Do not change preprocess_subject default parameters from this test alone.",
            "Do not treat component 0 or 7 as global default artifact components.",
            "Use these results only as subject-specific optional evidence together with waveform, PSD, and manual ICA review figures.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure_plot_cache()
    summary_rows = []
    all_channel_rows = []
    for subject_id, exclude_components in CANDIDATE_EXCLUDES.items():
        summary, channel_rows = analyze_subject(subject_id, exclude_components)
        summary_rows.append(summary)
        all_channel_rows.extend(channel_rows)

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(CHANNEL_STD_CSV, all_channel_rows)
    REPORT_PATH.write_text(build_report(summary_rows), encoding="utf-8")
    print(f"saved summary csv to {SUMMARY_CSV}")
    print(f"saved channel std csv to {CHANNEL_STD_CSV}")
    print(f"saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
