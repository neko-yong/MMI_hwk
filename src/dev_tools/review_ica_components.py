"""Manual ICA component review helper.

This script does not change the preprocessing pipeline. It creates plots and a
review table so a human reviewer can decide whether any subject-specific ICA
component should be excluded.
"""

from __future__ import annotations

import csv
import os
from array import array
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

from src.preprocess import (
    DEFAULT_ORIGINAL_DIR,
    EEG_CHANNEL_COUNT,
    PROJECT_ROOT,
    get_trial_boundaries_from_bdf,
    preprocess_subject,
    read_bdf_header,
)


SUBJECT_IDS = (1, 2, 3, 10, 20, 24, 32)
RESULTS_DIR = Path("results/ica_manual_review")
REVIEW_TABLE_PATH = RESULTS_DIR / "ica_manual_review_table.csv"
REPORT_PATH = RESULTS_DIR / "ica_manual_review_report.txt"

N_COMPONENTS = 16
RANDOM_STATE = 42
SAMPLING_RATE = 512
ANALYSIS_TRIALS = 10
ANALYSIS_SECONDS = 20
PLOT_SECONDS = 5


def configure_plot_cache() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))


def get_pyplot():
    configure_plot_cache()
    import matplotlib.pyplot as plt

    return plt


def signed_int24_values(raw_bytes: bytes):
    for index in range(0, len(raw_bytes), 3):
        value = (
            raw_bytes[index]
            | (raw_bytes[index + 1] << 8)
            | (raw_bytes[index + 2] << 16)
        )
        if value & 0x800000:
            value -= 0x1000000
        yield value


def read_channels_interval(
    bdf_path: Path,
    header: dict,
    channel_indices: list[int],
    start_sample: int,
    end_sample: int,
) -> np.ndarray:
    """Read selected BDF channels in a sample interval as (channels, samples)."""
    samples_per_record = header["samples_per_record"]
    samples_per_record_ref = samples_per_record[0]
    first_record = start_sample // samples_per_record_ref
    last_record = (end_sample - 1) // samples_per_record_ref
    record_bytes = sum(samples_per_record) * 3
    channel_data = [array("i") for _ in channel_indices]

    with bdf_path.open("rb") as file:
        for record_index in range(first_record, last_record + 1):
            record_start_sample = record_index * samples_per_record_ref
            local_start = max(start_sample - record_start_sample, 0)
            local_end = min(end_sample - record_start_sample, samples_per_record_ref)

            for out_index, channel_index in enumerate(channel_indices):
                channel_offset = sum(samples_per_record[:channel_index]) * 3
                channel_bytes = samples_per_record[channel_index] * 3
                file.seek(
                    header["header_bytes"]
                    + record_index * record_bytes
                    + channel_offset
                )
                values = list(signed_int24_values(file.read(channel_bytes)))
                channel_data[out_index].extend(values[local_start:local_end])

    return np.asarray([list(channel) for channel in channel_data], dtype=np.float32)


def flatten_trials(data: np.ndarray) -> np.ndarray:
    """Convert (trials, channels, samples) to (trials*samples, channels)."""
    return data.transpose(0, 2, 1).reshape(-1, data.shape[1])


def zscore_columns(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def find_auxiliary_channels(header: dict) -> dict:
    """Find broad EOG/EMG-like auxiliary channels for review only."""
    labels = header["channel_labels"]
    eog_indices = []
    emg_indices = []

    for index, name in enumerate(labels):
        lowered = name.lower()
        if "eog" in lowered or lowered in {"exg1", "exg2", "exg3", "exg4"}:
            eog_indices.append(index)
        if "emg" in lowered or lowered in {"exg5", "exg6", "exg7", "exg8"}:
            emg_indices.append(index)

    return {
        "eog_indices": eog_indices,
        "emg_indices": emg_indices,
        "eog_names": [labels[index] for index in eog_indices],
        "emg_names": [labels[index] for index in emg_indices],
    }


def load_auxiliary_subset(
    subject_id: int,
    aux_indices: list[int],
    n_trials: int,
    n_samples: int,
) -> np.ndarray | None:
    """Load auxiliary channels for the same stimulus subset used by ICA."""
    if not aux_indices:
        return None

    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    boundaries = get_trial_boundaries_from_bdf(bdf_path)
    segments = []
    for boundary in boundaries[:n_trials]:
        start = boundary["stimulus_start_sample"]
        segments.append(
            read_channels_interval(
                bdf_path,
                header,
                aux_indices,
                start,
                start + n_samples,
            )
        )
    return np.asarray(segments, dtype=np.float32)


def max_abs_correlation(
    sources: np.ndarray,
    aux_data: np.ndarray | None,
) -> tuple[np.ndarray, list[int]]:
    if aux_data is None or aux_data.size == 0:
        return np.full(sources.shape[1], np.nan), [-1] * sources.shape[1]

    aux_flat = zscore_columns(flatten_trials(aux_data))
    source_z = zscore_columns(sources)
    corr = np.corrcoef(source_z.T, aux_flat.T)[: source_z.shape[1], source_z.shape[1] :]
    abs_corr = np.nan_to_num(np.abs(corr), nan=0.0)
    best_indices = np.argmax(abs_corr, axis=1)
    best_values = abs_corr[np.arange(abs_corr.shape[0]), best_indices]
    return best_values, best_indices.tolist()


def high_freq_ratio(signal: np.ndarray) -> float:
    freqs, psd = welch(signal, fs=SAMPLING_RATE, nperseg=min(1024, len(signal)))
    high_mask = (freqs >= 30) & (freqs <= 45)
    total_mask = (freqs >= 4) & (freqs <= 45)
    total_power = float(psd[total_mask].sum())
    return float(psd[high_mask].sum()) / total_power if total_power > 0 else 0.0


def suspicion_level(row: dict) -> tuple[str, str]:
    """Conservative automatic suspicion label, not an exclusion decision."""
    reasons = []
    score = 0

    if row["eog_corr_max"] >= 0.35:
        score += 3
        reasons.append("high EOG-like correlation")
    elif row["eog_corr_max"] >= 0.25:
        score += 1
        reasons.append("moderate EOG-like correlation")

    if row["emg_corr_max"] >= 0.35:
        score += 3
        reasons.append("high EMG-like correlation")
    elif row["emg_corr_max"] >= 0.25:
        score += 1
        reasons.append("moderate EMG-like correlation")

    if row["high_freq_ratio"] >= 0.40:
        score += 2
        reasons.append("very high 30-45 Hz ratio")
    elif row["high_freq_ratio"] >= 0.30:
        score += 1
        reasons.append("moderate 30-45 Hz ratio")

    if row["kurtosis_or_peak_score"] >= 10:
        score += 1
        reasons.append("large peak/kurtosis score")

    if score >= 4 and any("high" in reason for reason in reasons):
        return "high", "; ".join(reasons)
    if score >= 2:
        return "medium", "; ".join(reasons)
    return "low", "; ".join(reasons) or "no strong automatic artifact evidence"


def fit_subject_ica(subject_id: int) -> dict:
    """Fit ICA on filtered EEG only; no component is removed."""
    preprocessed = preprocess_subject(subject_id=subject_id, use_ica=False)
    filtered = preprocessed["filtered_stimulus"]
    n_trials = min(ANALYSIS_TRIALS, filtered.shape[0])
    n_samples = min(ANALYSIS_SECONDS * SAMPLING_RATE, filtered.shape[-1])
    eeg_subset = filtered[:n_trials, :EEG_CHANNEL_COUNT, :n_samples]
    flattened = flatten_trials(eeg_subset)

    ica = FastICA(
        n_components=min(N_COMPONENTS, EEG_CHANNEL_COUNT),
        random_state=RANDOM_STATE,
        whiten="unit-variance",
        max_iter=300,
        tol=0.001,
    )
    sources = ica.fit_transform(flattened)
    source_trials = sources.reshape(n_trials, n_samples, sources.shape[1])

    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    aux = find_auxiliary_channels(header)
    eog_data = load_auxiliary_subset(subject_id, aux["eog_indices"], n_trials, n_samples)
    emg_data = load_auxiliary_subset(subject_id, aux["emg_indices"], n_trials, n_samples)

    return {
        "subject_id": f"s{subject_id:02d}",
        "subject_num": subject_id,
        "sources": sources,
        "source_trials": source_trials,
        "mixing": getattr(ica, "mixing_", None),
        "eeg_channel_names": header["channel_labels"][:EEG_CHANNEL_COUNT],
        "eog_data": eog_data,
        "emg_data": emg_data,
        "eog_names": aux["eog_names"],
        "emg_names": aux["emg_names"],
        "n_trials": n_trials,
        "n_samples": n_samples,
    }


def compute_component_rows(fitted: dict) -> list[dict]:
    sources = fitted["sources"]
    eog_corr, _ = max_abs_correlation(sources, fitted["eog_data"])
    emg_corr, _ = max_abs_correlation(sources, fitted["emg_data"])
    rows = []

    for component in range(sources.shape[1]):
        signal = sources[:, component]
        signal_std = signal.std() or 1.0
        peak_z = float(np.max(np.abs((signal - signal.mean()) / signal_std)))
        kurt = float(kurtosis(signal, fisher=False, nan_policy="omit"))
        row = {
            "subject_id": fitted["subject_id"],
            "component_id": component,
            "eog_corr_max": float(eog_corr[component]) if not np.isnan(eog_corr[component]) else 0.0,
            "emg_corr_max": float(emg_corr[component]) if not np.isnan(emg_corr[component]) else 0.0,
            "high_freq_ratio": high_freq_ratio(signal),
            "kurtosis_or_peak_score": max(kurt, peak_z),
            "visual_review_decision": "",
            "final_exclude_decision": "",
            "notes": "",
        }
        level, reason = suspicion_level(row)
        row["auto_suspicion_level"] = level
        row["auto_suspicion_reason"] = reason
        rows.append(row)

    return rows


def top_review_components(rows: list[dict], n: int = 5) -> list[dict]:
    level_weight = {"high": 2, "medium": 1, "low": 0}
    return sorted(
        rows,
        key=lambda row: (
            level_weight[row["auto_suspicion_level"]],
            row["eog_corr_max"],
            row["emg_corr_max"],
            row["high_freq_ratio"],
        ),
        reverse=True,
    )[:n]


def plot_component_timeseries_overview(fitted: dict, output_dir: Path) -> Path:
    plt = get_pyplot()
    sources = fitted["source_trials"][0]
    n_components = sources.shape[1]
    max_samples = min(PLOT_SECONDS * SAMPLING_RATE, sources.shape[0])
    time_axis = np.arange(max_samples) / SAMPLING_RATE
    path = output_dir / f"{fitted['subject_id']}_all_components_timeseries.png"

    figure, axes = plt.subplots(4, 4, figsize=(14, 9), sharex=True)
    for component, axis in enumerate(axes.ravel()[:n_components]):
        axis.plot(time_axis, sources[:max_samples, component], linewidth=0.8)
        axis.set_title(f"C{component:02d}", fontsize=9)
        axis.grid(True, linestyle="--", alpha=0.2)
    figure.suptitle(f"{fitted['subject_id']} ICA component time series overview")
    figure.supxlabel("Time (s)")
    figure.supylabel("Activation")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_component_psd_overview(fitted: dict, output_dir: Path) -> Path:
    plt = get_pyplot()
    sources = fitted["sources"]
    n_components = sources.shape[1]
    path = output_dir / f"{fitted['subject_id']}_all_components_psd.png"

    figure, axes = plt.subplots(4, 4, figsize=(14, 9), sharex=True)
    for component, axis in enumerate(axes.ravel()[:n_components]):
        freqs, psd = welch(sources[:, component], fs=SAMPLING_RATE, nperseg=1024)
        axis.semilogy(freqs, psd, linewidth=0.8)
        axis.set_title(f"C{component:02d}", fontsize=9)
        axis.set_xlim(0, 60)
        axis.grid(True, linestyle="--", alpha=0.2)
    figure.suptitle(f"{fitted['subject_id']} ICA component PSD overview")
    figure.supxlabel("Frequency (Hz)")
    figure.supylabel("PSD")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_bar_metric(
    fitted: dict,
    rows: list[dict],
    metric: str,
    ylabel: str,
    filename_suffix: str,
    output_dir: Path,
) -> Path:
    plt = get_pyplot()
    components = [row["component_id"] for row in rows]
    values = [row[metric] for row in rows]
    path = output_dir / f"{fitted['subject_id']}_{filename_suffix}.png"

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(components, values)
    axis.set_title(f"{fitted['subject_id']} {ylabel}")
    axis.set_xlabel("ICA component")
    axis.set_ylabel(ylabel)
    axis.set_xticks(components)
    axis.grid(True, axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_topomap_if_available(fitted: dict, output_dir: Path) -> str:
    """Try MNE topomaps. Return a short status for the report."""
    if fitted["mixing"] is None:
        return "topomap unavailable: ICA mixing matrix missing"

    try:
        import mne
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local optional package
        return f"topomap unavailable: MNE import failed ({exc})"

    try:
        ch_names = fitted["eeg_channel_names"]
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="ignore")
        path = output_dir / f"{fitted['subject_id']}_ica_topomap_overview.png"

        n_components = min(fitted["mixing"].shape[1], N_COMPONENTS)
        figure, axes = plt.subplots(4, 4, figsize=(12, 10))
        for component, axis in enumerate(axes.ravel()[:n_components]):
            mne.viz.plot_topomap(
                fitted["mixing"][:, component],
                info,
                axes=axis,
                show=False,
                contours=0,
            )
            axis.set_title(f"C{component:02d}", fontsize=9)
        figure.suptitle(f"{fitted['subject_id']} ICA spatial maps")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return f"topomap saved: {path}"
    except Exception as exc:  # pragma: no cover - plotting can vary by install
        return f"topomap unavailable: {exc}"


def save_subject_review_plots(fitted: dict, rows: list[dict]) -> str:
    output_dir = RESULTS_DIR / fitted["subject_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_component_timeseries_overview(fitted, output_dir)
    plot_component_psd_overview(fitted, output_dir)
    plot_bar_metric(fitted, rows, "eog_corr_max", "max abs EOG-like correlation", "eog_correlation", output_dir)
    plot_bar_metric(fitted, rows, "emg_corr_max", "max abs EMG-like correlation", "emg_correlation", output_dir)
    plot_bar_metric(fitted, rows, "high_freq_ratio", "30-45 Hz high-frequency ratio", "high_freq_ratio_ranking", output_dir)
    return plot_topomap_if_available(fitted, output_dir)


def write_review_table(rows: list[dict]) -> None:
    REVIEW_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "component_id",
        "eog_corr_max",
        "emg_corr_max",
        "high_freq_ratio",
        "kurtosis_or_peak_score",
        "auto_suspicion_level",
        "visual_review_decision",
        "final_exclude_decision",
        "notes",
        "auto_suspicion_reason",
    ]
    with REVIEW_TABLE_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(subject_reports: list[dict]) -> str:
    lines = [
        "ICA manual review helper report",
        "",
        "Purpose:",
        "This script prepares visual and tabular evidence for manual ICA review. "
        "It does not delete components and does not change preprocess_subject defaults.",
        "",
        "Settings:",
        f"subjects: {', '.join(report['subject_id'] for report in subject_reports)}",
        f"ICA n_components: {N_COMPONENTS}",
        f"random_state: {RANDOM_STATE}",
        f"analysis subset: first {ANALYSIS_TRIALS} trials, first {ANALYSIS_SECONDS}s of filtered stimulus",
        "",
        "Subject-specific components worth manual review:",
    ]

    high_count = 0
    for report in subject_reports:
        lines.extend(
            [
                "",
                f"{report['subject_id']}:",
                f"EOG-like channels: {report['eog_names'] or 'unavailable'}",
                f"EMG-like channels: {report['emg_names'] or 'unavailable'}",
                f"Topomap status: {report['topomap_status']}",
            ]
        )
        for row in report["top_rows"]:
            if row["auto_suspicion_level"] == "high":
                high_count += 1
            lines.append(
                "component "
                f"{row['component_id']:02d}: level={row['auto_suspicion_level']}, "
                f"EOG={row['eog_corr_max']:.3f}, EMG={row['emg_corr_max']:.3f}, "
                f"HF={row['high_freq_ratio']:.3f}, peak/kurt={row['kurtosis_or_peak_score']:.3f}, "
                f"reason={row['auto_suspicion_reason']}"
            )

    lines.extend(
        [
            "",
            "Default strategy recommendation:",
            "Do not automatically modify preprocess_subject defaults. Keep "
            "ica_exclude_components empty unless a subject-specific component is "
            "confirmed by manual visual review.",
            "",
            "Uncertainty:",
            "EOG/EMG channel matching is intentionally broad for DEAP raw BDF files. "
            "Use the generated plots and the review table before making exclusion decisions.",
        ]
    )
    if high_count == 0:
        lines.append("No component reached a strong enough automatic pattern to justify automatic deletion.")
    else:
        lines.append(
            f"{high_count} component(s) were marked high suspicion, but this remains a manual-review cue only."
        )
    return "\n".join(lines)


def main() -> None:
    configure_plot_cache()
    all_rows = []
    subject_reports = []

    for subject_id in SUBJECT_IDS:
        print(f"preparing ICA manual review for s{subject_id:02d}...")
        fitted = fit_subject_ica(subject_id)
        rows = compute_component_rows(fitted)
        topomap_status = save_subject_review_plots(fitted, rows)
        all_rows.extend(rows)
        subject_reports.append(
            {
                "subject_id": fitted["subject_id"],
                "eog_names": ", ".join(fitted["eog_names"]),
                "emg_names": ", ".join(fitted["emg_names"]),
                "topomap_status": topomap_status,
                "top_rows": top_review_components(rows),
            }
        )

    write_review_table(all_rows)
    REPORT_PATH.write_text(build_report(subject_reports), encoding="utf-8")
    print(f"saved manual review table to {REVIEW_TABLE_PATH}")
    print(f"saved manual review report to {REPORT_PATH}")
    print(f"saved subject figures under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
