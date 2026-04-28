"""Multi-subject ICA artifact candidate analysis.

This script analyzes ICA components for a small set of DEAP subjects without
changing the default preprocessing strategy. It is a diagnostic tool: it reads
filtered EEG from preprocess_subject, fits ICA on a short stimulus subset, and
scores components using conservative artifact-like indicators.
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
RESULTS_DIR = Path("results/ica_artifact_multisubject")
SUBJECT_SUMMARY_CSV = RESULTS_DIR / "ica_artifact_subject_summary.csv"
COMPONENT_SCORES_CSV = RESULTS_DIR / "ica_artifact_component_scores.csv"
REPORT_PATH = RESULTS_DIR / "ica_artifact_recommendation_report.txt"
N_COMPONENTS = 16
RANDOM_STATE = 42
ANALYSIS_TRIALS = 10
ANALYSIS_SECONDS = 20
SAMPLING_RATE = 512
MAX_PLOT_SECONDS = 5


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
    """Read arbitrary BDF channel interval as (channels, samples)."""
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
                raw_channel = file.read(channel_bytes)
                values = list(signed_int24_values(raw_channel))
                channel_data[out_index].extend(values[local_start:local_end])

    return np.asarray([list(channel) for channel in channel_data], dtype=np.float32)


def find_auxiliary_channels(header: dict) -> dict:
    """Find EOG/EMG-like channels with broad matching.

    DEAP raw BDF files often use EXG names rather than explicit hEOG/vEOG/zEMG.
    In that case, this script treats EXG1-EXG4 as EOG-like and EXG5-EXG8 as
    EMG-like for diagnostic correlation only.
    """
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


def load_auxiliary_stimulus_subset(
    subject_id: int,
    aux_indices: list[int],
    n_trials: int,
    n_samples: int,
) -> np.ndarray | None:
    """Load auxiliary channels for the same trial/time subset."""
    if not aux_indices:
        return None

    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    boundaries = get_trial_boundaries_from_bdf(bdf_path)
    segments = []

    for boundary in boundaries[:n_trials]:
        start = boundary["stimulus_start_sample"]
        end = start + n_samples
        segments.append(read_channels_interval(bdf_path, header, aux_indices, start, end))

    return np.asarray(segments, dtype=np.float32)


def flatten_trials(data: np.ndarray) -> np.ndarray:
    """Convert (trials, channels, samples) to (trials*samples, channels)."""
    return data.transpose(0, 2, 1).reshape(-1, data.shape[1])


def zscore_columns(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def max_abs_correlation(sources: np.ndarray, aux_data: np.ndarray | None) -> tuple[np.ndarray, list[int]]:
    """Return max absolute correlation between each component and aux channels."""
    if aux_data is None or aux_data.size == 0:
        return np.full(sources.shape[1], np.nan), [-1] * sources.shape[1]

    aux_flat = zscore_columns(flatten_trials(aux_data))
    source_z = zscore_columns(sources)
    corr = np.corrcoef(source_z.T, aux_flat.T)[: source_z.shape[1], source_z.shape[1] :]
    abs_corr = np.abs(corr)
    best_indices = np.nanargmax(abs_corr, axis=1)
    best_values = abs_corr[np.arange(abs_corr.shape[0]), best_indices]
    return best_values, best_indices.tolist()


def component_high_freq_ratio(component_signal: np.ndarray, sampling_rate: int) -> float:
    freqs, psd = welch(
        component_signal,
        fs=sampling_rate,
        nperseg=min(1024, len(component_signal)),
    )
    high_mask = (freqs >= 30) & (freqs <= 45)
    total_mask = (freqs >= 4) & (freqs <= 45)
    high_power = float(psd[high_mask].sum())
    total_power = float(psd[total_mask].sum())
    return high_power / total_power if total_power > 0 else 0.0


def score_component(row: dict) -> tuple[float, list[str]]:
    """Conservative component score and human-readable evidence."""
    score = 0.0
    reasons = []

    if row["max_abs_eog_corr"] >= 0.35:
        score += 3.0
        reasons.append("high EOG correlation")
    elif row["max_abs_eog_corr"] >= 0.25:
        score += 1.5
        reasons.append("moderate EOG correlation")

    if row["max_abs_emg_corr"] >= 0.35:
        score += 3.0
        reasons.append("high EMG correlation")
    elif row["max_abs_emg_corr"] >= 0.25:
        score += 1.5
        reasons.append("moderate EMG correlation")

    if row["high_freq_ratio_30_45"] >= 0.35:
        score += 2.0
        reasons.append("high 30-45 Hz ratio")
    elif row["high_freq_ratio_30_45"] >= 0.25:
        score += 1.0
        reasons.append("moderate 30-45 Hz ratio")

    if row["kurtosis"] >= 8:
        score += 1.5
        reasons.append("high kurtosis")
    if row["peak_z"] >= 8:
        score += 1.0
        reasons.append("large peak z-score")

    return score, reasons


def fit_ica_for_subject(subject_id: int) -> dict:
    """Load filtered EEG subset and fit ICA."""
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
    eog_data = load_auxiliary_stimulus_subset(subject_id, aux["eog_indices"], n_trials, n_samples)
    emg_data = load_auxiliary_stimulus_subset(subject_id, aux["emg_indices"], n_trials, n_samples)

    return {
        "subject_id": subject_id,
        "eeg_subset": eeg_subset,
        "sources": sources,
        "source_trials": source_trials,
        "mixing": getattr(ica, "mixing_", None),
        "eog_data": eog_data,
        "emg_data": emg_data,
        "eog_names": aux["eog_names"],
        "emg_names": aux["emg_names"],
        "n_trials": n_trials,
        "n_samples": n_samples,
    }


def analyze_subject_components(subject_id: int) -> tuple[dict, list[dict], dict]:
    """Compute component metrics and subject-level recommendation."""
    fitted = fit_ica_for_subject(subject_id)
    sources = fitted["sources"]
    eog_corr, eog_best = max_abs_correlation(sources, fitted["eog_data"])
    emg_corr, emg_best = max_abs_correlation(sources, fitted["emg_data"])
    component_rows = []

    for component in range(sources.shape[1]):
        signal = sources[:, component]
        signal_z = (signal - signal.mean()) / (signal.std() or 1.0)
        row = {
            "subject_id": f"s{subject_id:02d}",
            "component": component,
            "component_energy": float(np.var(signal)),
            "high_freq_ratio_30_45": component_high_freq_ratio(signal, SAMPLING_RATE),
            "max_abs_eog_corr": float(eog_corr[component]) if not np.isnan(eog_corr[component]) else "",
            "best_eog_channel_index": eog_best[component],
            "max_abs_emg_corr": float(emg_corr[component]) if not np.isnan(emg_corr[component]) else "",
            "best_emg_channel_index": emg_best[component],
            "kurtosis": float(kurtosis(signal, fisher=False)),
            "peak_z": float(np.max(np.abs(signal_z))),
        }
        score, reasons = score_component(
            {
                **row,
                "max_abs_eog_corr": 0.0 if row["max_abs_eog_corr"] == "" else row["max_abs_eog_corr"],
                "max_abs_emg_corr": 0.0 if row["max_abs_emg_corr"] == "" else row["max_abs_emg_corr"],
            }
        )
        row["artifact_score"] = score
        row["score_reasons"] = "; ".join(reasons)
        component_rows.append(row)

    ranked = sorted(component_rows, key=lambda item: item["artifact_score"], reverse=True)
    recommended = [
        row["component"]
        for row in ranked
        if row["artifact_score"] >= 5.0
        and (
            (row["max_abs_eog_corr"] != "" and row["max_abs_eog_corr"] >= 0.35)
            or (row["max_abs_emg_corr"] != "" and row["max_abs_emg_corr"] >= 0.35)
        )
    ][:2]
    recommendation_reason = (
        "strong auxiliary-channel correlation plus supporting component metrics"
        if recommended
        else "evidence insufficient for default component deletion"
    )
    subject_summary = {
        "subject_id": f"s{subject_id:02d}",
        "n_trials_used": fitted["n_trials"],
        "seconds_used_per_trial": fitted["n_samples"] / SAMPLING_RATE,
        "eog_channels": ", ".join(fitted["eog_names"]) or "unavailable",
        "emg_channels": ", ".join(fitted["emg_names"]) or "unavailable",
        "top_eog_like_components": top_components(component_rows, "max_abs_eog_corr"),
        "top_emg_like_components": top_components(component_rows, "max_abs_emg_corr"),
        "top_high_frequency_components": top_components(component_rows, "high_freq_ratio_30_45"),
        "recommend_exclude": str(recommended),
        "recommendation_reason": recommendation_reason,
    }

    return subject_summary, component_rows, fitted


def top_components(rows: list[dict], key: str, n: int = 3) -> str:
    valid_rows = [row for row in rows if row[key] != ""]
    ranked = sorted(valid_rows, key=lambda row: row[key], reverse=True)
    return "; ".join(
        f"c{row['component']}={row[key]:.3f}"
        for row in ranked[:n]
    ) or "unavailable"


def save_subject_plots(subject_id: int, component_rows: list[dict], fitted: dict) -> list[Path]:
    """Save compact subject-level ICA diagnostic plots."""
    plt = get_pyplot()
    output_dir = RESULTS_DIR / f"s{subject_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    top_rows = sorted(component_rows, key=lambda row: row["artifact_score"], reverse=True)[:3]
    saved = []

    for row in top_rows:
        component = row["component"]
        signal = fitted["source_trials"][0, :, component]
        max_samples = min(MAX_PLOT_SECONDS * SAMPLING_RATE, len(signal))
        time_axis = np.arange(max_samples) / SAMPLING_RATE

        path = output_dir / f"s{subject_id:02d}_component_{component:02d}_timeseries.png"
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(time_axis, signal[:max_samples], linewidth=1.0)
        axis.set_title(f"s{subject_id:02d} ICA component {component} time series")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Activation")
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        saved.append(path)

        freqs, psd = welch(fitted["sources"][:, component], fs=SAMPLING_RATE, nperseg=1024)
        path = output_dir / f"s{subject_id:02d}_component_{component:02d}_psd.png"
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.semilogy(freqs, psd)
        axis.set_title(f"s{subject_id:02d} ICA component {component} PSD")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.set_xlim(0, 60)
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        saved.append(path)

    path = output_dir / f"s{subject_id:02d}_component_score_ranking.png"
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        [row["component"] for row in component_rows],
        [row["artifact_score"] for row in component_rows],
    )
    axis.set_title(f"s{subject_id:02d} ICA component score ranking")
    axis.set_xlabel("Component")
    axis.set_ylabel("Artifact score")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    saved.append(path)

    path = output_dir / f"s{subject_id:02d}_aux_correlation.png"
    figure, axis = plt.subplots(figsize=(9, 4))
    components = [row["component"] for row in component_rows]
    eog = [0 if row["max_abs_eog_corr"] == "" else row["max_abs_eog_corr"] for row in component_rows]
    emg = [0 if row["max_abs_emg_corr"] == "" else row["max_abs_emg_corr"] for row in component_rows]
    axis.bar(np.asarray(components) - 0.2, eog, width=0.4, label="EOG-like")
    axis.bar(np.asarray(components) + 0.2, emg, width=0.4, label="EMG-like")
    axis.set_title(f"s{subject_id:02d} ICA component auxiliary correlation")
    axis.set_xlabel("Component")
    axis.set_ylabel("Max abs correlation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    saved.append(path)

    return saved


def save_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_report(subject_rows: list[dict]) -> str:
    recommended_non_empty = [
        row for row in subject_rows if row["recommend_exclude"] != "[]"
    ]
    lines = [
        "Multi-subject ICA artifact candidate report",
        "",
        "Analysis scope:",
        f"subjects: {', '.join(row['subject_id'] for row in subject_rows)}",
        f"ICA n_components: {N_COMPONENTS}",
        f"random_state: {RANDOM_STATE}",
        f"subset: first {ANALYSIS_TRIALS} trials, first {ANALYSIS_SECONDS}s stimulus per trial",
        "",
        "Subject recommendations:",
    ]
    for row in subject_rows:
        lines.extend(
            [
                "",
                f"{row['subject_id']}: recommend_exclude={row['recommend_exclude']}",
                f"reason: {row['recommendation_reason']}",
                f"EOG channels: {row['eog_channels']}",
                f"EMG channels: {row['emg_channels']}",
                f"top EOG-like: {row['top_eog_like_components']}",
                f"top EMG-like: {row['top_emg_like_components']}",
                f"top high-frequency: {row['top_high_frequency_components']}",
            ]
        )

    lines.extend(
        [
            "",
            "Cross-subject pattern:",
            "No component index should be treated as a global default deletion "
            "unless it appears consistently artifact-like across subjects and is "
            "confirmed by visual inspection.",
            "",
            "Default preprocessing recommendation:",
            "Do not modify preprocess_subject default ica_exclude_components. "
            "Keep ICA as a debugging/manual-cleaning option.",
            "",
            "Uncertainty:",
            "EXG channels are used with broad EOG/EMG-like matching. If the exact "
            "DEAP auxiliary channel mapping is needed, verify it against dataset "
            "documentation before making stronger claims.",
        ]
    )
    if recommended_non_empty:
        lines.append(
            "Some subject-specific candidates passed the conservative automatic "
            "threshold, but they are not recommended as global defaults."
        )
    else:
        lines.append(
            "No subject produced strong enough automatic evidence to recommend "
            "component deletion without manual review."
        )
    return "\n".join(lines)


def main() -> None:
    configure_plot_cache()
    subject_rows = []
    component_rows = []

    for subject_id in SUBJECT_IDS:
        print(f"analyzing ICA artifacts for s{subject_id:02d}...")
        subject_summary, rows, fitted = analyze_subject_components(subject_id)
        save_subject_plots(subject_id, rows, fitted)
        subject_rows.append(subject_summary)
        component_rows.extend(rows)

    save_csv(RESULTS_DIR / "ica_artifact_subject_summary.csv", subject_rows)
    save_csv(RESULTS_DIR / "ica_artifact_component_scores.csv", component_rows)
    REPORT_PATH.write_text(build_report(subject_rows), encoding="utf-8")
    print(f"saved subject summary to {RESULTS_DIR / 'ica_artifact_subject_summary.csv'}")
    print(f"saved component scores to {RESULTS_DIR / 'ica_artifact_component_scores.csv'}")
    print(f"saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
