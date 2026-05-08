"""MNE official-like EEG preprocessing validation pipeline.

This standalone script keeps src/preprocess.py unchanged. It builds an
MNE-based enhanced preprocessing route for comparison with DEAP official
preprocessed .dat files:

raw BDF -> channel typing/montage/reference -> output filtering -> MNE ICA
artifact review -> trial cutting -> baseline correction -> 512 Hz and 128 Hz
comparison summaries.
"""

from __future__ import annotations

import csv
import os
import pickle
import warnings
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, welch

from src.preprocess import (
    DEFAULT_METADATA_DIR,
    DEFAULT_ORIGINAL_DIR,
    EEG_CHANNEL_COUNT,
    PROJECT_ROOT,
    TARGET_BASELINE_SAMPLES,
    TARGET_STIMULUS_SAMPLES,
    get_trial_boundaries_from_bdf,
    load_participant_ratings,
)


SUBJECT_IDS = (1, 10, 24)
RESULTS_DIR = PROJECT_ROOT / "results/official_like_handoff"
SUMMARY_CSV = RESULTS_DIR / "official_like_validation_summary.csv"
REPORT_PATH = RESULTS_DIR / "official_like_validation_report.txt"

OFFICIAL_DAT_DIRS = (
    PROJECT_ROOT / "data_preprocessed_python",
    PROJECT_ROOT / "data_preprocessed_python/data_preprocessed_python",
)

EEG_OUTPUT_LOW_HZ = 4.0
EEG_OUTPUT_HIGH_HZ = 45.0
NOTCH_HZ = 50.0
ICA_FIT_LOW_HZ = 1.0
ICA_FIT_HIGH_HZ = 45.0
ICA_N_COMPONENTS = 16
ICA_RANDOM_STATE = 42
EOG_SCORE_THRESHOLD = 0.35
MAX_EXCLUDE_COMPONENTS = 2
SAMPLING_RATE = 512
OFFICIAL_SAMPLING_RATE = 128
OFFICIAL_BASELINE_SAMPLES = 384
OFFICIAL_LIKE_CHANNEL_COUNT = 40
OFFICIAL_LIKE_BASELINE_SAMPLES_512 = 3 * SAMPLING_RATE
OFFICIAL_LIKE_TOTAL_SAMPLES_128 = 8064
LABEL_COLUMNS = ("Valence", "Arousal", "Dominance", "Liking")

EXG_EOG_NAMES = ("EXG1", "EXG2", "EXG3", "EXG4")
EXG_EMG_NAMES = ("EXG5", "EXG6", "EXG7", "EXG8")


def configure_environment() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_HOME", str(RESULTS_DIR / "mne_home"))


def import_mne():
    try:
        import mne
        from mne.preprocessing import ICA
    except ImportError:
        print("MNE-Python is not installed.")
        print("pip install mne")
        return None, None
    return mne, ICA


def subject_label(subject_id: int) -> str:
    return f"s{subject_id:02d}"


def save_figure(fig, output_path: Path, notes: list[str]) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(fig, list):
            for index, item in enumerate(fig):
                path = output_path.with_name(f"{output_path.stem}_{index:02d}{output_path.suffix}")
                item.savefig(path, dpi=150)
                plt.close(item)
        else:
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"could not save {output_path.name}: {exc}")


def set_channel_types(raw, subject: str, notes: list[str]) -> tuple[list[str], list[str]]:
    ch_names = raw.ch_names
    channel_types = {name: "misc" for name in ch_names}
    for name in ch_names[:EEG_CHANNEL_COUNT]:
        channel_types[name] = "eeg"

    eog_names = [name for name in ch_names if name.upper() in EXG_EOG_NAMES]
    emg_names = [name for name in ch_names if name.upper() in EXG_EMG_NAMES]
    for name in eog_names:
        channel_types[name] = "eog"
    for name in emg_names:
        channel_types[name] = "misc"

    stim_candidates = [
        name
        for name in ch_names
        if name.strip().lower() == "status"
        or any(key in name.strip().lower() for key in ("stim", "trigger", "event"))
    ]
    if stim_candidates:
        channel_types[stim_candidates[0]] = "stim"
        notes.append(f"{subject}: stim channel set to {stim_candidates[0]!r}")
    else:
        # Some DEAP copies use an empty final channel as the event channel.
        last_name = ch_names[-1]
        channel_types[last_name] = "stim" if last_name else "misc"
        notes.append(f"{subject}: no named Status channel; final channel {last_name!r} treated as event-like/misc")

    raw.set_channel_types(channel_types, verbose="ERROR")
    print(f"{subject}: EOG channels = {eog_names or 'none'}")
    print(f"{subject}: EMG/misc channels = {emg_names or 'none'}")
    return eog_names, emg_names


def set_montage_and_reference(raw, mne, subject: str, notes: list[str]) -> None:
    try:
        montage = mne.channels.make_standard_montage("biosemi32")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
        notes.append(f"{subject}: montage set to biosemi32")
    except Exception as exc:
        warning = f"{subject}: warning, biosemi32 montage failed: {exc}"
        print(warning)
        notes.append(warning)

    try:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        notes.append(f"{subject}: EEG average reference applied")
    except Exception as exc:
        notes.append(f"{subject}: average reference failed: {exc}")


def load_raw_subject(subject_id: int, mne) -> tuple:
    subject = subject_label(subject_id)
    notes: list[str] = []
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"{subject}.bdf"
    if not bdf_path.exists():
        raise FileNotFoundError(f"{bdf_path} does not exist")

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    print(f"{subject}: sfreq = {raw.info['sfreq']}")
    print(f"{subject}: channel count = {len(raw.ch_names)}")
    print(f"{subject}: channel names = {raw.ch_names}")
    notes.append(f"{subject}: sfreq={raw.info['sfreq']}")
    notes.append(f"{subject}: channel count={len(raw.ch_names)}")
    notes.append(f"{subject}: channel names={raw.ch_names}")

    eog_names, emg_names = set_channel_types(raw, subject, notes)
    set_montage_and_reference(raw, mne, subject, notes)
    return raw, eog_names, emg_names, notes


def filter_output_raw(raw, mne, subject: str, notes: list[str]):
    filtered = raw.copy()
    picks = mne.pick_types(filtered.info, eeg=True, exclude="bads")
    filtered.filter(EEG_OUTPUT_LOW_HZ, EEG_OUTPUT_HIGH_HZ, picks=picks, verbose="ERROR")
    filtered.notch_filter(NOTCH_HZ, picks=picks, verbose="ERROR")
    notes.append(f"{subject}: output EEG filtered {EEG_OUTPUT_LOW_HZ}-{EEG_OUTPUT_HIGH_HZ} Hz and notch {NOTCH_HZ} Hz")
    return filtered


def fit_ica(raw, mne, ICA, subject: str, notes: list[str]):
    raw_ica = raw.copy()
    picks = mne.pick_types(raw_ica.info, eeg=True, exclude="bads")
    raw_ica.filter(ICA_FIT_LOW_HZ, ICA_FIT_HIGH_HZ, picks=picks, verbose="ERROR")
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        method="fastica",
        max_iter="auto",
    )
    ica.fit(raw_ica, picks=picks, verbose="ERROR")
    notes.append(f"{subject}: ICA fitted on EEG picks={len(picks)}, filter={ICA_FIT_LOW_HZ}-{ICA_FIT_HIGH_HZ} Hz")
    return ica, raw_ica


def ranked_eog_scores(score_by_channel: dict[str, np.ndarray]) -> list[tuple[int, float, str]]:
    best: dict[int, tuple[float, str]] = {}
    for channel, scores in score_by_channel.items():
        for component, score in enumerate(np.ravel(scores)):
            value = abs(float(score))
            if component not in best or value > best[component][0]:
                best[component] = (value, channel)
    return sorted(
        [(component, value, channel) for component, (value, channel) in best.items()],
        key=lambda item: item[1],
        reverse=True,
    )


def find_eog_candidates(ica, raw, eog_names: list[str], subject: str, notes: list[str]) -> tuple[list[int], dict[str, np.ndarray], list[tuple[int, float, str]]]:
    detected: set[int] = set()
    score_by_channel: dict[str, np.ndarray] = {}
    for eog_name in eog_names:
        try:
            inds, scores = ica.find_bads_eog(raw, ch_name=eog_name, verbose="ERROR")
            detected.update(int(index) for index in inds)
            score_by_channel[eog_name] = np.asarray(scores, dtype=float)
            notes.append(f"{subject}: EOG scoring {eog_name}, inds={inds}")
        except Exception as exc:
            notes.append(f"{subject}: EOG scoring failed for {eog_name}: {exc}")

    ranked = ranked_eog_scores(score_by_channel)
    recommended = [
        component
        for component, score, _ in ranked
        if component in detected and score >= EOG_SCORE_THRESHOLD
    ][:MAX_EXCLUDE_COMPONENTS]
    notes.append(f"{subject}: recommended subject-specific ICA excludes={recommended}")
    return recommended, score_by_channel, ranked


def apply_ica_cleaning(ica, raw_filtered, excludes: list[int]):
    cleaned = raw_filtered.copy()
    ica.apply(cleaned, exclude=excludes, verbose="ERROR")
    return cleaned


def fixed_length(segment: np.ndarray, target_samples: int, label: str, notes: list[str]) -> np.ndarray:
    if segment.shape[-1] >= target_samples:
        return segment[:, :target_samples]
    notes.append(f"{label}: segment short ({segment.shape[-1]} < {target_samples}); zero-padded")
    padded = np.zeros((segment.shape[0], target_samples), dtype=segment.dtype)
    padded[:, : segment.shape[-1]] = segment
    return padded


def extract_baseline_corrected_trials(raw_variant, bdf_path: Path, notes: list[str]) -> np.ndarray:
    boundaries = get_trial_boundaries_from_bdf(bdf_path)
    eeg_names = raw_variant.copy().pick("eeg").ch_names[:EEG_CHANNEL_COUNT]
    baselines = []
    stimuli = []
    for boundary in boundaries:
        baseline = raw_variant.get_data(
            picks=eeg_names,
            start=boundary["baseline_start_sample"],
            stop=boundary["stimulus_start_sample"],
        ) * 1e6
        stimulus = raw_variant.get_data(
            picks=eeg_names,
            start=boundary["stimulus_start_sample"],
            stop=boundary["stimulus_end_sample"],
        ) * 1e6
        baselines.append(
            fixed_length(baseline.astype("float32"), TARGET_BASELINE_SAMPLES, "baseline", notes)
        )
        stimuli.append(
            fixed_length(stimulus.astype("float32"), TARGET_STIMULUS_SAMPLES, "stimulus", notes)
        )

    baseline_array = np.asarray(baselines, dtype="float32")
    stimulus_array = np.asarray(stimuli, dtype="float32")
    corrected = stimulus_array - baseline_array.mean(axis=-1, keepdims=True)
    return corrected.astype("float32")


def downsample_to_128(data: np.ndarray) -> np.ndarray:
    return resample_poly(data, up=1, down=4, axis=-1).astype("float32")


def load_official_dat_dict(subject_id: int) -> dict | None:
    """Load DEAP official preprocessed Python .dat for labels/comparison."""
    subject = subject_label(subject_id)
    for directory in OFFICIAL_DAT_DIRS:
        path = directory / f"{subject}.dat"
        if path.exists():
            with path.open("rb") as file:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loaded = pickle.load(file, encoding="latin1")
            return loaded
    return None


def load_official_dat(subject_id: int) -> np.ndarray | None:
    loaded = load_official_dat_dict(subject_id)
    if loaded is not None:
        data = np.asarray(loaded["data"], dtype="float32")
        return data[:, :EEG_CHANNEL_COUNT, OFFICIAL_BASELINE_SAMPLES:]
    return None


def load_labels(subject_id: int) -> tuple[np.ndarray, str]:
    """Return labels shaped like official DEAP labels: (40, 4)."""
    official = load_official_dat_dict(subject_id)
    if official is not None and "labels" in official:
        return np.asarray(official["labels"], dtype="float32"), "official_dat"

    ratings = load_participant_ratings(subject_id, DEFAULT_METADATA_DIR)
    labels = [
        [float(row[column]) for column in LABEL_COLUMNS]
        for row in sorted(ratings, key=lambda item: int(item["Trial"]))
    ]
    return np.asarray(labels, dtype="float32"), "participant_ratings.csv"


def official_like_channel_names(raw) -> list[str]:
    """Use official-style 40 channels: 32 EEG + EXG1-EXG8."""
    names = list(raw.ch_names[:EEG_CHANNEL_COUNT])
    for exg_name in (*EXG_EOG_NAMES, *EXG_EMG_NAMES):
        if exg_name in raw.ch_names:
            names.append(exg_name)
    if len(names) != OFFICIAL_LIKE_CHANNEL_COUNT:
        raise ValueError(
            "Could not build 40-channel official-like channel list. "
            f"Got {len(names)} channels: {names}"
        )
    return names


def extract_official_like_trials(raw_variant, bdf_path: Path, channel_names: list[str], notes: list[str]) -> np.ndarray:
    """Return official-style data (40 trials, 40 channels, 8064 samples).

    The 63s window is the last 3s of baseline plus 60s stimulus, downsampled
    from 512 Hz to 128 Hz.
    """
    boundaries = get_trial_boundaries_from_bdf(bdf_path)
    trial_data = []
    for boundary in boundaries:
        baseline_start = max(
            boundary["stimulus_start_sample"] - OFFICIAL_LIKE_BASELINE_SAMPLES_512,
            boundary["baseline_start_sample"],
        )
        segment = raw_variant.get_data(
            picks=channel_names,
            start=baseline_start,
            stop=boundary["stimulus_end_sample"],
        ) * 1e6
        target_512_samples = OFFICIAL_LIKE_BASELINE_SAMPLES_512 + TARGET_STIMULUS_SAMPLES
        fixed = fixed_length(
            segment.astype("float32"),
            target_512_samples,
            "official_like_baseline_plus_stimulus",
            notes,
        )
        downsampled = downsample_to_128(fixed)
        trial_data.append(
            fixed_length(
                downsampled,
                OFFICIAL_LIKE_TOTAL_SAMPLES_128,
                "official_like_128hz",
                notes,
            )
        )
    return np.asarray(trial_data, dtype="float32")


def baseline_correct_eeg_from_official_like(data: np.ndarray) -> np.ndarray:
    """Return modeling-friendly (40, 32, 7680) baseline-corrected EEG."""
    eeg = data[:, :EEG_CHANNEL_COUNT, :]
    baseline = eeg[:, :, :OFFICIAL_BASELINE_SAMPLES]
    stimulus = eeg[:, :, OFFICIAL_BASELINE_SAMPLES:]
    return (stimulus - baseline.mean(axis=-1, keepdims=True)).astype("float32")


def overall_stats(data: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> dict:
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "high_freq_ratio_30_45": high_freq_ratio(data, sampling_rate=sampling_rate),
    }


def high_freq_ratio(data: np.ndarray, sampling_rate: int = SAMPLING_RATE) -> float:
    flattened = data.reshape(-1)
    freqs, psd = welch(flattened, fs=sampling_rate, nperseg=min(4096, flattened.shape[0]))
    high_mask = (freqs >= 30) & (freqs <= 45)
    total_mask = (freqs >= 4) & (freqs <= 45)
    total_power = float(psd[total_mask].sum())
    return float(psd[high_mask].sum()) / total_power if total_power > 0 else 0.0


def pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def flatten_trials(data: np.ndarray) -> np.ndarray:
    return data.transpose(0, 2, 1).reshape(-1, data.shape[1])


def zscore_columns(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (data - mean) / std


def extract_eog_trials(raw, bdf_path: Path, eog_names: list[str], notes: list[str]) -> np.ndarray | None:
    if not eog_names:
        return None
    boundaries = get_trial_boundaries_from_bdf(bdf_path)
    eog_trials = []
    for boundary in boundaries:
        eog = raw.get_data(
            picks=eog_names,
            start=boundary["stimulus_start_sample"],
            stop=boundary["stimulus_end_sample"],
        ) * 1e6
        fixed_eog = fixed_length(eog.astype("float32"), TARGET_STIMULUS_SAMPLES, "eog", notes)
        eog_trials.append(downsample_to_128(fixed_eog))
    return np.asarray(eog_trials, dtype="float32")


def eeg_eog_corr_summary(data: np.ndarray, eog_data: np.ndarray | None) -> tuple[float | None, float | None]:
    if eog_data is None or eog_data.size == 0:
        return None, None
    eeg_flat = zscore_columns(flatten_trials(data))
    eog_flat = zscore_columns(flatten_trials(eog_data))
    corr = np.corrcoef(eeg_flat.T, eog_flat.T)[: eeg_flat.shape[1], eeg_flat.shape[1] :]
    channel_max = np.nan_to_num(np.abs(corr), nan=0.0).max(axis=1)
    return float(channel_max.mean()), float(channel_max.max())


def official_alignment_stats(data_128: np.ndarray, official: np.ndarray | None) -> dict:
    if official is None:
        return {
            "official_available": False,
            "official_std": "",
            "std_ratio_to_official": "",
            "mean_abs_diff_z": "",
        }
    n_trials = min(data_128.shape[0], official.shape[0])
    n_channels = min(data_128.shape[1], official.shape[1])
    n_samples = min(data_128.shape[2], official.shape[2])
    aligned = data_128[:n_trials, :n_channels, :n_samples]
    official_aligned = official[:n_trials, :n_channels, :n_samples]
    aligned_z = (aligned - aligned.mean()) / (aligned.std() or 1.0)
    official_z = (official_aligned - official_aligned.mean()) / (official_aligned.std() or 1.0)
    official_std = float(official_aligned.std())
    return {
        "official_available": True,
        "official_shape": str(tuple(official.shape)),
        "aligned_shape": str(tuple(aligned.shape)),
        "official_std": official_std,
        "std_ratio_to_official": float(aligned.std() / official_std) if official_std else "",
        "mean_abs_diff_z": float(np.mean(np.abs(aligned_z - official_z))),
    }


def save_waveform_plot(subject: str, no_ica: np.ndarray, cleaned: np.ndarray, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / f"{subject}_waveform_no_ica_vs_ica_cleaned.png"
    max_samples = min(5 * OFFICIAL_SAMPLING_RATE, no_ica.shape[-1])
    time_axis = np.arange(max_samples) / OFFICIAL_SAMPLING_RATE
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, no_ica[0, 0, :max_samples], label="no_ica", linewidth=1.0)
    ax.plot(time_axis, cleaned[0, 0, :max_samples], label="ica_cleaned", linewidth=1.0)
    ax.set_title(f"{subject} waveform, trial 0 channel 0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_psd_plot(subject: str, no_ica: np.ndarray, cleaned: np.ndarray, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / f"{subject}_psd_no_ica_vs_ica_cleaned.png"
    freqs_no, psd_no = welch(no_ica[0, 0], fs=OFFICIAL_SAMPLING_RATE, nperseg=1024)
    freqs_clean, psd_clean = welch(cleaned[0, 0], fs=OFFICIAL_SAMPLING_RATE, nperseg=1024)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs_no, psd_no, label="no_ica")
    ax.semilogy(freqs_clean, psd_clean, label="ica_cleaned")
    ax.set_title(f"{subject} PSD, trial 0 channel 0")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_xlim(0, 60)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_eog_corr_plot(subject: str, summary: dict, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    path = output_dir / f"{subject}_eog_corr_no_ica_vs_ica_cleaned.png"
    labels = ["mean max corr", "max corr"]
    no_values = [summary["eog_corr_mean_no_ica"] or 0.0, summary["eog_corr_max_no_ica"] or 0.0]
    cleaned_values = [summary["eog_corr_mean_ica_cleaned"] or 0.0, summary["eog_corr_max_ica_cleaned"] or 0.0]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, no_values, width=0.4, label="no_ica")
    ax.bar(x + 0.2, cleaned_values, width=0.4, label="ica_cleaned")
    ax.set_title(f"{subject} EEG vs EOG-like correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_subject_summary(
    subject_id: int,
    excludes: list[int],
    ranked: list[tuple[int, float, str]],
    no_ica: np.ndarray,
    cleaned: np.ndarray,
    official_like_no_ica: np.ndarray,
    official_like_cleaned: np.ndarray,
    labels: np.ndarray,
    official: np.ndarray | None,
    eog_corrs: tuple,
    notes: list[str],
) -> dict:
    subject = subject_label(subject_id)
    no_stats = overall_stats(no_ica, sampling_rate=OFFICIAL_SAMPLING_RATE)
    cleaned_stats = overall_stats(cleaned, sampling_rate=OFFICIAL_SAMPLING_RATE)
    official_no = official_alignment_stats(no_ica, official)
    official_clean = official_alignment_stats(cleaned, official)
    no_eog_mean, no_eog_max, cleaned_eog_mean, cleaned_eog_max = eog_corrs
    std_drop = -pct_change(no_stats["std"], cleaned_stats["std"])
    eog_drop = "" if no_eog_mean is None else -pct_change(no_eog_mean, cleaned_eog_mean)
    overcleaning = std_drop > 30.0 or abs(pct_change(no_stats["var"], cleaned_stats["var"])) > 50.0
    eog_improved = eog_drop != "" and eog_drop > 10.0
    hf_not_worse = pct_change(no_stats["high_freq_ratio_30_45"], cleaned_stats["high_freq_ratio_30_45"]) <= 10.0
    official_similarity_better = False
    if official_no["official_available"]:
        no_z = official_no["mean_abs_diff_z"]
        clean_z = official_clean["mean_abs_diff_z"]
        no_ratio = official_no["std_ratio_to_official"]
        clean_ratio = official_clean["std_ratio_to_official"]
        official_similarity_better = (
            clean_z <= no_z
            and abs(1.0 - clean_ratio) <= abs(1.0 - no_ratio)
        )

    if excludes and eog_improved and not overcleaning and hf_not_worse and official_similarity_better:
        recommendation = "optional enhanced output is supported by EOG and official-alignment metrics"
        selected_variant = "ica_cleaned"
    elif excludes and eog_improved:
        recommendation = "EOG influence decreased, but official-like metrics do not consistently improve; keep as manual-review evidence"
        selected_variant = "no_ica"
    elif excludes:
        recommendation = "candidate components found, but cleaning benefit is not strong enough for default replacement"
        selected_variant = "no_ica"
    else:
        recommendation = "no ICA deletion; keep no_ica output"
        selected_variant = "no_ica"

    return {
        "subject_id": subject,
        "official_like_data_shape_no_ica": str(tuple(official_like_no_ica.shape)),
        "official_like_data_shape_ica_cleaned": str(tuple(official_like_cleaned.shape)),
        "labels_shape": str(tuple(labels.shape)),
        "selected_variant_auto": selected_variant,
        "baseline_corrected_eeg_shape_no_ica": str(tuple(no_ica.shape)),
        "baseline_corrected_eeg_shape_ica_cleaned": str(tuple(cleaned.shape)),
        "eog_candidate_components": str(excludes),
        "top_eog_scores": "; ".join(f"C{c:02d}={s:.3f}({ch})" for c, s, ch in ranked[:5]),
        "std_no_ica": no_stats["std"],
        "std_ica_cleaned": cleaned_stats["std"],
        "std_change_pct": pct_change(no_stats["std"], cleaned_stats["std"]),
        "var_no_ica": no_stats["var"],
        "var_ica_cleaned": cleaned_stats["var"],
        "var_change_pct": pct_change(no_stats["var"], cleaned_stats["var"]),
        "hf_ratio_no_ica": no_stats["high_freq_ratio_30_45"],
        "hf_ratio_ica_cleaned": cleaned_stats["high_freq_ratio_30_45"],
        "hf_ratio_change_pct": pct_change(no_stats["high_freq_ratio_30_45"], cleaned_stats["high_freq_ratio_30_45"]),
        "eog_corr_mean_no_ica": "" if no_eog_mean is None else no_eog_mean,
        "eog_corr_mean_ica_cleaned": "" if cleaned_eog_mean is None else cleaned_eog_mean,
        "eog_corr_mean_change_pct": "" if no_eog_mean is None else pct_change(no_eog_mean, cleaned_eog_mean),
        "eog_corr_max_no_ica": "" if no_eog_max is None else no_eog_max,
        "eog_corr_max_ica_cleaned": "" if cleaned_eog_max is None else cleaned_eog_max,
        "official_available": official_no["official_available"],
        "official_std": official_no["official_std"],
        "std_ratio_to_official_no_ica": official_no["std_ratio_to_official"],
        "std_ratio_to_official_ica_cleaned": official_clean["std_ratio_to_official"],
        "mean_abs_diff_z_no_ica": official_no["mean_abs_diff_z"],
        "mean_abs_diff_z_ica_cleaned": official_clean["mean_abs_diff_z"],
        "overcleaning_risk": overcleaning,
        "hf_not_worse": hf_not_worse,
        "official_similarity_better": official_similarity_better,
        "recommendation": recommendation,
        "notes": " | ".join(notes[-8:]),
    }


def _prepare_subject_official_like(subject_id: int, apply_ica: str | bool = "auto") -> tuple[dict, dict, dict]:
    """Build official-style output dict plus validation summary/debug data."""
    subject = subject_label(subject_id)
    mne, ICA = import_mne()
    if mne is None:
        raise ImportError("MNE-Python is required. Install with: pip install mne")

    raw, eog_names, _, notes = load_raw_subject(subject_id, mne)
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"{subject}.bdf"

    raw_filtered = filter_output_raw(raw, mne, subject, notes)
    ica, raw_ica = fit_ica(raw, mne, ICA, subject, notes)
    excludes, score_by_channel, ranked = find_eog_candidates(ica, raw, eog_names, subject, notes)
    raw_cleaned = apply_ica_cleaning(ica, raw_filtered, excludes) if excludes else raw_filtered.copy()

    channel_names = official_like_channel_names(raw)
    official_like_no_ica = extract_official_like_trials(raw_filtered, bdf_path, channel_names, notes)
    official_like_cleaned = extract_official_like_trials(raw_cleaned, bdf_path, channel_names, notes)
    no_ica = baseline_correct_eeg_from_official_like(official_like_no_ica)
    cleaned = baseline_correct_eeg_from_official_like(official_like_cleaned)
    eog_data = extract_eog_trials(raw_filtered, bdf_path, eog_names, notes)
    eog_corrs = (
        *eeg_eog_corr_summary(no_ica, eog_data),
        *eeg_eog_corr_summary(cleaned, eog_data),
    )
    official = load_official_dat(subject_id)
    labels, label_source = load_labels(subject_id)
    summary = build_subject_summary(
        subject_id,
        excludes,
        ranked,
        no_ica,
        cleaned,
        official_like_no_ica,
        official_like_cleaned,
        labels,
        official,
        eog_corrs,
        notes,
    )

    if apply_ica is True:
        selected_variant = "ica_cleaned"
    elif apply_ica is False:
        selected_variant = "no_ica"
    else:
        selected_variant = summary["selected_variant_auto"]

    if selected_variant == "ica_cleaned":
        selected_data = official_like_cleaned
        selected_baseline_corrected = cleaned
    else:
        selected_data = official_like_no_ica
        selected_baseline_corrected = no_ica

    result = {
        "data": selected_data.astype("float32"),
        "labels": labels.astype("float32"),
        "subject_id": subject,
        "sampling_rate": OFFICIAL_SAMPLING_RATE,
        "channel_names": channel_names,
        "baseline_corrected_eeg": selected_baseline_corrected.astype("float32"),
        "preprocessing_info": {
            "format": "official_like",
            "data_shape": tuple(selected_data.shape),
            "labels_shape": tuple(labels.shape),
            "data_units": "microvolts",
            "data_window": "3s baseline + 60s stimulus",
            "baseline_samples_128hz": OFFICIAL_BASELINE_SAMPLES,
            "stimulus_samples_128hz": OFFICIAL_LIKE_TOTAL_SAMPLES_128 - OFFICIAL_BASELINE_SAMPLES,
            "baseline_corrected_eeg_shape": tuple(selected_baseline_corrected.shape),
            "label_columns": LABEL_COLUMNS,
            "label_source": label_source,
            "selected_variant": selected_variant,
            "apply_ica_parameter": apply_ica,
            "eog_candidate_components": excludes,
            "top_eog_scores": summary["top_eog_scores"],
            "recommendation": summary["recommendation"],
            "eeg_processing": "average reference, 4-45 Hz bandpass, 50 Hz notch, optional MNE ICA",
            "auxiliary_channels": "EXG1-EXG8 kept in channels 32-39; EEG-focused preprocessing and ICA only apply to first 32 EEG channels",
            "notes": notes,
        },
    }
    debug = {
        "ica": ica,
        "score_by_channel": score_by_channel,
        "ranked": ranked,
        "no_ica": no_ica,
        "cleaned": cleaned,
        "raw_filtered": raw_filtered,
        "raw_cleaned": raw_cleaned,
    }
    return result, summary, debug


def preprocess_subject_to_official_like(subject_id: int, apply_ica: str | bool = "auto") -> dict:
    """Return a DEAP official .dat-like dict without saving large arrays.

    data shape: (40, 40, 8064)
    labels shape: (40, 4)
    baseline_corrected_eeg shape: (40, 32, 7680)
    """
    configure_environment()
    result, _, _ = _prepare_subject_official_like(subject_id, apply_ica=apply_ica)
    return result


def analyze_subject(subject_id: int, make_plots: bool = True) -> dict:
    subject = subject_label(subject_id)
    output_dir = RESULTS_DIR / subject
    output_dir.mkdir(parents=True, exist_ok=True)
    result, summary, debug = _prepare_subject_official_like(subject_id, apply_ica="auto")
    no_ica = debug["no_ica"]
    cleaned = debug["cleaned"]
    score_by_channel = debug["score_by_channel"]
    ranked = debug["ranked"]
    ica = debug["ica"]
    notes = result["preprocessing_info"]["notes"]

    if not make_plots:
        return summary

    if score_by_channel:
        try:
            fig = ica.plot_scores(score_by_channel[ranked[0][2]], show=False)
            save_figure(fig, output_dir / f"{subject}_mne_eog_scores.png", notes)
        except Exception as exc:
            notes.append(f"{subject}: score plot failed: {exc}")
    save_waveform_plot(subject, no_ica, cleaned, output_dir)
    save_psd_plot(subject, no_ica, cleaned, output_dir)
    save_eog_corr_plot(subject, summary, output_dir)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(value, digits: int = 2) -> str:
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_report(rows: list[dict]) -> str:
    lines = [
        "MNE official-like preprocessing report",
        "",
        "Purpose:",
        "Build an enhanced MNE-based preprocessing route to compare against the stable baseline and DEAP official .dat results.",
        "",
        "Important:",
        "src/preprocess.py defaults are unchanged. ICA deletion is subject-specific and evidence-based; component IDs are not global rules.",
        "",
        "Pipeline:",
        "BDF -> channel typing -> biosemi32 montage -> average reference -> output 4-45 Hz + 50 Hz notch -> ICA fit on 1-45 Hz EEG copy -> EOG candidate detection -> trial cutting -> baseline correction -> 512 Hz and 128 Hz summaries.",
        "",
        "Subject summaries:",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                f"{row['subject_id']}:",
                f"official-like data no_ica/cleaned: {row['official_like_data_shape_no_ica']} / {row['official_like_data_shape_ica_cleaned']}",
                f"labels shape: {row['labels_shape']}",
                f"auto-selected variant: {row['selected_variant_auto']}",
                f"baseline-corrected EEG no_ica/cleaned: {row['baseline_corrected_eeg_shape_no_ica']} / {row['baseline_corrected_eeg_shape_ica_cleaned']}",
                f"EOG candidates: {row['eog_candidate_components']}",
                f"top EOG scores: {row['top_eog_scores']}",
                f"std change after ICA: {fmt_num(row['std_change_pct'])}%",
                f"var change after ICA: {fmt_num(row['var_change_pct'])}%",
                f"30-45 Hz ratio change after ICA: {fmt_num(row['hf_ratio_change_pct'])}%",
                f"EOG mean corr change: {row['eog_corr_mean_change_pct']}",
                f"official std ratio no_ica/cleaned: {row['std_ratio_to_official_no_ica']} / {row['std_ratio_to_official_ica_cleaned']}",
                f"z-scored mean abs diff no_ica/cleaned: {row['mean_abs_diff_z_no_ica']} / {row['mean_abs_diff_z_ica_cleaned']}",
                f"overcleaning risk: {row['overcleaning_risk']}",
                f"high-frequency not worse: {row['hf_not_worse']}",
                f"official similarity better: {row['official_similarity_better']}",
                f"recommendation: {row['recommendation']}",
            ]
        )
    lines.extend(
        [
            "",
            "Final conservative conclusion:",
            "This official-like MNE route is suitable as an enhanced validation output. It can support subject-specific ICA review, but should not automatically replace the stable preprocess.py baseline until benefits are consistent across more subjects and downstream modeling.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DEAP official .dat-like preprocessing summaries.",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Run lightweight summary for s01-s32 without plots.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=None,
        help="Specific subject numbers to process, e.g. --subjects 1 10 24.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip validation figures.",
    )
    return parser.parse_args()


def main() -> None:
    configure_environment()
    args = parse_args()
    mne, ICA = import_mne()
    if mne is None:
        return

    if args.all_subjects:
        subject_ids = tuple(range(1, 33))
        make_plots = False
    elif args.subjects:
        subject_ids = tuple(args.subjects)
        make_plots = not args.no_plots
    else:
        subject_ids = SUBJECT_IDS
        make_plots = not args.no_plots

    rows = []
    for subject_id in subject_ids:
        subject = subject_label(subject_id)
        print(f"running official-like MNE preprocessing for {subject}...")
        try:
            rows.append(analyze_subject(subject_id, make_plots=make_plots))
        except Exception as exc:
            print(f"{subject}: failed: {exc}")
            rows.append(
                {
                    "subject_id": subject,
                    "official_like_data_shape_no_ica": "failed",
                    "official_like_data_shape_ica_cleaned": "failed",
                    "labels_shape": "failed",
                    "selected_variant_auto": "failed",
                    "baseline_corrected_eeg_shape_no_ica": "failed",
                    "baseline_corrected_eeg_shape_ica_cleaned": "failed",
                    "eog_candidate_components": "failed",
                    "top_eog_scores": "",
                    "std_no_ica": "",
                    "std_ica_cleaned": "",
                    "std_change_pct": "",
                    "var_no_ica": "",
                    "var_ica_cleaned": "",
                    "var_change_pct": "",
                    "hf_ratio_no_ica": "",
                    "hf_ratio_ica_cleaned": "",
                    "hf_ratio_change_pct": "",
                    "eog_corr_mean_no_ica": "",
                    "eog_corr_mean_ica_cleaned": "",
                    "eog_corr_mean_change_pct": "",
                    "eog_corr_max_no_ica": "",
                    "eog_corr_max_ica_cleaned": "",
                    "official_available": "",
                    "official_std": "",
                    "std_ratio_to_official_no_ica": "",
                    "std_ratio_to_official_ica_cleaned": "",
                    "mean_abs_diff_z_no_ica": "",
                    "mean_abs_diff_z_ica_cleaned": "",
                    "overcleaning_risk": "",
                    "hf_not_worse": "",
                    "official_similarity_better": "",
                    "recommendation": f"failed: {exc}",
                    "notes": str(exc),
                }
            )

    write_csv(SUMMARY_CSV, rows)
    REPORT_PATH.write_text(build_report(rows), encoding="utf-8")
    print(f"saved summary to {SUMMARY_CSV}")
    print(f"saved report to {REPORT_PATH}")
    print(f"saved figures under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
