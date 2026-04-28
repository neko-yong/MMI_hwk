"""Compare our raw-BDF preprocessing with DEAP official preprocessed .dat.

The goal is a reasonableness check, not point-by-point equality. The script
compares structure, scale, waveform snippets, PSD trends, and basic statistics
for one subject by default: s01.
"""

from __future__ import annotations

import csv
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, welch

from src.preprocess import EEG_CHANNEL_COUNT, PROJECT_ROOT, preprocess_subject


SUBJECT_ID = 1
OFFICIAL_DAT_DIR = Path("data_preprocessed_python/data_preprocessed_python")
RESULTS_DIR = Path("results/preprocessing_compare")
SELF_SAMPLING_RATE = 512
OFFICIAL_SAMPLING_RATE = 128
OFFICIAL_BASELINE_SECONDS = 3
TRIALS_TO_PLOT = (0, 10, 20)
CHANNELS_TO_PLOT = (0, 15, 31)
MAX_PLOT_SECONDS = 5


def _configure_matplotlib_cache() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))


def _get_pyplot():
    _configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    return plt


def load_self_preprocessed(subject_id: int = SUBJECT_ID) -> dict:
    """Load our preprocessed baseline-corrected stimulus data."""
    result = preprocess_subject(subject_id=subject_id, enable_ica=False)
    data = np.asarray(result["baseline_corrected_stimulus"], dtype=np.float32)
    return {
        "data": data,
        "sampling_rate": result["sampling_rate"],
        "source": "self raw-BDF preprocessing",
    }


def load_official_dat(
    subject_id: int = SUBJECT_ID,
    dat_dir: Path = OFFICIAL_DAT_DIR,
) -> dict:
    """Load DEAP official preprocessed .dat data and keep first 32 EEG channels."""
    dat_path = PROJECT_ROOT / dat_dir / f"s{subject_id:02d}.dat"

    if not dat_path.exists():
        raise FileNotFoundError(
            f"Official DEAP .dat file not found: {dat_path}. "
            "Expected files such as data_preprocessed_python/.../s01.dat."
        )

    try:
        with dat_path.open("rb") as file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = pickle.load(file, encoding="latin1")
    except Exception as exc:
        raise RuntimeError(f"Failed to read official .dat file: {dat_path}") from exc

    if "data" not in loaded:
        raise KeyError(f"Official .dat file has no 'data' key: {dat_path}")

    data = np.asarray(loaded["data"], dtype=np.float32)
    eeg_data = data[:, :EEG_CHANNEL_COUNT, :]
    baseline_samples = OFFICIAL_BASELINE_SECONDS * OFFICIAL_SAMPLING_RATE

    if eeg_data.shape[-1] > baseline_samples:
        eeg_data = eeg_data[:, :, baseline_samples:]

    return {
        "data": eeg_data,
        "labels": np.asarray(loaded.get("labels", [])),
        "sampling_rate": OFFICIAL_SAMPLING_RATE,
        "source": "DEAP official preprocessed .dat",
        "path": dat_path,
        "raw_shape": data.shape,
        "trim_note": (
            f"Kept first {EEG_CHANNEL_COUNT} EEG channels and removed the first "
            f"{baseline_samples} samples ({OFFICIAL_BASELINE_SECONDS}s) to align "
            "with our baseline-corrected stimulus segment."
        ),
    }


def align_for_comparison(
    self_data: np.ndarray,
    official_data: np.ndarray,
    self_sampling_rate: int = SELF_SAMPLING_RATE,
    official_sampling_rate: int = OFFICIAL_SAMPLING_RATE,
) -> dict:
    """Crop both arrays to shared trial/channel/sample dimensions."""
    aligned_self = self_data

    if self_sampling_rate != official_sampling_rate:
        if self_sampling_rate % official_sampling_rate != 0:
            raise ValueError(
                "Cannot align sampling rates with simple integer downsampling: "
                f"self={self_sampling_rate}, official={official_sampling_rate}."
            )
        downsample_factor = self_sampling_rate // official_sampling_rate
        aligned_self = resample_poly(self_data, up=1, down=downsample_factor, axis=-1)

    n_trials = min(self_data.shape[0], official_data.shape[0])
    n_channels = min(aligned_self.shape[1], official_data.shape[1], EEG_CHANNEL_COUNT)
    n_samples = min(aligned_self.shape[2], official_data.shape[2])

    return {
        "self": aligned_self[:n_trials, :n_channels, :n_samples],
        "official": official_data[:n_trials, :n_channels, :n_samples],
        "n_trials": n_trials,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "sampling_rate": official_sampling_rate,
        "crop_note": (
            "Aligned by removing official baseline, downsampling self data to "
            f"{official_sampling_rate} Hz, then cropping to the shared shape "
            f"({n_trials}, {n_channels}, {n_samples})."
        ),
    }


def _zscore_trace(trace: np.ndarray) -> np.ndarray:
    std = float(np.std(trace))
    if std == 0:
        return trace - float(np.mean(trace))
    return (trace - float(np.mean(trace))) / std


def plot_waveform_compare(
    self_data: np.ndarray,
    official_data: np.ndarray,
    sampling_rate: int,
    subject_id: int = SUBJECT_ID,
) -> list[Path]:
    """Save waveform comparison plots for selected trials/channels."""
    plt = _get_pyplot()
    saved_paths = []
    max_samples = min(int(MAX_PLOT_SECONDS * sampling_rate), self_data.shape[-1])
    time_axis = np.arange(max_samples) / sampling_rate

    for trial in TRIALS_TO_PLOT:
        for channel in CHANNELS_TO_PLOT:
            if trial >= self_data.shape[0] or channel >= self_data.shape[1]:
                continue

            output_path = (
                RESULTS_DIR
                / f"s{subject_id:02d}_trial{trial:02d}_ch{channel:02d}_waveform_compare.png"
            )
            plt.figure(figsize=(9, 4))
            plt.plot(
                time_axis,
                _zscore_trace(self_data[trial, channel, :max_samples]),
                label="self preprocessed",
                linewidth=1,
            )
            plt.plot(
                time_axis,
                _zscore_trace(official_data[trial, channel, :max_samples]),
                label="official dat",
                linewidth=1,
            )
            plt.title(f"Waveform Compare s{subject_id:02d}, trial {trial}, ch {channel}")
            plt.xlabel("Time (s, aligned sample index)")
            plt.ylabel("Z-scored amplitude")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            saved_paths.append(output_path)

    return saved_paths


def plot_psd_compare(
    self_data: np.ndarray,
    official_data: np.ndarray,
    sampling_rate: int,
    subject_id: int = SUBJECT_ID,
) -> list[Path]:
    """Save Welch PSD comparison plots for selected trials/channels."""
    plt = _get_pyplot()
    saved_paths = []

    for trial in TRIALS_TO_PLOT:
        for channel in CHANNELS_TO_PLOT:
            if trial >= self_data.shape[0] or channel >= self_data.shape[1]:
                continue

            self_freqs, self_psd = welch(
                self_data[trial, channel],
                fs=sampling_rate,
                nperseg=min(1024, self_data.shape[-1]),
            )
            official_freqs, official_psd = welch(
                official_data[trial, channel],
                fs=sampling_rate,
                nperseg=min(1024, official_data.shape[-1]),
            )
            output_path = (
                RESULTS_DIR
                / f"s{subject_id:02d}_trial{trial:02d}_ch{channel:02d}_psd_compare.png"
            )

            plt.figure(figsize=(9, 4))
            plt.semilogy(self_freqs, self_psd, label="self preprocessed")
            plt.semilogy(official_freqs, official_psd, label="official dat")
            plt.title(f"PSD Compare s{subject_id:02d}, trial {trial}, ch {channel}")
            plt.xlabel("Frequency (Hz, aligned sampling assumption)")
            plt.ylabel("PSD")
            plt.xlim(0, 60)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            saved_paths.append(output_path)

    return saved_paths


def compute_summary_stats(data: np.ndarray) -> dict:
    """Compute overall and per-channel summary statistics."""
    return {
        "overall_mean": float(np.mean(data)),
        "overall_std": float(np.std(data)),
        "overall_var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "channel_mean": np.mean(data, axis=(0, 2)),
        "channel_std": np.std(data, axis=(0, 2)),
    }


def save_channel_stats(
    self_stats: dict,
    official_stats: dict,
    output_path: Path = RESULTS_DIR / "s01_channel_stats.csv",
) -> Path:
    """Save per-channel mean/std comparison to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "channel",
                "self_mean",
                "self_std",
                "official_mean",
                "official_std",
            ]
        )
        for channel in range(len(self_stats["channel_mean"])):
            writer.writerow(
                [
                    channel,
                    self_stats["channel_mean"][channel],
                    self_stats["channel_std"][channel],
                    official_stats["channel_mean"][channel],
                    official_stats["channel_std"][channel],
                ]
            )

    return output_path


def save_summary(
    self_info: dict,
    official_info: dict,
    aligned: dict,
    self_stats: dict,
    official_stats: dict,
    waveform_paths: list[Path],
    psd_paths: list[Path],
    channel_stats_path: Path,
    output_path: Path = RESULTS_DIR / "s01_compare_summary.txt",
) -> Path:
    """Save a text summary for the report's reasonableness validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "DEAP preprocessing comparison summary",
        "",
        "Purpose:",
        "This comparison checks reasonableness, not point-by-point equality.",
        "",
        "Data structure:",
        f"self_data shape: {self_info['data'].shape}",
        f"official_data shape: {official_info['data'].shape}",
        f"official raw .dat shape: {official_info['raw_shape']}",
        f"official handling: {official_info['trim_note']}",
        f"aligned shape: {aligned['self'].shape}",
        f"n_trials: {aligned['n_trials']}",
        f"n_channels: {aligned['n_channels']}",
        f"n_samples: {aligned['n_samples']}",
        f"self original sampling rate: {self_info['sampling_rate']}",
        f"official sampling rate: {official_info['sampling_rate']}",
        f"comparison sampling rate: {aligned['sampling_rate']}",
        aligned["crop_note"],
        "",
        "Overall statistics on aligned arrays:",
        f"self mean/std/var/min/max: {self_stats['overall_mean']:.6g}, "
        f"{self_stats['overall_std']:.6g}, {self_stats['overall_var']:.6g}, "
        f"{self_stats['min']:.6g}, {self_stats['max']:.6g}",
        f"official mean/std/var/min/max: {official_stats['overall_mean']:.6g}, "
        f"{official_stats['overall_std']:.6g}, {official_stats['overall_var']:.6g}, "
        f"{official_stats['min']:.6g}, {official_stats['max']:.6g}",
        "",
        "Important interpretation notes:",
        "- DEAP official .dat data is already preprocessed and downsampled.",
        "- Our self pipeline starts from raw BDF and uses the local settings in preprocess.py.",
        "- Differences may come from filter parameters, baseline correction details, "
        "time-window definitions, reference strategy, and artifact-removal strategy.",
        "- When sampling rate, window length, or scale differ, plots use aligned/cropped "
        "arrays and z-scored waveform snippets for visual trend comparison.",
        "",
        f"waveform plots: {len(waveform_paths)} files in {RESULTS_DIR}",
        f"PSD plots: {len(psd_paths)} files in {RESULTS_DIR}",
        f"channel stats CSV: {channel_stats_path}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    """Run s01 self-vs-official preprocessing comparison."""
    self_info = load_self_preprocessed(SUBJECT_ID)
    official_info = load_official_dat(SUBJECT_ID)

    print(f"loaded self preprocessed data: {self_info['data'].shape}")
    print(f"loaded official dat data: {official_info['data'].shape}")

    aligned = align_for_comparison(
        self_info["data"],
        official_info["data"],
        self_sampling_rate=int(self_info["sampling_rate"]),
        official_sampling_rate=official_info["sampling_rate"],
    )
    print(f"aligned shapes for comparison: {aligned['self'].shape}")

    self_stats = compute_summary_stats(aligned["self"])
    official_stats = compute_summary_stats(aligned["official"])
    waveform_paths = plot_waveform_compare(
        aligned["self"],
        aligned["official"],
        sampling_rate=aligned["sampling_rate"],
        subject_id=SUBJECT_ID,
    )
    psd_paths = plot_psd_compare(
        aligned["self"],
        aligned["official"],
        sampling_rate=aligned["sampling_rate"],
        subject_id=SUBJECT_ID,
    )
    channel_stats_path = save_channel_stats(self_stats, official_stats)
    summary_path = save_summary(
        self_info,
        official_info,
        aligned,
        self_stats,
        official_stats,
        waveform_paths,
        psd_paths,
        channel_stats_path,
    )

    print(f"waveform plots saved to: {RESULTS_DIR}")
    print(f"psd plots saved to: {RESULTS_DIR}")
    print(f"summary saved to: {summary_path}")
    print(f"channel stats saved to: {channel_stats_path}")


if __name__ == "__main__":
    main()
