"""MNE-based ICA review for DEAP raw BDF files.

This script is independent from the main preprocessing pipeline. It uses
MNE-Python to fit ICA, draw component topomaps/properties, and provide a
conservative manual-review report. It does not modify preprocess.py defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


SUBJECT_IDS = (1, 10, 24)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/DEAP/original"
RESULTS_DIR = PROJECT_ROOT / "results/mne_ica_review"

EEG_CHANNEL_COUNT = 32
EOG_LIKE_NAMES = ("EXG1", "EXG2", "EXG3", "EXG4")
EMG_LIKE_NAMES = ("EXG5", "EXG6", "EXG7", "EXG8")

BANDPASS_LOW_HZ = 4.0
BANDPASS_HIGH_HZ = 45.0
NOTCH_FREQ_HZ = 50.0
ICA_COMPONENTS = 16
ICA_RANDOM_STATE = 42


def import_mne():
    """Import MNE lazily so missing dependency exits cleanly."""
    try:
        import mne
        from mne.preprocessing import ICA
    except ImportError:
        print("MNE-Python is not installed.")
        print("pip install mne")
        return None, None
    return mne, ICA


def configure_plot_cache() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / "mpl_cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def subject_label(subject_id: int) -> str:
    return f"s{subject_id:02d}"


def safe_save_figure(fig, path: Path, notes: list[str]) -> None:
    """Save MNE/matplotlib figure or figure list without failing the run."""
    import matplotlib.pyplot as plt

    try:
        if isinstance(fig, list):
            for index, item in enumerate(fig):
                item.savefig(path.with_name(f"{path.stem}_{index:02d}{path.suffix}"), dpi=150)
                plt.close(item)
        else:
            fig.savefig(path, dpi=150)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"could not save figure {path.name}: {exc}")


def set_channel_types(raw, subject: str, notes: list[str]) -> tuple[list[str], list[str]]:
    """Set approximate channel types for DEAP BDF review."""
    ch_names = raw.ch_names
    channel_types = {name: "misc" for name in ch_names}

    for name in ch_names[:EEG_CHANNEL_COUNT]:
        channel_types[name] = "eeg"

    eog_names = [name for name in EOG_LIKE_NAMES if name in ch_names]
    emg_names = [name for name in EMG_LIKE_NAMES if name in ch_names]
    for name in eog_names:
        channel_types[name] = "eog"
    for name in emg_names:
        channel_types[name] = "misc"

    status_like = []
    for name in ch_names:
        lowered = name.strip().lower()
        if lowered == "status" or any(key in lowered for key in ("stim", "trigger", "event", "annotation")):
            status_like.append(name)
    if status_like:
        channel_types[status_like[0]] = "stim"
        for name in status_like[1:]:
            channel_types[name] = "misc"
    else:
        last_name = ch_names[-1] if ch_names else ""
        notes.append(f"{subject}: no explicit Status/stim-like channel; last channel is {last_name!r}")
        if last_name:
            channel_types[last_name] = "misc"

    try:
        raw.set_channel_types(channel_types, verbose="ERROR")
    except Exception as exc:
        notes.append(f"{subject}: channel type setting warning: {exc}")

    missing_eog = [name for name in EOG_LIKE_NAMES if name not in ch_names]
    missing_emg = [name for name in EMG_LIKE_NAMES if name not in ch_names]
    if missing_eog:
        notes.append(f"{subject}: missing EOG-like channels: {', '.join(missing_eog)}")
    if missing_emg:
        notes.append(f"{subject}: missing EMG-like/misc channels: {', '.join(missing_emg)}")

    return eog_names, emg_names


def set_montage(raw, mne, subject: str, notes: list[str]) -> str:
    """Try standard montages. Continue if they do not match perfectly."""
    for montage_name in ("biosemi32", "standard_1020"):
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
            return montage_name
        except Exception as exc:
            notes.append(f"{subject}: montage {montage_name} warning: {exc}")
    return "unavailable"


def load_and_prepare_raw(subject_id: int, mne, notes: list[str]):
    subject = subject_label(subject_id)
    bdf_path = DATA_DIR / f"{subject}.bdf"
    if not bdf_path.exists():
        raise FileNotFoundError(f"{bdf_path} does not exist")

    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    notes.append(f"{subject}: loaded {bdf_path}, sfreq={raw.info['sfreq']}, channels={len(raw.ch_names)}")
    notes.append(f"{subject}: last 5 channels: {raw.ch_names[-5:]}")

    eog_names, emg_names = set_channel_types(raw, subject, notes)
    montage_name = set_montage(raw, mne, subject, notes)

    raw.filter(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, picks="eeg", verbose="ERROR")
    raw.notch_filter(NOTCH_FREQ_HZ, picks="eeg", verbose="ERROR")
    notes.append(
        f"{subject}: filtered EEG with {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz bandpass "
        f"and {NOTCH_FREQ_HZ} Hz notch"
    )
    return raw, eog_names, emg_names, montage_name


def fit_mne_ica(raw, ICA):
    raw_eeg = raw.copy().pick("eeg")
    ica = ICA(
        n_components=ICA_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        method="fastica",
        max_iter="auto",
    )
    ica.fit(raw_eeg, verbose="ERROR")
    return ica


def find_eog_candidates(raw, ica, eog_names: list[str], notes: list[str]) -> tuple[list[int], dict[str, list[float]]]:
    """Run MNE EOG scoring for each available EOG-like channel."""
    all_inds = []
    score_by_channel = {}
    for eog_name in eog_names:
        try:
            inds, scores = ica.find_bads_eog(raw, ch_name=eog_name, verbose="ERROR")
            all_inds.extend(int(index) for index in inds)
            score_by_channel[eog_name] = [float(value) for value in np.ravel(scores)]
            notes.append(f"EOG scoring with {eog_name}: candidate inds={inds}")
        except Exception as exc:
            notes.append(f"EOG scoring failed for {eog_name}: {exc}")
    return sorted(set(all_inds)), score_by_channel


def summarize_eog_scores(score_by_channel: dict[str, list[float]]) -> list[tuple[int, float, str]]:
    best = {}
    for channel, scores in score_by_channel.items():
        for index, score in enumerate(scores):
            value = abs(float(score))
            if index not in best or value > best[index][0]:
                best[index] = (value, channel)
    return sorted(
        [(component, value, channel) for component, (value, channel) in best.items()],
        key=lambda item: item[1],
        reverse=True,
    )


def save_mne_plots(raw, ica, subject: str, eog_inds: list[int], score_by_channel: dict[str, list[float]], output_dir: Path, notes: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_eeg = raw.copy().pick("eeg")

    try:
        fig = ica.plot_components(inst=raw_eeg, ch_type="eeg", show=False)
        safe_save_figure(fig, output_dir / "ica_components_topomap.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot ICA topomap: {exc}")

    try:
        import matplotlib.pyplot as plt

        sources = ica.get_sources(raw).get_data(start=0, stop=int(min(10, raw.times[-1]) * raw.info["sfreq"]))
        n_components = min(ICA_COMPONENTS, sources.shape[0])
        time_axis = np.arange(sources.shape[1]) / raw.info["sfreq"]
        fig, axes = plt.subplots(4, 4, figsize=(14, 9), sharex=True)
        for component, axis in enumerate(axes.ravel()[:n_components]):
            axis.plot(time_axis, sources[component], linewidth=0.8)
            axis.set_title(f"C{component:02d}", fontsize=9)
            axis.grid(True, linestyle="--", alpha=0.2)
        fig.suptitle(f"{subject} ICA sources overview, first 10 seconds")
        fig.supxlabel("Time (s)")
        fig.supylabel("Activation")
        fig.tight_layout()
        safe_save_figure(fig, output_dir / "ica_sources_overview.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot ICA sources overview: {exc}")

    if score_by_channel:
        try:
            import matplotlib.pyplot as plt

            ranked = summarize_eog_scores(score_by_channel)
            components = [item[0] for item in ranked]
            values = [item[1] for item in ranked]
            labels = [item[2] for item in ranked]
            fig, axis = plt.subplots(figsize=(10, 4))
            axis.bar(components, values)
            axis.set_title(f"{subject} ICA EOG score ranking")
            axis.set_xlabel("ICA component")
            axis.set_ylabel("max abs EOG score")
            axis.set_xticks(components)
            for component, value, label in zip(components, values, labels):
                axis.text(component, value, label, rotation=90, fontsize=7, ha="center", va="bottom")
            fig.tight_layout()
            safe_save_figure(fig, output_dir / "ica_eog_scores.png", notes)
        except Exception as exc:
            notes.append(f"{subject}: could not plot EOG score figure: {exc}")
    else:
        notes.append(f"{subject}: no EOG scores available; skipped ica_eog_scores.png")

    candidates = eog_inds[:3]
    if not candidates and score_by_channel:
        candidates = [component for component, _, _ in summarize_eog_scores(score_by_channel)[:3]]

    for component in candidates:
        try:
            fig = ica.plot_properties(raw_eeg, picks=[component], show=False)
            safe_save_figure(fig, output_dir / f"component_{component:02d}_properties.png", notes)
        except Exception as exc:
            notes.append(f"{subject}: could not plot properties for component {component}: {exc}")


def recommendation_text(eog_inds: list[int], ranked_scores: list[tuple[int, float, str]]) -> tuple[str, str]:
    if not eog_inds:
        return "manual review only; no automatic deletion", "no MNE EOG candidate was strong enough"

    strong = []
    for component, score, channel in ranked_scores:
        if component in eog_inds and score >= 0.35:
            strong.append((component, score, channel))

    if strong:
        candidates = ", ".join(f"C{component:02d} via {channel} score={score:.3f}" for component, score, channel in strong[:3])
        return "review candidate before subject-specific deletion", candidates
    return "manual review only; no automatic deletion", "EOG candidates exist but scores are not clearly strong"


def analyze_subject(subject_id: int, mne, ICA) -> dict:
    subject = subject_label(subject_id)
    notes = []
    output_dir = RESULTS_DIR / subject
    raw, eog_names, emg_names, montage_name = load_and_prepare_raw(subject_id, mne, notes)
    ica = fit_mne_ica(raw, ICA)
    notes.append(f"{subject}: fitted ICA with n_components={ICA_COMPONENTS}, method=fastica")

    eog_inds, score_by_channel = find_eog_candidates(raw, ica, eog_names, notes)
    ranked_scores = summarize_eog_scores(score_by_channel)
    save_mne_plots(raw, ica, subject, eog_inds, score_by_channel, output_dir, notes)
    recommendation, reason = recommendation_text(eog_inds, ranked_scores)

    return {
        "subject": subject,
        "eog_names": eog_names,
        "emg_names": emg_names,
        "montage_name": montage_name,
        "eog_inds": eog_inds,
        "ranked_scores": ranked_scores[:5],
        "recommendation": recommendation,
        "reason": reason,
        "notes": notes,
    }


def build_report(subject_reports: list[dict]) -> str:
    lines = [
        "MNE ICA review report",
        "",
        "Purpose:",
        "Use MNE-Python to provide ICA topomaps, EOG scores, source plots, "
        "and component properties for manual artifact review.",
        "",
        "Important:",
        "This script does not modify preprocess.py and does not set default component deletion.",
        "",
        "Settings:",
        f"subjects: {', '.join(report['subject'] for report in subject_reports)}",
        f"filter: {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz, notch {NOTCH_FREQ_HZ} Hz",
        f"ICA: n_components={ICA_COMPONENTS}, random_state={ICA_RANDOM_STATE}, method=fastica",
        "",
        "Subject summaries:",
    ]

    for report in subject_reports:
        lines.extend(
            [
                "",
                f"{report['subject']}:",
                f"EOG-like channels: {', '.join(report['eog_names']) or 'unavailable'}",
                f"EMG-like/misc channels: {', '.join(report['emg_names']) or 'unavailable'}",
                f"montage: {report['montage_name']}",
                f"MNE EOG candidate components: {report['eog_inds']}",
                f"recommendation: {report['recommendation']}",
                f"reason: {report['reason']}",
                "top EOG scores:",
            ]
        )
        if report["ranked_scores"]:
            for component, score, channel in report["ranked_scores"]:
                lines.append(f"  C{component:02d}: abs_score={score:.3f}, channel={channel}")
        else:
            lines.append("  unavailable")

        lines.append("notes:")
        lines.extend(f"  - {note}" for note in report["notes"])

    lines.extend(
        [
            "",
            "Default preprocess.py strategy:",
            "Do not modify the default ICA exclusion policy unless a subject-specific "
            "component is confirmed by MNE scores, topomap/properties, and manual review.",
            "",
            "Conservative conclusion:",
            "MNE outputs should be used as evidence for manual review, not as a global "
            "rule such as always deleting component 0 or 1.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure_plot_cache()
    mne, ICA = import_mne()
    if mne is None:
        return

    subject_reports = []
    for subject_id in SUBJECT_IDS:
        subject = subject_label(subject_id)
        print(f"running MNE ICA review for {subject}...")
        try:
            subject_reports.append(analyze_subject(subject_id, mne, ICA))
        except Exception as exc:
            subject_reports.append(
                {
                    "subject": subject,
                    "eog_names": [],
                    "emg_names": [],
                    "montage_name": "unavailable",
                    "eog_inds": [],
                    "ranked_scores": [],
                    "recommendation": "failed; manual inspection needed",
                    "reason": str(exc),
                    "notes": [f"{subject}: failed with error: {exc}"],
                }
            )
            print(f"{subject} failed: {exc}")

    REPORT_PATH = RESULTS_DIR / "mne_ica_review_report.txt"
    REPORT_PATH.write_text(build_report(subject_reports), encoding="utf-8")
    print(f"saved MNE ICA review report to {REPORT_PATH}")
    print(f"saved subject figures under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
