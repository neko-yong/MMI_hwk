"""MNE full ICA artifact-review pipeline for DEAP raw BDF files.

This script is a standalone validation tool. It does not modify preprocess.py
or change any default preprocessing strategy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


SUBJECT_IDS = (1, 10, 24)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/DEAP/original"
RESULTS_DIR = PROJECT_ROOT / "results/mne_ica_full_pipeline"
REPORT_PATH = RESULTS_DIR / "mne_full_report.txt"

EEG_CHANNEL_COUNT = 32
EXG_EOG_NAMES = ("EXG1", "EXG2", "EXG3", "EXG4")
EXG_EMG_NAMES = ("EXG5", "EXG6", "EXG7", "EXG8")

ICA_HIGHPASS_HZ = 1.0
ICA_LOWPASS_HZ = 45.0
ICA_N_COMPONENTS = 16
ICA_RANDOM_STATE = 42


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
    """Save one Matplotlib/MNE figure or a list of figures."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(fig, list):
            for index, item in enumerate(fig):
                item_path = output_path.with_name(
                    f"{output_path.stem}_{index:02d}{output_path.suffix}"
                )
                item.savefig(item_path, dpi=150)
                plt.close(item)
        else:
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
    except Exception as exc:
        notes.append(f"could not save {output_path.name}: {exc}")


def set_channel_types(raw, subject: str, notes: list[str]) -> tuple[list[str], list[str]]:
    """Set DEAP channel types for MNE ICA review."""
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

    stim_candidates = []
    for name in ch_names:
        lowered = name.strip().lower()
        if lowered == "status" or any(
            key in lowered for key in ("stim", "trigger", "event", "annotation")
        ):
            stim_candidates.append(name)

    if stim_candidates:
        channel_types[stim_candidates[0]] = "stim"
        notes.append(f"{subject}: stim/event channel set to {stim_candidates[0]!r}")
    else:
        last_name = ch_names[-1] if ch_names else ""
        if last_name:
            channel_types[last_name] = "stim"
            notes.append(
                f"{subject}: no explicit Status channel; using last channel "
                f"{last_name!r} as stim/event-like channel"
            )
        else:
            notes.append(f"{subject}: no channel available for stim/event type")

    try:
        raw.set_channel_types(channel_types, verbose="ERROR")
    except Exception as exc:
        notes.append(f"{subject}: channel type warning: {exc}")

    print(f"{subject}: EOG channels = {eog_names or 'none'}")
    print(f"{subject}: EMG/misc channels = {emg_names or 'none'}")
    notes.append(f"{subject}: EOG channels set to {eog_names or 'none'}")
    notes.append(f"{subject}: EMG/misc channels set to {emg_names or 'none'}")
    return eog_names, emg_names


def set_biosemi32_montage(raw, mne, subject: str, notes: list[str]) -> str:
    try:
        montage = mne.channels.make_standard_montage("biosemi32")
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
        notes.append(f"{subject}: montage set to biosemi32")
        return "biosemi32"
    except Exception as exc:
        warning = f"{subject}: warning, failed to set biosemi32 montage: {exc}"
        print(warning)
        notes.append(warning)
        return "failed"


def read_subject_raw(subject_id: int, mne) -> tuple:
    subject = subject_label(subject_id)
    notes: list[str] = []
    bdf_path = DATA_DIR / f"{subject}.bdf"
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
    montage_status = set_biosemi32_montage(raw, mne, subject, notes)
    return raw, eog_names, emg_names, montage_status, notes


def make_ica_fit_copy(raw, mne, subject: str, notes: list[str]):
    """Create high-pass ICA-specific copy, following common MNE guidance."""
    raw_ica = raw.copy()
    raw_ica.filter(
        l_freq=ICA_HIGHPASS_HZ,
        h_freq=ICA_LOWPASS_HZ,
        picks=mne.pick_types(raw_ica.info, eeg=True, exclude="bads"),
        verbose="ERROR",
    )
    notes.append(
        f"{subject}: ICA fit copy filtered with highpass={ICA_HIGHPASS_HZ} Hz, "
        f"lowpass={ICA_LOWPASS_HZ} Hz"
    )
    return raw_ica


def fit_ica(raw_ica, mne, ICA, subject: str, notes: list[str]):
    picks = mne.pick_types(raw_ica.info, eeg=True, exclude="bads")
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=ICA_RANDOM_STATE,
        method="fastica",
        max_iter="auto",
    )
    ica.fit(raw_ica, picks=picks, verbose="ERROR")
    notes.append(
        f"{subject}: fitted ICA with n_components={ICA_N_COMPONENTS}, "
        f"random_state={ICA_RANDOM_STATE}, method=fastica, eeg_picks={len(picks)}"
    )
    return ica, picks


def find_eog_artifacts(ica, raw, eog_names: list[str], subject: str, notes: list[str]):
    all_inds: set[int] = set()
    score_by_channel: dict[str, np.ndarray] = {}
    for eog_name in eog_names:
        try:
            inds, scores = ica.find_bads_eog(raw, ch_name=eog_name, verbose="ERROR")
            all_inds.update(int(index) for index in inds)
            score_by_channel[eog_name] = np.asarray(scores, dtype=float)
            print(f"{subject}: eog channel {eog_name}, eog_inds={inds}")
            notes.append(f"{subject}: EOG scoring with {eog_name}, eog_inds={inds}")
        except Exception as exc:
            notes.append(f"{subject}: EOG scoring failed for {eog_name}: {exc}")
    return sorted(all_inds), score_by_channel


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


def save_eog_score_plot(ica, eog_scores, output_dir: Path, subject: str, notes: list[str]) -> None:
    try:
        fig = ica.plot_scores(eog_scores, show=False)
        save_figure(fig, output_dir / "ica_eog_scores.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot EOG scores: {exc}")


def save_sources_overview(ica, raw, output_dir: Path, subject: str, notes: list[str]) -> None:
    """Save a non-interactive sources overview to avoid GUI/browser side effects."""
    try:
        import matplotlib.pyplot as plt

        duration = min(10.0, raw.times[-1])
        sources = ica.get_sources(raw).get_data(start=0, stop=int(duration * raw.info["sfreq"]))
        n_components = min(ICA_N_COMPONENTS, sources.shape[0])
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
        save_figure(fig, output_dir / "ica_sources_overview.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot sources overview: {exc}")


def save_review_plots(ica, raw, raw_ica, eog_inds: list[int], score_by_channel, output_dir: Path, subject: str, notes: list[str]) -> None:
    raw_eeg = raw.copy().pick("eeg")
    raw_ica_eeg = raw_ica.copy().pick("eeg")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = ica.plot_components(inst=raw_ica_eeg, ch_type="eeg", show=False)
        save_figure(fig, output_dir / "ica_components_topomap.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot ICA topomap: {exc}")

    save_sources_overview(ica, raw_eeg, output_dir, subject, notes)

    if score_by_channel:
        top_channel = ranked_eog_scores(score_by_channel)[0][2]
        save_eog_score_plot(ica, score_by_channel[top_channel], output_dir, subject, notes)
    else:
        notes.append(f"{subject}: no EOG scores available, skipped score plot")

    if eog_inds:
        try:
            fig = ica.plot_properties(raw_eeg, picks=eog_inds, show=False)
            save_figure(fig, output_dir / "ica_properties_eog_candidates.png", notes)
        except Exception as exc:
            notes.append(f"{subject}: could not plot ICA properties: {exc}")
    else:
        notes.append(f"{subject}: no EOG candidates, skipped candidate properties")

    try:
        fig = ica.plot_overlay(raw_eeg, exclude=eog_inds, show=False)
        save_figure(fig, output_dir / "ica_overlay.png", notes)
    except Exception as exc:
        notes.append(f"{subject}: could not plot ICA overlay: {exc}")


def artifact_judgment(eog_inds: list[int], ranked_scores: list[tuple[int, float, str]]) -> tuple[str, str, str]:
    if not eog_inds:
        return (
            "no clear automatic EOG artifact",
            "manual review only; no deletion recommended by default",
            "consistent with current conservative preprocess conclusion",
        )

    strong = [
        (component, score, channel)
        for component, score, channel in ranked_scores
        if component in eog_inds and score >= 0.35
    ]
    if strong:
        detail = "; ".join(
            f"C{component:02d} via {channel}, score={score:.3f}"
            for component, score, channel in strong
        )
        return (
            f"possible EOG artifact candidate(s): {detail}",
            "consider subject-specific manual deletion only after checking topomap/properties/overlay",
            "adds evidence for manual review, but does not justify global default deletion",
        )

    return (
        "weak EOG candidates only",
        "manual review only; no deletion recommended by default",
        "consistent with current conservative preprocess conclusion",
    )


def analyze_subject(subject_id: int, mne, ICA) -> dict:
    subject = subject_label(subject_id)
    output_dir = RESULTS_DIR / subject
    raw, eog_names, emg_names, montage_status, notes = read_subject_raw(subject_id, mne)
    raw_ica = make_ica_fit_copy(raw, mne, subject, notes)
    ica, picks = fit_ica(raw_ica, mne, ICA, subject, notes)
    eog_inds, score_by_channel = find_eog_artifacts(ica, raw, eog_names, subject, notes)
    ranked_scores = ranked_eog_scores(score_by_channel)
    save_review_plots(ica, raw, raw_ica, eog_inds, score_by_channel, output_dir, subject, notes)
    artifact_status, deletion_advice, preprocess_consistency = artifact_judgment(
        eog_inds,
        ranked_scores,
    )

    return {
        "subject": subject,
        "sfreq": raw.info["sfreq"],
        "channel_count": len(raw.ch_names),
        "eog_names": eog_names,
        "emg_names": emg_names,
        "montage_status": montage_status,
        "eeg_pick_count": len(picks),
        "eog_inds": eog_inds,
        "ranked_scores": ranked_scores[:5],
        "artifact_status": artifact_status,
        "deletion_advice": deletion_advice,
        "preprocess_consistency": preprocess_consistency,
        "notes": notes,
    }


def build_report(subject_reports: list[dict]) -> str:
    lines = [
        "MNE full ICA pipeline report",
        "",
        "Purpose:",
        "Use MNE-Python as a standalone standard ICA artifact-review workflow "
        "to strengthen preprocessing quality evidence.",
        "",
        "Important:",
        "This script does not modify preprocess.py, does not write default ICA "
        "exclusions, and does not hard-code component IDs as global rules.",
        "",
        "Settings:",
        f"subjects: {', '.join(report['subject'] for report in subject_reports)}",
        f"ICA fit filter: highpass={ICA_HIGHPASS_HZ} Hz, lowpass={ICA_LOWPASS_HZ} Hz",
        f"ICA: n_components={ICA_N_COMPONENTS}, random_state={ICA_RANDOM_STATE}, method=fastica",
        "montage: biosemi32",
        "",
        "Subject summaries:",
    ]

    for report in subject_reports:
        lines.extend(
            [
                "",
                f"{report['subject']}:",
                f"sfreq: {report['sfreq']}",
                f"channel count: {report['channel_count']}",
                f"EEG picks used for ICA: {report['eeg_pick_count']}",
                f"EOG channels: {', '.join(report['eog_names']) or 'unavailable'}",
                f"EMG/misc channels: {', '.join(report['emg_names']) or 'unavailable'}",
                f"montage status: {report['montage_status']}",
                f"EOG candidates: {report['eog_inds']}",
                f"artifact judgment: {report['artifact_status']}",
                f"deletion advice: {report['deletion_advice']}",
                f"consistency with current preprocess conclusion: {report['preprocess_consistency']}",
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
            "Final conservative conclusion:",
            "The MNE full pipeline is useful for validation and evidence enhancement. "
            "It may identify subject-specific candidates for manual review, but it "
            "does not provide enough justification to modify preprocess.py defaults "
            "or to globally delete the same component ID across subjects.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure_environment()
    mne, ICA = import_mne()
    if mne is None:
        return

    subject_reports = []
    for subject_id in SUBJECT_IDS:
        subject = subject_label(subject_id)
        print(f"running MNE full ICA pipeline for {subject}...")
        try:
            subject_reports.append(analyze_subject(subject_id, mne, ICA))
        except Exception as exc:
            print(f"{subject}: failed: {exc}")
            subject_reports.append(
                {
                    "subject": subject,
                    "sfreq": "unavailable",
                    "channel_count": "unavailable",
                    "eog_names": [],
                    "emg_names": [],
                    "montage_status": "unavailable",
                    "eeg_pick_count": 0,
                    "eog_inds": [],
                    "ranked_scores": [],
                    "artifact_status": "failed",
                    "deletion_advice": "manual inspection needed",
                    "preprocess_consistency": "not assessed",
                    "notes": [f"{subject}: failed with error: {exc}"],
                }
            )

    REPORT_PATH.write_text(build_report(subject_reports), encoding="utf-8")
    print(f"saved report to {REPORT_PATH}")
    print(f"saved figures under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
