"""DEAP raw BDF preprocessing utilities for the EEG course project.

The task-1 pipeline is:
load raw BDF -> recover trial boundaries -> cut baseline/stimulus trials ->
fix trial length -> bandpass + notch filtering -> optional ICA -> baseline
correction. Feature extraction and modeling live in other modules.
"""

from __future__ import annotations

import csv
import os
from array import array
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_DIR = Path("data/DEAP/original")
DEFAULT_METADATA_DIR = Path("data/DEAP/metadata")

EEG_CHANNEL_COUNT = 32
DEAP_TRIAL_DURATION_SECONDS = 60
DEAP_BASELINE_DURATION_SECONDS = 5
EXPECTED_DEAP_TRIAL_COUNT = 40
DEAP_SAMPLING_RATE = 512
TARGET_BASELINE_SAMPLES = DEAP_BASELINE_DURATION_SECONDS * DEAP_SAMPLING_RATE
TARGET_STIMULUS_SAMPLES = DEAP_TRIAL_DURATION_SECONDS * DEAP_SAMPLING_RATE
STATUS_CHANNEL_NAME = "Status"
STATUS_EVENT_MASK = 0xFF
EVENT_BASELINE_START = 3
EVENT_STIMULUS_START = 4
EVENT_STIMULUS_END = 5
MIN_BASELINE_SECONDS = 4.0
MAX_BASELINE_SECONDS = 8.0
MIN_STIMULUS_SECONDS = 55.0
MAX_STIMULUS_SECONDS = 65.0

# Conservative defaults for course demonstration:
# - 4-45 Hz keeps common EEG emotion-related bands while reducing drift/noise.
# - 50 Hz notch suppresses mains interference in China.
BANDPASS_LOW_HZ = 4.0
BANDPASS_HIGH_HZ = 45.0
BANDPASS_ORDER = 4
NOTCH_FREQ_HZ = 50.0
NOTCH_QUALITY_FACTOR = 30.0
DEFAULT_ICA_COMPONENTS = 16
DEFAULT_ICA_RANDOM_STATE = 42
PREPROCESSING_RESULTS_DIR = Path("results/preprocessing")
ICA_DEBUG_RESULTS_DIR = Path("results/preprocessing_ica_debug")
DEFAULT_ICA_DEBUG_EXCLUDE_SETS = ((), (0,), (1,), (0, 1))
OFFICIAL_LIKE_SAMPLING_RATE = 128
OFFICIAL_LIKE_BASELINE_SECONDS = 3
OFFICIAL_LIKE_CHANNEL_COUNT = 40
OFFICIAL_LIKE_PERIPHERAL_CHANNEL_COUNT = 8
OFFICIAL_LIKE_LABEL_COLUMNS = ("Valence", "Arousal", "Dominance", "Liking")


def preprocess_eeg(eeg_data):
    """Return EEG data without modification as a placeholder step."""
    return eeg_data


def _decode_ascii(raw_bytes: bytes) -> str:
    return raw_bytes.decode("ascii", errors="ignore").strip()


def read_bdf_header(bdf_path: str | Path) -> dict:
    """Read only the BDF header and return basic structure information."""
    path = Path(bdf_path)
    with path.open("rb") as file:
        fixed_header = file.read(256)

    header_bytes = int(_decode_ascii(fixed_header[184:192]))
    num_records = int(_decode_ascii(fixed_header[236:244]))
    record_duration = float(_decode_ascii(fixed_header[244:252]))
    num_signals = int(_decode_ascii(fixed_header[252:256]))

    with path.open("rb") as file:
        header = file.read(header_bytes)

    offset = 256
    labels = [
        _decode_ascii(header[offset + i * 16 : offset + (i + 1) * 16])
        for i in range(num_signals)
    ]

    offset += num_signals * 16
    offset += num_signals * 80
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 8
    offset += num_signals * 80

    samples_per_record = [
        int(_decode_ascii(header[offset + i * 8 : offset + (i + 1) * 8]))
        for i in range(num_signals)
    ]

    sampling_rates = [
        samples / record_duration for samples in samples_per_record
    ]

    return {
        "file_name": path.name,
        "data_type": "BioSemi BDF raw continuous recording",
        "version": _decode_ascii(fixed_header[0:8]),
        "participant": _decode_ascii(fixed_header[8:88]),
        "header_bytes": header_bytes,
        "num_records": num_records,
        "record_duration_seconds": record_duration,
        "num_signals": num_signals,
        "channel_labels": labels,
        "samples_per_record": samples_per_record,
        "sampling_rates": sampling_rates,
        "total_samples_per_channel": num_records * samples_per_record[0],
    }


def load_bdf_subject(
    subject_id: int = 1,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
) -> dict:
    """Load lightweight metadata for one DEAP raw BDF subject.

    This does not load all EEG samples. The heavy signal read happens later
    when trial intervals are known.
    """
    bdf_path = PROJECT_ROOT / Path(original_dir) / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    return {
        "subject_id": subject_id,
        "bdf_path": bdf_path,
        "header": header,
        "sampling_rate": header["sampling_rates"][0],
        "eeg_channel_labels": header["channel_labels"][:EEG_CHANNEL_COUNT],
    }


def _read_uint24_samples(raw_bytes: bytes) -> list[int]:
    """Decode little-endian 24-bit BDF samples as unsigned integers."""
    samples = []
    for index in range(0, len(raw_bytes), 3):
        samples.append(
            raw_bytes[index]
            | (raw_bytes[index + 1] << 8)
            | (raw_bytes[index + 2] << 16)
        )
    return samples


def _append_signed_int24_samples(
    target: array,
    raw_bytes: bytes,
    start_sample: int = 0,
    end_sample: int | None = None,
) -> None:
    """Append little-endian signed 24-bit BDF samples to an integer array."""
    if end_sample is None:
        end_sample = len(raw_bytes) // 3

    for sample_index in range(start_sample, end_sample):
        byte_index = sample_index * 3
        value = (
            raw_bytes[byte_index]
            | (raw_bytes[byte_index + 1] << 8)
            | (raw_bytes[byte_index + 2] << 16)
        )
        if value & 0x800000:
            value -= 0x1000000
        target.append(value)


def _extract_events_from_channel(bdf_path: Path, header: dict, channel_index: int) -> dict:
    """Extract low-8-bit rising events from one BDF channel."""
    channel_labels = header["channel_labels"]
    samples_per_record = header["samples_per_record"]
    record_bytes = sum(samples_per_record) * 3
    status_offset_in_record = sum(samples_per_record[:channel_index]) * 3
    status_bytes_per_record = samples_per_record[channel_index] * 3
    sampling_rate = header["sampling_rates"][channel_index]

    events = []
    event_value_counts = Counter()
    rising_event_counts = Counter()
    previous_value = None
    sample_index = 0

    with bdf_path.open("rb") as file:
        for record_index in range(header["num_records"]):
            file.seek(
                header["header_bytes"]
                + record_index * record_bytes
                + status_offset_in_record
            )
            raw_status = file.read(status_bytes_per_record)

            for raw_value in _read_uint24_samples(raw_status):
                event_value = raw_value & STATUS_EVENT_MASK
                event_value_counts[event_value] += 1

                if previous_value is None:
                    previous_value = event_value
                elif event_value != previous_value:
                    if previous_value == 0 and event_value != 0:
                        events.append(
                            {
                                "sample": sample_index,
                                "time_seconds": sample_index / sampling_rate,
                                "event_code": event_value,
                            }
                        )
                        rising_event_counts[event_value] += 1
                    previous_value = event_value

                sample_index += 1

    return {
        "status_channel_name": channel_labels[channel_index],
        "status_channel_index": channel_index,
        "sampling_rate": sampling_rate,
        "event_value_counts": dict(sorted(event_value_counts.items())),
        "rising_event_counts": dict(sorted(rising_event_counts.items())),
        "events": events,
    }


def _candidate_event_channel_indices(header: dict) -> list[int]:
    """Prefer Status, otherwise try the last one or two channels."""
    labels = header["channel_labels"]
    if STATUS_CHANNEL_NAME in labels:
        return [labels.index(STATUS_CHANNEL_NAME)]

    start_index = max(0, len(labels) - 2)
    return list(range(start_index, len(labels)))


def _event_info_score(event_info: dict) -> tuple[int, int, int]:
    """Score event channel candidates by recoverable trials and code coverage."""
    trial_count = len(
        infer_trial_boundaries(
            event_info["events"],
            sampling_rate=event_info["sampling_rate"],
            log=False,
            allow_less_than_expected=True,
        )
    )
    counts = event_info["rising_event_counts"]
    code_coverage = sum(1 for code in (3, 4, 5) if counts.get(code, 0) > 0)
    total_345 = sum(counts.get(code, 0) for code in (3, 4, 5))
    return trial_count, code_coverage, total_345


def extract_status_events(bdf_path: str | Path) -> dict:
    """Extract normalized events from the BDF status channel.

    Only the status channel is scanned. EEG signal channels are not loaded.
    BioSemi status values contain high-bit device state information, so this
    function keeps the low 8 bits as event codes.
    """
    path = Path(bdf_path)
    header = read_bdf_header(path)
    candidates = [
        _extract_events_from_channel(path, header, channel_index)
        for channel_index in _candidate_event_channel_indices(header)
    ]
    best = max(candidates, key=_event_info_score)
    if STATUS_CHANNEL_NAME not in header["channel_labels"]:
        print(
            "Status channel not found; selected event-like channel "
            f"index {best['status_channel_index']} name={best['status_channel_name']!r}."
        )
    return best


def infer_trial_boundaries(
    events: list[dict],
    sampling_rate: float | None = None,
    log: bool = True,
    allow_less_than_expected: bool = False,
) -> list[dict]:
    """Infer DEAP trial boundaries from a possibly redundant event stream."""
    sorted_events = sorted(events, key=lambda event: event["sample"])
    counts = Counter(event["event_code"] for event in sorted_events)
    code_counts = {
        EVENT_BASELINE_START: counts.get(EVENT_BASELINE_START, 0),
        EVENT_STIMULUS_START: counts.get(EVENT_STIMULUS_START, 0),
        EVENT_STIMULUS_END: counts.get(EVENT_STIMULUS_END, 0),
    }

    if log:
        print(f"detected code counts 3/4/5: {code_counts}")

    candidate_trials = []
    valid_trials = []
    rejected_trials = []
    last_valid_end_sample = -1

    for start_index, baseline_start in enumerate(sorted_events):
        if baseline_start["event_code"] != EVENT_BASELINE_START:
            continue
        if baseline_start["sample"] <= last_valid_end_sample:
            continue

        stimulus_start = next(
            (
                event for event in sorted_events[start_index + 1 :]
                if event["event_code"] == EVENT_STIMULUS_START
                and event["sample"] > baseline_start["sample"]
            ),
            None,
        )
        if stimulus_start is None:
            continue

        stimulus_end = next(
            (
                event for event in sorted_events
                if event["event_code"] == EVENT_STIMULUS_END
                and event["sample"] > stimulus_start["sample"]
            ),
            None,
        )
        if stimulus_end is None:
            continue

        baseline_duration = (
            stimulus_start["time_seconds"] - baseline_start["time_seconds"]
        )
        stimulus_duration = (
            stimulus_end["time_seconds"] - stimulus_start["time_seconds"]
        )
        candidate = {
            "baseline_start": baseline_start,
            "stimulus_start": stimulus_start,
            "stimulus_end": stimulus_end,
            "baseline_duration_seconds": baseline_duration,
            "stimulus_duration_seconds": stimulus_duration,
            "duration_error": abs(baseline_duration - 5.0) + abs(stimulus_duration - 60.0),
        }
        candidate_trials.append(candidate)

        is_valid = (
            MIN_BASELINE_SECONDS <= baseline_duration <= MAX_BASELINE_SECONDS
            and MIN_STIMULUS_SECONDS <= stimulus_duration <= MAX_STIMULUS_SECONDS
        )

        if is_valid:
            valid_trials.append(candidate)
            last_valid_end_sample = stimulus_end["sample"]
        else:
            rejected_trials.append(candidate)

    if log:
        print(f"candidate trials found: {len(candidate_trials)}")
        print(f"valid trials after duration filtering: {len(valid_trials)}")
        print(
            "removed/rejected anomalous candidate trials: "
            f"{len(candidate_trials) - len(valid_trials)}"
        )

    if len(valid_trials) > EXPECTED_DEAP_TRIAL_COUNT:
        valid_trials = sorted(valid_trials, key=lambda trial: trial["duration_error"])[
            :EXPECTED_DEAP_TRIAL_COUNT
        ]
        valid_trials = sorted(
            valid_trials,
            key=lambda trial: trial["baseline_start"]["sample"],
        )
        if log:
            print(
                "more than 40 valid trials found; selected 40 with durations "
                "closest to 5s baseline and 60s stimulus."
            )

    if len(valid_trials) < EXPECTED_DEAP_TRIAL_COUNT and not allow_less_than_expected:
        details = [
            (
                candidate["baseline_duration_seconds"],
                candidate["stimulus_duration_seconds"],
            )
            for candidate in candidate_trials
        ]
        raise ValueError(
            "Could not recover 40 valid trials from event stream. "
            f"valid={len(valid_trials)}, candidates={len(candidate_trials)}, "
            f"candidate durations={details}"
        )

    trials = []
    for trial_index, candidate in enumerate(valid_trials, start=1):
        baseline_start = candidate["baseline_start"]
        stimulus_start = candidate["stimulus_start"]
        stimulus_end = candidate["stimulus_end"]
        trials.append(
            {
                "trial": trial_index,
                "baseline_start_sample": baseline_start["sample"],
                "stimulus_start_sample": stimulus_start["sample"],
                "stimulus_end_sample": stimulus_end["sample"],
                "baseline_start_time": baseline_start["time_seconds"],
                "stimulus_start_time": stimulus_start["time_seconds"],
                "stimulus_end_time": stimulus_end["time_seconds"],
                "baseline_duration_seconds": candidate["baseline_duration_seconds"],
                "stimulus_duration_seconds": candidate["stimulus_duration_seconds"],
            }
        )

    if log and trials:
        print(
            "first/last valid trial sample range: "
            f"{trials[0]['baseline_start_sample']}-{trials[0]['stimulus_end_sample']} / "
            f"{trials[-1]['baseline_start_sample']}-{trials[-1]['stimulus_end_sample']}"
        )

    return trials


def get_trial_boundaries_from_bdf(bdf_path: str | Path) -> list[dict]:
    """Return inferred trial time boundaries for one raw DEAP BDF file.

    The returned list contains timing and sample-index boundaries only. It does
    not load EEG channels, save trial data, or apply preprocessing.
    """
    event_info = extract_status_events(bdf_path)
    return infer_trial_boundaries(event_info["events"])


def extract_trials_from_status(bdf_path: str | Path) -> dict:
    """Extract DEAP trial boundaries from Status codes 3/4/5.

    code 3 = baseline start, code 4 = stimulus start, code 5 = stimulus end.
    The returned boundaries define baseline=[3,4) and stimulus=[4,5).
    """
    event_info = extract_status_events(bdf_path)
    boundaries = infer_trial_boundaries(event_info["events"])
    return {
        "event_info": event_info,
        "boundaries": boundaries,
    }


def _read_eeg_interval(
    bdf_path: str | Path,
    header: dict,
    start_sample: int,
    end_sample: int,
    eeg_channel_count: int = EEG_CHANNEL_COUNT,
) -> list[array]:
    """Read raw digital EEG samples for one interval and first EEG channels."""
    path = Path(bdf_path)
    samples_per_record = header["samples_per_record"]
    eeg_samples_per_record = samples_per_record[0]

    for channel_index in range(eeg_channel_count):
        if samples_per_record[channel_index] != eeg_samples_per_record:
            raise ValueError("EEG channels have inconsistent samples per record.")

    first_record = start_sample // eeg_samples_per_record
    last_record = (end_sample - 1) // eeg_samples_per_record
    record_bytes = sum(samples_per_record) * 3
    eeg_channel_bytes = eeg_samples_per_record * 3
    channel_data = [array("i") for _ in range(eeg_channel_count)]

    with path.open("rb") as file:
        for record_index in range(first_record, last_record + 1):
            record_start_sample = record_index * eeg_samples_per_record
            local_start = max(start_sample - record_start_sample, 0)
            local_end = min(end_sample - record_start_sample, eeg_samples_per_record)

            for channel_index in range(eeg_channel_count):
                channel_offset = channel_index * eeg_channel_bytes
                file.seek(
                    header["header_bytes"]
                    + record_index * record_bytes
                    + channel_offset
                )
                raw_channel = file.read(eeg_channel_bytes)
                _append_signed_int24_samples(
                    channel_data[channel_index],
                    raw_channel,
                    local_start,
                    local_end,
                )

    return channel_data


def _read_channels_interval(
    bdf_path: str | Path,
    header: dict,
    start_sample: int,
    end_sample: int,
    channel_indices: list[int],
) -> list[array]:
    """Read raw digital samples for selected BDF channels.

    This helper is used only by the optional official-like output branch. It
    keeps the stable first-32-EEG preprocessing path untouched.
    """
    path = Path(bdf_path)
    samples_per_record = header["samples_per_record"]
    reference_samples_per_record = samples_per_record[channel_indices[0]]

    for channel_index in channel_indices:
        if samples_per_record[channel_index] != reference_samples_per_record:
            raise ValueError(
                "Selected official-like channels have inconsistent sampling "
                "rates; official_like_data cannot be built reliably."
            )

    first_record = start_sample // reference_samples_per_record
    last_record = (end_sample - 1) // reference_samples_per_record
    record_bytes = sum(samples_per_record) * 3
    channel_bytes_per_record = reference_samples_per_record * 3
    channel_offsets = [
        sum(samples_per_record[:channel_index]) * 3
        for channel_index in channel_indices
    ]
    channel_data = [array("i") for _ in channel_indices]

    with path.open("rb") as file:
        for record_index in range(first_record, last_record + 1):
            record_start_sample = record_index * reference_samples_per_record
            local_start = max(start_sample - record_start_sample, 0)
            local_end = min(
                end_sample - record_start_sample,
                reference_samples_per_record,
            )

            for output_index, channel_offset in enumerate(channel_offsets):
                file.seek(
                    header["header_bytes"]
                    + record_index * record_bytes
                    + channel_offset
                )
                raw_channel = file.read(channel_bytes_per_record)
                _append_signed_int24_samples(
                    channel_data[output_index],
                    raw_channel,
                    local_start,
                    local_end,
                )

    return channel_data


def extract_raw_eeg_trials_from_bdf(
    bdf_path: str | Path,
    eeg_channel_count: int = EEG_CHANNEL_COUNT,
) -> dict:
    """Cut raw EEG trial snippets from one subject's continuous BDF file.

    This function only extracts raw digital samples. It does not filter,
    remove artifacts, baseline-correct, extract features, or save files.
    """
    path = Path(bdf_path)
    header = read_bdf_header(path)
    boundaries = get_trial_boundaries_from_bdf(path)
    sampling_rate = header["sampling_rates"][0]
    trials = []

    for boundary in boundaries:
        baseline = _read_eeg_interval(
            path,
            header,
            boundary["baseline_start_sample"],
            boundary["stimulus_start_sample"],
            eeg_channel_count,
        )
        stimulus = _read_eeg_interval(
            path,
            header,
            boundary["stimulus_start_sample"],
            boundary["stimulus_end_sample"],
            eeg_channel_count,
        )
        trials.append(
            {
                "trial": boundary["trial"],
                "boundary": boundary,
                "baseline": baseline,
                "stimulus": stimulus,
            }
        )

    return {
        "file_name": path.name,
        "sampling_rate": sampling_rate,
        "eeg_channel_labels": header["channel_labels"][:eeg_channel_count],
        "trials": trials,
    }


def summarize_extracted_raw_eeg_trials(extracted: dict) -> dict:
    """Summarize shapes and memory use of extracted raw EEG trial snippets."""
    trials = extracted["trials"]
    first_trial = trials[0]
    baseline_sample_counts = [
        len(trial["baseline"][0]) for trial in trials
    ]
    stimulus_sample_counts = [
        len(trial["stimulus"][0]) for trial in trials
    ]
    baseline_channel_count = len(first_trial["baseline"])
    stimulus_channel_count = len(first_trial["stimulus"])
    baseline_total_values = sum(
        len(channel) for trial in trials for channel in trial["baseline"]
    )
    stimulus_total_values = sum(
        len(channel) for trial in trials for channel in trial["stimulus"]
    )
    bytes_per_sample = first_trial["baseline"][0].itemsize

    return {
        "file_name": extracted["file_name"],
        "trial_count": len(trials),
        "eeg_channel_count": baseline_channel_count,
        "baseline_sample_counts": baseline_sample_counts,
        "stimulus_sample_counts": stimulus_sample_counts,
        "baseline_shape_first_trial": (
            baseline_channel_count,
            len(first_trial["baseline"][0]),
        ),
        "stimulus_shape_first_trial": (
            stimulus_channel_count,
            len(first_trial["stimulus"][0]),
        ),
        "all_baseline_shape": (
            len(trials),
            baseline_channel_count,
            "variable_samples",
        ),
        "all_stimulus_shape": (
            len(trials),
            stimulus_channel_count,
            "variable_samples",
        ),
        "stored_sample_type": "array('i') signed 24-bit raw digital values",
        "estimated_memory_mb": (
            (baseline_total_values + stimulus_total_values)
            * bytes_per_sample
            / (1024 * 1024)
        ),
    }


def _crop_or_pad_channel(
    channel: array,
    target_samples: int,
    trial_index: int,
    segment_name: str,
    channel_index: int,
    warnings: list[str],
) -> array:
    """Crop one channel to fixed length or pad short data with zeros."""
    current_samples = len(channel)

    if current_samples >= target_samples:
        return array("i", channel[:target_samples])

    warning = (
        f"trial {trial_index}, {segment_name}, channel {channel_index + 1}: "
        f"{current_samples} samples < target {target_samples}; "
        "zero-padding is applied to keep a fixed shape."
    )
    warnings.append(warning)

    fixed_channel = array("i", channel)
    fixed_channel.extend([0] * (target_samples - current_samples))
    return fixed_channel


def standardize_raw_eeg_trial_lengths(
    extracted: dict,
    baseline_samples: int = TARGET_BASELINE_SAMPLES,
    stimulus_samples: int = TARGET_STIMULUS_SAMPLES,
) -> dict:
    """Crop/pad raw EEG trial snippets to fixed DEAP baseline/stimulus lengths.

    This step only organizes raw data into uniform shapes. It does not filter,
    apply ICA, remove artifacts, baseline-correct, or extract features.
    """
    warnings = []
    standardized_trials = []

    for trial in extracted["trials"]:
        fixed_baseline = []
        fixed_stimulus = []

        for channel_index, channel in enumerate(trial["baseline"]):
            fixed_baseline.append(
                _crop_or_pad_channel(
                    channel,
                    baseline_samples,
                    trial["trial"],
                    "baseline",
                    channel_index,
                    warnings,
                )
            )

        for channel_index, channel in enumerate(trial["stimulus"]):
            fixed_stimulus.append(
                _crop_or_pad_channel(
                    channel,
                    stimulus_samples,
                    trial["trial"],
                    "stimulus",
                    channel_index,
                    warnings,
                )
            )

        standardized_trials.append(
            {
                "trial": trial["trial"],
                "boundary": trial["boundary"],
                "baseline": fixed_baseline,
                "stimulus": fixed_stimulus,
            }
        )

    trial_count = len(standardized_trials)
    channel_count = len(standardized_trials[0]["baseline"]) if trial_count else 0

    return {
        "file_name": extracted["file_name"],
        "sampling_rate": extracted["sampling_rate"],
        "eeg_channel_labels": extracted["eeg_channel_labels"],
        "trials": standardized_trials,
        "baseline_shape": (trial_count, channel_count, baseline_samples),
        "stimulus_shape": (trial_count, channel_count, stimulus_samples),
        "warnings": warnings,
    }


def fix_trial_length(
    extracted: dict,
    baseline_samples: int = TARGET_BASELINE_SAMPLES,
    stimulus_samples: int = TARGET_STIMULUS_SAMPLES,
) -> dict:
    """Public wrapper for fixed-length DEAP trial organization."""
    return standardize_raw_eeg_trial_lengths(
        extracted,
        baseline_samples=baseline_samples,
        stimulus_samples=stimulus_samples,
    )


def _require_signal_processing_dependencies():
    """Import NumPy/SciPy only when the preprocessing chain is requested."""
    try:
        import numpy as np
        from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "Basic preprocessing requires numpy and scipy. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    return np, butter, filtfilt, iirnotch, sosfiltfilt


def _segments_to_numpy(trials: list[dict], segment_name: str):
    """Convert fixed-length array('i') trial segments to a 3D NumPy array."""
    np, _, _, _, _ = _require_signal_processing_dependencies()
    return np.asarray(
        [
            [list(channel) for channel in trial[segment_name]]
            for trial in trials
        ],
        dtype=np.float32,
    )


def bandpass_filter_eeg(
    eeg_data,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
):
    """Apply Butterworth bandpass filtering to EEG data.

    Expected input shape: (trials, channels, samples). Filtering is applied
    along the last axis. No ICA, artifact removal, or feature extraction is
    performed here.
    """
    _, butter, _, _, sosfiltfilt = _require_signal_processing_dependencies()
    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return sosfiltfilt(sos, eeg_data, axis=-1).astype("float32")


def notch_filter_eeg(
    eeg_data,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    notch_hz: float = NOTCH_FREQ_HZ,
    quality_factor: float = NOTCH_QUALITY_FACTOR,
):
    """Apply a 50 Hz notch filter to suppress mains interference."""
    _, _, filtfilt, iirnotch, _ = _require_signal_processing_dependencies()
    b, a = iirnotch(notch_hz, quality_factor, fs=sampling_rate)
    return filtfilt(b, a, eeg_data, axis=-1).astype("float32")


def bandpass_and_notch_filter(
    baseline_data,
    stimulus_data,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    bandpass_order: int = BANDPASS_ORDER,
    notch_hz: float = NOTCH_FREQ_HZ,
    notch_quality_factor: float = NOTCH_QUALITY_FACTOR,
) -> dict:
    """Apply 4-45 Hz bandpass and 50 Hz notch filtering to EEG trials."""
    bandpassed_baseline = bandpass_filter_eeg(
        baseline_data,
        sampling_rate=sampling_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        order=bandpass_order,
    )
    bandpassed_stimulus = bandpass_filter_eeg(
        stimulus_data,
        sampling_rate=sampling_rate,
        low_hz=low_hz,
        high_hz=high_hz,
        order=bandpass_order,
    )
    filtered_baseline = notch_filter_eeg(
        bandpassed_baseline,
        sampling_rate=sampling_rate,
        notch_hz=notch_hz,
        quality_factor=notch_quality_factor,
    )
    filtered_stimulus = notch_filter_eeg(
        bandpassed_stimulus,
        sampling_rate=sampling_rate,
        notch_hz=notch_hz,
        quality_factor=notch_quality_factor,
    )
    return {
        "filtered_baseline": filtered_baseline,
        "filtered_stimulus": filtered_stimulus,
    }


def _fit_ica_decomposition(
    baseline_data,
    stimulus_data,
    n_components: int | None,
    random_state: int,
) -> dict:
    """Fit ICA once on baseline+stimulus EEG data."""
    np, _, _, _, _ = _require_signal_processing_dependencies()
    try:
        from sklearn.decomposition import FastICA
    except ImportError as exc:
        raise ImportError(
            "ICA artifact removal requires scikit-learn. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    combined = np.concatenate([baseline_data, stimulus_data], axis=-1)
    trial_count, channel_count, sample_count = combined.shape
    component_count = min(n_components or channel_count, channel_count)
    flattened = combined.transpose(0, 2, 1).reshape(-1, channel_count)
    print(f"ICA fitted data shape: {flattened.shape}")

    ica = FastICA(
        n_components=component_count,
        random_state=random_state,
        whiten="unit-variance",
        max_iter=300,
        tol=0.001,
    )
    sources = ica.fit_transform(flattened)
    if getattr(ica, "mixing_", None) is not None:
        component_energy = np.linalg.norm(ica.mixing_, axis=0)
    else:
        component_energy = np.mean(sources ** 2, axis=0)
    return {
        "ica": ica,
        "sources": sources,
        "fit_shape": flattened.shape,
        "source_shape": (trial_count, sample_count, sources.shape[1]),
        "component_energy": component_energy,
        "baseline_samples": baseline_data.shape[-1],
    }


def _apply_ica_exclusion(decomposition: dict, exclude_components) -> dict:
    """Reconstruct EEG data after zeroing selected ICA components."""
    sources = decomposition["sources"].copy()
    source_count = sources.shape[1]
    valid_excluded = [
        component for component in list(exclude_components or [])
        if 0 <= component < source_count
    ]
    sources[:, valid_excluded] = 0.0
    reconstructed = decomposition["ica"].inverse_transform(sources)
    trial_count, sample_count, _ = decomposition["source_shape"]
    reconstructed = reconstructed.reshape(trial_count, sample_count, -1)
    reconstructed = reconstructed.transpose(0, 2, 1).astype("float32")
    baseline_samples = decomposition["baseline_samples"]
    return {
        "baseline": reconstructed[:, :, :baseline_samples],
        "stimulus": reconstructed[:, :, baseline_samples:],
        "removed_components": valid_excluded,
    }


def run_ica_artifact_removal(
    baseline_data,
    stimulus_data,
    enable_ica: bool = False,
    n_components: int | None = DEFAULT_ICA_COMPONENTS,
    random_state: int = DEFAULT_ICA_RANDOM_STATE,
    exclude_components: list[int] | tuple[int, ...] | None = None,
) -> dict:
    """Optionally remove manually selected ICA components from EEG trials.

    Automatic artifact detection is intentionally not forced here because this
    project does not include EOG labels. For stable course demos, pass manual
    component IDs through exclude_components, or leave ICA disabled.
    """
    np, _, _, _, _ = _require_signal_processing_dependencies()
    excluded = list(exclude_components or [])

    print(f"ICA enabled: {enable_ica}")
    print(f"ICA n_components: {n_components}")
    print(f"ICA exclude components: {excluded}")

    if not enable_ica:
        return {
            "baseline": baseline_data,
            "stimulus": stimulus_data,
            "removed_components": [],
            "enabled": False,
            "fit_shape": None,
            "sources": None,
            "mixing_matrix": None,
            "unmixing_matrix": None,
            "component_energy": [],
            "note": "ICA disabled by parameter.",
        }

    decomposition = _fit_ica_decomposition(
        baseline_data,
        stimulus_data,
        n_components=n_components,
        random_state=random_state,
    )
    applied = _apply_ica_exclusion(decomposition, excluded)
    valid_excluded = applied["removed_components"]
    print(f"ICA removed components: {valid_excluded}")
    return {
        "baseline": applied["baseline"],
        "stimulus": applied["stimulus"],
        "removed_components": valid_excluded,
        "enabled": True,
        "fit_shape": decomposition["fit_shape"],
        "sources": decomposition["sources"],
        "source_shape": decomposition["source_shape"],
        "mixing_matrix": getattr(decomposition["ica"], "mixing_", None),
        "unmixing_matrix": getattr(decomposition["ica"], "components_", None),
        "component_energy": decomposition["component_energy"].tolist(),
        "note": "ICA fitted; manual component exclusion was applied if provided.",
    }


def baseline_correction(filtered_baseline, filtered_stimulus):
    """Subtract each trial/channel baseline mean from the stimulus segment."""
    baseline_mean = filtered_baseline.mean(axis=-1, keepdims=True)
    return (filtered_stimulus - baseline_mean).astype("float32")


def baseline_correct_stimulus(filtered_baseline, filtered_stimulus):
    """Backward-compatible wrapper for baseline correction."""
    return baseline_correction(filtered_baseline, filtered_stimulus)


def _target_sample_count(duration_seconds: float, sampling_rate: float) -> int:
    """Return integer sample count for a duration/sampling-rate pair."""
    return int(round(duration_seconds * sampling_rate))


def _downsample_last_axis(
    data,
    source_sfreq: float,
    target_sfreq: float,
    expected_samples: int,
    name: str,
):
    """Downsample with polyphase filtering and validate the final length."""
    from math import gcd
    from scipy.signal import resample_poly

    source = int(round(source_sfreq))
    target = int(round(target_sfreq))
    divisor = gcd(source, target)
    downsampled = resample_poly(
        data,
        up=target // divisor,
        down=source // divisor,
        axis=-1,
    ).astype("float32")

    if downsampled.shape[-1] != expected_samples:
        raise ValueError(
            f"{name} downsampled to {downsampled.shape[-1]} samples, "
            f"expected {expected_samples} samples."
        )

    return downsampled


def load_official_like_labels(
    subject_id: int,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
):
    """Load Valence/Arousal/Dominance/Liking labels as a (40, 4) array."""
    np, _, _, _, _ = _require_signal_processing_dependencies()
    rows = load_participant_ratings(subject_id, metadata_dir)
    rows = sorted(rows, key=lambda row: int(row["Trial"]))

    if len(rows) != EXPECTED_DEAP_TRIAL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DEAP_TRIAL_COUNT} ratings for s{subject_id:02d}, "
            f"got {len(rows)}."
        )

    labels = np.asarray(
        [
            [float(row[column]) for column in OFFICIAL_LIKE_LABEL_COLUMNS]
            for row in rows
        ],
        dtype="float32",
    )
    expected_shape = (
        EXPECTED_DEAP_TRIAL_COUNT,
        len(OFFICIAL_LIKE_LABEL_COLUMNS),
    )
    if labels.shape != expected_shape:
        raise ValueError(
            f"official_like_labels shape is {labels.shape}, "
            f"expected {expected_shape}."
        )
    return labels


def _build_official_like_baseline_corrected_eeg(
    baseline_corrected_stimulus,
    source_sfreq: float,
    target_sfreq: float,
):
    """Return modeling-friendly baseline-corrected EEG at official-like 128 Hz."""
    expected_samples = _target_sample_count(DEAP_TRIAL_DURATION_SECONDS, target_sfreq)
    official_like = _downsample_last_axis(
        baseline_corrected_stimulus,
        source_sfreq,
        target_sfreq,
        expected_samples,
        "official_like_baseline_corrected_eeg",
    )
    expected_shape = (
        EXPECTED_DEAP_TRIAL_COUNT,
        EEG_CHANNEL_COUNT,
        expected_samples,
    )
    if official_like.shape != expected_shape:
        raise ValueError(
            "official_like_baseline_corrected_eeg shape is "
            f"{official_like.shape}, expected {expected_shape}."
        )
    return official_like


def _build_official_like_eeg_window(
    preprocessed: dict,
    source_sfreq: float,
    target_sfreq: float,
):
    """Build first-32 EEG channels as 3s baseline + 60s stimulus at target Hz."""
    baseline_source_samples = _target_sample_count(
        OFFICIAL_LIKE_BASELINE_SECONDS,
        source_sfreq,
    )
    baseline_target_samples = _target_sample_count(
        OFFICIAL_LIKE_BASELINE_SECONDS,
        target_sfreq,
    )
    stimulus_target_samples = _target_sample_count(
        DEAP_TRIAL_DURATION_SECONDS,
        target_sfreq,
    )

    baseline = preprocessed["ica_baseline"][:, :, -baseline_source_samples:]
    stimulus = preprocessed["ica_stimulus"]
    baseline_128 = _downsample_last_axis(
        baseline,
        source_sfreq,
        target_sfreq,
        baseline_target_samples,
        "official_like_eeg_baseline",
    )
    stimulus_128 = _downsample_last_axis(
        stimulus,
        source_sfreq,
        target_sfreq,
        stimulus_target_samples,
        "official_like_eeg_stimulus",
    )
    return preprocessed_array_concatenate((baseline_128, stimulus_128), axis=-1)


def preprocessed_array_concatenate(arrays, axis: int = -1):
    """Small NumPy wrapper to keep NumPy as a lazy preprocessing dependency."""
    np, _, _, _, _ = _require_signal_processing_dependencies()
    return np.concatenate(arrays, axis=axis).astype("float32")


def _build_official_like_peripheral_window(
    bdf_path: Path,
    header: dict,
    fixed: dict,
    source_sfreq: float,
    target_sfreq: float,
):
    """Build real EXG/peripheral channels for optional official-like output.

    The first 32 EEG channels use the project EEG preprocessing chain. These
    8 channels are read from the raw BDF and resampled to match the official
    3s+60s layout. They are real channel data, not zero placeholders, but they
    are not claimed to reproduce DEAP's undisclosed peripheral preprocessing.
    """
    np, _, _, _, _ = _require_signal_processing_dependencies()
    if header["num_signals"] < OFFICIAL_LIKE_CHANNEL_COUNT:
        raise ValueError(
            f"BDF has {header['num_signals']} channels; cannot read first "
            f"{OFFICIAL_LIKE_CHANNEL_COUNT} channels for official_like_data."
        )

    peripheral_indices = list(
        range(
            EEG_CHANNEL_COUNT,
            EEG_CHANNEL_COUNT + OFFICIAL_LIKE_PERIPHERAL_CHANNEL_COUNT,
        )
    )
    for channel_index in peripheral_indices:
        channel_sfreq = float(header["sampling_rates"][channel_index])
        if abs(channel_sfreq - source_sfreq) > 1e-6:
            raise ValueError(
                "Peripheral channel sampling rate mismatch: "
                f"channel {channel_index} is {channel_sfreq} Hz, "
                f"expected {source_sfreq} Hz."
            )

    baseline_source_samples = _target_sample_count(
        OFFICIAL_LIKE_BASELINE_SECONDS,
        source_sfreq,
    )
    stimulus_source_samples = _target_sample_count(
        DEAP_TRIAL_DURATION_SECONDS,
        source_sfreq,
    )
    baseline_target_samples = _target_sample_count(
        OFFICIAL_LIKE_BASELINE_SECONDS,
        target_sfreq,
    )
    stimulus_target_samples = _target_sample_count(
        DEAP_TRIAL_DURATION_SECONDS,
        target_sfreq,
    )

    warnings: list[str] = []
    trial_windows = []
    for trial in fixed["trials"]:
        boundary = trial["boundary"]
        stimulus_start = boundary["stimulus_start_sample"]
        baseline_start = max(
            boundary["baseline_start_sample"],
            stimulus_start - baseline_source_samples,
        )
        stimulus_end = min(
            boundary["stimulus_end_sample"],
            stimulus_start + stimulus_source_samples,
        )

        baseline_channels = _read_channels_interval(
            bdf_path,
            header,
            baseline_start,
            stimulus_start,
            peripheral_indices,
        )
        stimulus_channels = _read_channels_interval(
            bdf_path,
            header,
            stimulus_start,
            stimulus_end,
            peripheral_indices,
        )
        fixed_baseline = [
            _crop_or_pad_channel(
                channel,
                baseline_source_samples,
                trial["trial"],
                "official_like_peripheral_baseline",
                output_index,
                warnings,
            )
            for output_index, channel in enumerate(baseline_channels)
        ]
        fixed_stimulus = [
            _crop_or_pad_channel(
                channel,
                stimulus_source_samples,
                trial["trial"],
                "official_like_peripheral_stimulus",
                output_index,
                warnings,
            )
            for output_index, channel in enumerate(stimulus_channels)
        ]
        baseline = np.asarray([list(channel) for channel in fixed_baseline], dtype="float32")
        stimulus = np.asarray([list(channel) for channel in fixed_stimulus], dtype="float32")
        baseline_128 = _downsample_last_axis(
            baseline,
            source_sfreq,
            target_sfreq,
            baseline_target_samples,
            "official_like_peripheral_baseline",
        )
        stimulus_128 = _downsample_last_axis(
            stimulus,
            source_sfreq,
            target_sfreq,
            stimulus_target_samples,
            "official_like_peripheral_stimulus",
        )
        trial_windows.append(
            preprocessed_array_concatenate((baseline_128, stimulus_128), axis=-1)
        )

    return np.asarray(trial_windows, dtype="float32"), warnings


def _build_official_like_outputs(
    subject_id: int,
    subject: dict,
    fixed: dict,
    preprocessed: dict,
    official_like_sfreq: float,
    include_official_like_data: bool,
) -> dict:
    """Build optional official .dat-style outputs without changing defaults."""
    source_sfreq = float(preprocessed["sampling_rate"])
    labels = load_official_like_labels(subject_id)
    baseline_corrected_eeg = _build_official_like_baseline_corrected_eeg(
        preprocessed["baseline_corrected_stimulus"],
        source_sfreq,
        official_like_sfreq,
    )

    eeg_window = _build_official_like_eeg_window(
        preprocessed,
        source_sfreq,
        official_like_sfreq,
    )
    official_like_data = None
    official_like_data_reason = "include_official_like_data=False"
    peripheral_warnings: list[str] = []
    channel_names = list(subject["header"]["channel_labels"][:EEG_CHANNEL_COUNT])

    if include_official_like_data:
        try:
            peripheral_window, peripheral_warnings = _build_official_like_peripheral_window(
                subject["bdf_path"],
                subject["header"],
                fixed,
                source_sfreq,
                official_like_sfreq,
            )
            official_like_data = preprocessed_array_concatenate(
                (eeg_window, peripheral_window),
                axis=1,
            )
            channel_names = list(
                subject["header"]["channel_labels"][:OFFICIAL_LIKE_CHANNEL_COUNT]
            )
            expected_shape = (
                EXPECTED_DEAP_TRIAL_COUNT,
                OFFICIAL_LIKE_CHANNEL_COUNT,
                _target_sample_count(
                    OFFICIAL_LIKE_BASELINE_SECONDS + DEAP_TRIAL_DURATION_SECONDS,
                    official_like_sfreq,
                ),
            )
            if official_like_data.shape != expected_shape:
                raise ValueError(
                    f"official_like_data shape is {official_like_data.shape}, "
                    f"expected {expected_shape}."
                )
            official_like_data_reason = (
                "constructed from project-preprocessed EEG channels plus real "
                "raw BDF channels 32-39 resampled to official-like timing"
            )
        except Exception as exc:
            official_like_data = None
            official_like_data_reason = (
                "official_like_data unavailable because real peripheral "
                f"channels could not be constructed reliably: {exc}"
            )

    preprocessing_info = {
        "original_sampling_rate": source_sfreq,
        "official_like_sampling_rate": official_like_sfreq,
        "official_like_data_shape": (
            None if official_like_data is None else tuple(official_like_data.shape)
        ),
        "official_like_baseline_corrected_eeg_shape": tuple(
            baseline_corrected_eeg.shape
        ),
        "official_like_labels_shape": tuple(labels.shape),
        "official_like_channel_names": channel_names,
        "official_like_label_columns": OFFICIAL_LIKE_LABEL_COLUMNS,
        "official_like_is_official_dat_exact_copy": False,
        "official_like_note": (
            "Official-like structure generated from project preprocessing "
            "pipeline; not an exact official .dat copy."
        ),
        "official_like_data_reason": official_like_data_reason,
        "official_like_peripheral_note": (
            "If official_like_data is available, channels 0-31 are EEG from "
            "the project preprocessing chain; channels 32-39 are real BDF "
            "auxiliary/peripheral channels resampled for structural "
            "compatibility, not a claimed reproduction of DEAP's private "
            "peripheral preprocessing."
        ),
        "official_like_peripheral_warnings": peripheral_warnings,
        "ica_default_strategy": "no automatic component deletion",
    }

    return {
        "official_like_baseline_corrected_eeg": baseline_corrected_eeg,
        "official_like_data": official_like_data,
        "official_like_labels": labels,
        "official_like_sampling_rate": official_like_sfreq,
        "preprocessing_info": preprocessing_info,
    }


def run_basic_preprocessing_on_standardized_trials(
    standardized: dict,
    enable_ica: bool = False,
    use_ica: bool | None = None,
    ica_n_components: int | None = DEFAULT_ICA_COMPONENTS,
    ica_random_state: int = DEFAULT_ICA_RANDOM_STATE,
    ica_exclude_components: list[int] | tuple[int, ...] | None = None,
) -> dict:
    """Run the minimal baseline preprocessing chain on fixed-length raw trials.

    Steps:
    1. Convert fixed-length raw digital samples to NumPy arrays.
    2. Apply 4-45 Hz bandpass filtering to baseline and stimulus.
    3. Apply 50 Hz notch filtering to the bandpass-filtered data.
    4. Optionally remove manually selected ICA components.
    5. Correct stimulus by subtracting the filtered baseline mean.
    """
    raw_baseline = _segments_to_numpy(standardized["trials"], "baseline")
    raw_stimulus = _segments_to_numpy(standardized["trials"], "stimulus")
    sampling_rate = standardized["sampling_rate"]
    ica_enabled = enable_ica if use_ica is None else use_ica

    filtered = bandpass_and_notch_filter(
        raw_baseline,
        raw_stimulus,
        sampling_rate=sampling_rate,
    )
    ica_result = run_ica_artifact_removal(
        filtered["filtered_baseline"],
        filtered["filtered_stimulus"],
        enable_ica=ica_enabled,
        n_components=ica_n_components,
        random_state=ica_random_state,
        exclude_components=ica_exclude_components,
    )
    corrected_stimulus = baseline_correction(
        ica_result["baseline"],
        ica_result["stimulus"],
    )

    return {
        "file_name": standardized["file_name"],
        "sampling_rate": sampling_rate,
        "filter_parameters": {
            "bandpass_hz": (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ),
            "bandpass_order": BANDPASS_ORDER,
            "notch_hz": NOTCH_FREQ_HZ,
            "notch_quality_factor": NOTCH_QUALITY_FACTOR,
        },
        "raw_fixed_baseline": raw_baseline,
        "raw_fixed_stimulus": raw_stimulus,
        "filtered_baseline": filtered["filtered_baseline"],
        "filtered_stimulus": filtered["filtered_stimulus"],
        "ica_baseline": ica_result["baseline"],
        "ica_stimulus": ica_result["stimulus"],
        "ica_info": {
            "enabled": ica_result["enabled"],
            "n_components": ica_n_components,
            "random_state": ica_random_state,
            "removed_components": ica_result["removed_components"],
            "fit_shape": ica_result["fit_shape"],
            "source_shape": ica_result.get("source_shape"),
            "sources": ica_result.get("sources"),
            "mixing_matrix": ica_result.get("mixing_matrix"),
            "unmixing_matrix": ica_result.get("unmixing_matrix"),
            "component_energy": ica_result["component_energy"],
            "note": ica_result["note"],
        },
        "baseline_corrected_stimulus": corrected_stimulus,
    }


def preprocess_subject(
    subject_id: int = 1,
    enable_ica: bool = False,
    use_ica: bool | None = None,
    ica_n_components: int | None = DEFAULT_ICA_COMPONENTS,
    ica_random_state: int = DEFAULT_ICA_RANDOM_STATE,
    ica_exclude_components: list[int] | tuple[int, ...] | None = None,
    output_official_like: bool = False,
    official_like_sfreq: float = OFFICIAL_LIKE_SAMPLING_RATE,
    include_official_like_data: bool = True,
) -> dict:
    """Run the complete task-1 EEG preprocessing pipeline for one subject.

    By default, the returned fields stay backward-compatible with earlier
    project code. Set output_official_like=True to add optional DEAP official
    .dat-style outputs without changing the stable baseline preprocessing.
    """
    ica_enabled = enable_ica if use_ica is None else use_ica
    subject = load_bdf_subject(subject_id)
    extracted = extract_raw_eeg_trials_from_bdf(subject["bdf_path"])
    fixed = fix_trial_length(extracted)
    preprocessed = run_basic_preprocessing_on_standardized_trials(
        fixed,
        enable_ica=ica_enabled,
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_exclude_components=ica_exclude_components,
    )
    result = {
        "subject_id": subject_id,
        "bdf_path": subject["bdf_path"],
        "header": subject["header"],
        "raw_trials": extracted,
        "fixed_trials": fixed,
        **preprocessed,
    }

    if output_official_like:
        result.update(
            _build_official_like_outputs(
                subject_id,
                subject,
                fixed,
                preprocessed,
                official_like_sfreq,
                include_official_like_data,
            )
        )

    return result


def _get_matplotlib_pyplot():
    PREPROCESSING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(PREPROCESSING_RESULTS_DIR / "mpl_cache"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Preprocessing visualization requires matplotlib. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return plt


def _plot_signal_comparison(
    before,
    after,
    title: str,
    before_label: str,
    after_label: str,
    output_path: Path,
    sampling_rate: float = DEAP_SAMPLING_RATE,
    max_seconds: float = 5.0,
) -> Path:
    """Save a small first-trial/first-channel signal comparison plot."""
    plt = _get_matplotlib_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_samples = min(int(max_seconds * sampling_rate), before.shape[-1], after.shape[-1])
    time_axis = [sample / sampling_rate for sample in range(max_samples)]

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


def save_raw_vs_filtered_plot(preprocessed: dict) -> Path:
    """Save raw stimulus vs bandpass+notch filtered stimulus comparison."""
    return _plot_signal_comparison(
        preprocessed["raw_fixed_stimulus"],
        preprocessed["filtered_stimulus"],
        "Raw vs Filtered EEG Stimulus, s01",
        "raw",
        "bandpass + notch",
        PREPROCESSING_RESULTS_DIR / "s01_raw_vs_filtered.png",
        sampling_rate=preprocessed["sampling_rate"],
    )


def save_ica_before_after_plot(preprocessed: dict) -> Path:
    """Save filtered stimulus vs ICA-output stimulus comparison."""
    return _plot_signal_comparison(
        preprocessed["filtered_stimulus"],
        preprocessed["ica_stimulus"],
        "ICA Before vs After EEG Stimulus, s01",
        "before ICA",
        "after ICA",
        PREPROCESSING_RESULTS_DIR / "s01_ica_before_after.png",
        sampling_rate=preprocessed["sampling_rate"],
    )


def save_baseline_correction_plot(preprocessed: dict) -> Path:
    """Save stimulus before vs after baseline correction comparison."""
    return _plot_signal_comparison(
        preprocessed["ica_stimulus"],
        preprocessed["baseline_corrected_stimulus"],
        "Baseline Correction Before vs After, s01",
        "before correction",
        "after correction",
        PREPROCESSING_RESULTS_DIR / "s01_baseline_correction_before_after.png",
        sampling_rate=preprocessed["sampling_rate"],
    )


def save_preprocessing_visualizations(preprocessed: dict) -> list[Path]:
    """Save the three minimal preprocessing figures required for task 1."""
    return [
        save_raw_vs_filtered_plot(preprocessed),
        save_ica_before_after_plot(preprocessed),
        save_baseline_correction_plot(preprocessed),
    ]


def _ica_exclude_label(exclude_components: list[int] | tuple[int, ...]) -> str:
    if not exclude_components:
        return "exclude_none"
    return "exclude_" + "_".join(str(component) for component in exclude_components)


def _compute_band_power_ratio(data, sampling_rate: float, band: tuple[float, float]) -> float:
    """Return target-band power divided by 4-45 Hz power."""
    from scipy.signal import welch

    freqs, psd = welch(data, fs=sampling_rate, nperseg=min(1024, data.shape[-1]), axis=-1)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total_mask = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
    band_power = float(psd[..., band_mask].sum())
    total_power = float(psd[..., total_mask].sum())
    return band_power / total_power if total_power > 0 else 0.0


def compute_ica_quantitative_summary(before, after, sampling_rate: float) -> dict:
    """Compute numerical checks before and after ICA."""
    return {
        "overall_mean_before": float(before.mean()),
        "overall_mean_after": float(after.mean()),
        "overall_std_before": float(before.std()),
        "overall_std_after": float(after.std()),
        "overall_var_before": float(before.var()),
        "overall_var_after": float(after.var()),
        "high_freq_ratio_30_45_before": _compute_band_power_ratio(
            before,
            sampling_rate,
            (30.0, 45.0),
        ),
        "high_freq_ratio_30_45_after": _compute_band_power_ratio(
            after,
            sampling_rate,
            (30.0, 45.0),
        ),
        "per_channel_std_before": before.std(axis=(0, 2)),
        "per_channel_std_after": after.std(axis=(0, 2)),
    }


def save_ica_waveform_debug_plot(
    before,
    after,
    output_dir: Path,
    sampling_rate: float,
    trial_index: int = 0,
    channel_index: int = 0,
) -> Path:
    """Save one stimulus waveform before/after ICA."""
    return _plot_signal_comparison(
        before,
        after,
        f"ICA Waveform Before vs After, trial {trial_index}, channel {channel_index}",
        "before ICA",
        "after ICA",
        output_dir / f"trial{trial_index:02d}_ch{channel_index:02d}_ica_waveform.png",
        sampling_rate=sampling_rate,
    )


def save_ica_psd_debug_plot(
    before,
    after,
    output_dir: Path,
    sampling_rate: float,
    trial_index: int = 0,
    channel_index: int = 0,
) -> Path:
    """Save one stimulus PSD before/after ICA."""
    from scipy.signal import welch

    plt = _get_matplotlib_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    before_freqs, before_psd = welch(
        before[trial_index, channel_index],
        fs=sampling_rate,
        nperseg=min(1024, before.shape[-1]),
    )
    after_freqs, after_psd = welch(
        after[trial_index, channel_index],
        fs=sampling_rate,
        nperseg=min(1024, after.shape[-1]),
    )
    output_path = output_dir / f"trial{trial_index:02d}_ch{channel_index:02d}_ica_psd.png"

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.semilogy(before_freqs, before_psd, label="before ICA")
    axis.semilogy(after_freqs, after_psd, label="after ICA")
    axis.set_title(f"ICA PSD Before vs After, trial {trial_index}, channel {channel_index}")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD")
    axis.set_xlim(0, 60)
    axis.legend()
    axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_ica_component_timeseries_plot(
    ica_info: dict,
    output_dir: Path,
    baseline_samples: int,
    sampling_rate: float,
    trial_index: int = 0,
) -> list[Path]:
    """Save time series for excluded ICA components when available."""
    sources = ica_info.get("sources")
    source_shape = ica_info.get("source_shape")
    removed_components = ica_info.get("removed_components", [])

    if sources is None or source_shape is None or not removed_components:
        return []

    plt = _get_matplotlib_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_trials = sources.reshape(source_shape)
    saved_paths = []

    for component in removed_components:
        component_signal = source_trials[trial_index, baseline_samples:, component]
        max_samples = min(int(5 * sampling_rate), len(component_signal))
        time_axis = [sample / sampling_rate for sample in range(max_samples)]
        output_path = output_dir / f"component_{component:02d}_trial{trial_index:02d}_timeseries.png"

        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(time_axis, component_signal[:max_samples], linewidth=1.0)
        axis.set_title(f"Excluded ICA Component {component}, trial {trial_index}")
        axis.set_xlabel("Stimulus time (s)")
        axis.set_ylabel("Component activation")
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def save_ica_matrix_debug_plots(ica_info: dict, output_dir: Path) -> list[Path]:
    """Save simple mixing/unmixing heatmaps and component energy ranking."""
    plt = _get_matplotlib_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for name, matrix_key in [
        ("mixing", "mixing_matrix"),
        ("unmixing", "unmixing_matrix"),
    ]:
        matrix = ica_info.get(matrix_key)
        if matrix is None:
            continue
        output_path = output_dir / f"ica_{name}_matrix.png"
        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        figure.colorbar(image, ax=axis)
        axis.set_title(f"ICA {name.title()} Matrix")
        axis.set_xlabel("Component")
        axis.set_ylabel("Channel / component")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        saved_paths.append(output_path)

    component_energy = ica_info.get("component_energy", [])
    if component_energy:
        ranked = sorted(
            enumerate(component_energy),
            key=lambda item: item[1],
            reverse=True,
        )
        output_path = output_dir / "ica_component_energy_rank.png"
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar([item[0] for item in ranked], [item[1] for item in ranked])
        axis.set_title("ICA Component Energy Ranking")
        axis.set_xlabel("Component")
        axis.set_ylabel("Mean squared activation")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths


def save_ica_trial_channel_stats_csv(before, after, output_dir: Path) -> Path:
    """Save trial/channel mean and std before/after ICA."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "trial_channel_mean_std_before_after_ica.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "trial",
                "channel",
                "mean_before",
                "mean_after",
                "std_before",
                "std_after",
                "std_delta",
            ]
        )
        for trial_index in range(before.shape[0]):
            for channel_index in range(before.shape[1]):
                before_trace = before[trial_index, channel_index]
                after_trace = after[trial_index, channel_index]
                before_std = float(before_trace.std())
                after_std = float(after_trace.std())
                writer.writerow(
                    [
                        trial_index,
                        channel_index,
                        float(before_trace.mean()),
                        float(after_trace.mean()),
                        before_std,
                        after_std,
                        after_std - before_std,
                    ]
                )

    return output_path


def save_ica_quantitative_summary(
    summary: dict,
    ica_info: dict,
    output_dir: Path,
) -> Path:
    """Save overall ICA before/after checks to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ica_debug_summary.txt"
    removed = ica_info.get("removed_components", [])
    energy = ica_info.get("component_energy", [])
    ranked_energy = sorted(enumerate(energy), key=lambda item: item[1], reverse=True)

    lines = [
        "ICA debug summary",
        "",
        f"ICA enabled: {ica_info.get('enabled')}",
        f"n_components: {ica_info.get('n_components')}",
        f"random_state: {ica_info.get('random_state')}",
        f"fit_shape: {ica_info.get('fit_shape')}",
        f"exclude components: {removed}",
        f"removed_components: {removed}",
        f"note: {ica_info.get('note')}",
        "",
        "Overall statistics before vs after ICA:",
        f"mean: {summary['overall_mean_before']:.6g} -> {summary['overall_mean_after']:.6g}",
        f"std: {summary['overall_std_before']:.6g} -> {summary['overall_std_after']:.6g}",
        f"var: {summary['overall_var_before']:.6g} -> {summary['overall_var_after']:.6g}",
        f"30-45 Hz power ratio: {summary['high_freq_ratio_30_45_before']:.6g} -> "
        f"{summary['high_freq_ratio_30_45_after']:.6g}",
        "",
        "Component energy ranking:",
    ]
    lines.extend(
        f"component {component}: {value:.6g}"
        for component, value in ranked_energy
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_ica_debug_outputs(
    before,
    after,
    ica_info: dict,
    output_dir: Path,
    sampling_rate: float,
    baseline_samples: int = TARGET_BASELINE_SAMPLES,
) -> list[Path]:
    """Save visual and numerical outputs for checking ICA behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_ica_quantitative_summary(before, after, sampling_rate)
    saved_paths = [
        save_ica_waveform_debug_plot(before, after, output_dir, sampling_rate),
        save_ica_psd_debug_plot(before, after, output_dir, sampling_rate),
        save_ica_trial_channel_stats_csv(before, after, output_dir),
        save_ica_quantitative_summary(summary, ica_info, output_dir),
    ]
    saved_paths.extend(
        save_ica_component_timeseries_plot(
            ica_info,
            output_dir,
            baseline_samples=baseline_samples,
            sampling_rate=sampling_rate,
        )
    )
    saved_paths.extend(save_ica_matrix_debug_plots(ica_info, output_dir))
    return saved_paths


def run_ica_debug_experiment(
    subject_id: int = 1,
    exclude_components: list[int] | tuple[int, ...] = (),
    ica_n_components: int | None = DEFAULT_ICA_COMPONENTS,
    ica_random_state: int = DEFAULT_ICA_RANDOM_STATE,
) -> dict:
    """Run one ICA-debug setting and save diagnostics for visual inspection."""
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
    ica_result = run_ica_artifact_removal(
        filtered["filtered_baseline"],
        filtered["filtered_stimulus"],
        enable_ica=True,
        n_components=ica_n_components,
        random_state=ica_random_state,
        exclude_components=exclude_components,
    )
    corrected_stimulus = baseline_correction(
        ica_result["baseline"],
        ica_result["stimulus"],
    )
    ica_info = {
        "enabled": ica_result["enabled"],
        "n_components": ica_n_components,
        "random_state": ica_random_state,
        "removed_components": ica_result["removed_components"],
        "fit_shape": ica_result["fit_shape"],
        "source_shape": ica_result.get("source_shape"),
        "sources": ica_result.get("sources"),
        "mixing_matrix": ica_result.get("mixing_matrix"),
        "unmixing_matrix": ica_result.get("unmixing_matrix"),
        "component_energy": ica_result["component_energy"],
        "note": ica_result["note"],
    }
    output_dir = (
        ICA_DEBUG_RESULTS_DIR
        / f"s{subject_id:02d}_{_ica_exclude_label(tuple(exclude_components))}"
    )
    saved_paths = save_ica_debug_outputs(
        filtered["filtered_stimulus"],
        ica_result["stimulus"],
        ica_info,
        output_dir,
        sampling_rate=fixed["sampling_rate"],
    )

    print(f"ICA debug setting saved: {output_dir}")
    print(f"exclude components: {list(exclude_components)}")
    print(f"removed components: {ica_result['removed_components']}")
    print(f"baseline-corrected stimulus shape: {corrected_stimulus.shape}")

    return {
        "subject_id": subject_id,
        "exclude_components": list(exclude_components),
        "filtered_stimulus": filtered["filtered_stimulus"],
        "ica_stimulus": ica_result["stimulus"],
        "baseline_corrected_stimulus": corrected_stimulus,
        "ica_info": ica_info,
        "output_dir": output_dir,
        "saved_paths": saved_paths,
    }


def run_ica_debug_experiments(
    subject_id: int = 1,
    exclude_sets: tuple[tuple[int, ...], ...] = DEFAULT_ICA_DEBUG_EXCLUDE_SETS,
    ica_n_components: int | None = DEFAULT_ICA_COMPONENTS,
    ica_random_state: int = DEFAULT_ICA_RANDOM_STATE,
) -> list[dict]:
    """Quickly try common ICA exclude settings for s01 debugging."""
    print("ICA debug experiments")
    print(f"subject: s{subject_id:02d}")
    print(f"use_ica: True")
    print(f"n_components: {ica_n_components}")
    print(f"random_state: {ica_random_state}")
    print(f"exclude sets: {[list(item) for item in exclude_sets]}")

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
    print("ICA enabled: True")
    print(f"ICA n_components: {ica_n_components}")
    decomposition = _fit_ica_decomposition(
        filtered["filtered_baseline"],
        filtered["filtered_stimulus"],
        n_components=ica_n_components,
        random_state=ica_random_state,
    )

    results = []
    for exclude_components in exclude_sets:
        print()
        print(f"running ICA exclude={list(exclude_components)}")
        applied = _apply_ica_exclusion(decomposition, exclude_components)
        print(f"ICA exclude components: {list(exclude_components)}")
        print(f"ICA removed components: {applied['removed_components']}")
        corrected_stimulus = baseline_correction(
            applied["baseline"],
            applied["stimulus"],
        )
        ica_info = {
            "enabled": True,
            "n_components": ica_n_components,
            "random_state": ica_random_state,
            "removed_components": applied["removed_components"],
            "fit_shape": decomposition["fit_shape"],
            "source_shape": decomposition["source_shape"],
            "sources": decomposition["sources"],
            "mixing_matrix": getattr(decomposition["ica"], "mixing_", None),
            "unmixing_matrix": getattr(decomposition["ica"], "components_", None),
            "component_energy": decomposition["component_energy"].tolist(),
            "note": "ICA fitted once for debug; manual component exclusion applied.",
        }
        output_dir = (
            ICA_DEBUG_RESULTS_DIR
            / f"s{subject_id:02d}_{_ica_exclude_label(tuple(exclude_components))}"
        )
        saved_paths = save_ica_debug_outputs(
            filtered["filtered_stimulus"],
            applied["stimulus"],
            ica_info,
            output_dir,
            sampling_rate=fixed["sampling_rate"],
        )
        print(f"ICA debug setting saved: {output_dir}")
        print(f"baseline-corrected stimulus shape: {corrected_stimulus.shape}")
        results.append(
            {
                "subject_id": subject_id,
                "exclude_components": list(exclude_components),
                "filtered_stimulus": filtered["filtered_stimulus"],
                "ica_stimulus": applied["stimulus"],
                "baseline_corrected_stimulus": corrected_stimulus,
                "ica_info": ica_info,
                "output_dir": output_dir,
                "saved_paths": saved_paths,
            }
        )

    return results


def summarize_subject_trial_boundaries(subject_id: int) -> dict:
    """Summarize Status-code counts and first trial durations for one subject."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    event_info = extract_status_events(bdf_path)
    trials = infer_trial_boundaries(event_info["events"])
    counts = event_info["rising_event_counts"]

    return {
        "subject_id": subject_id,
        "file_name": bdf_path.name,
        "code_3_count": counts.get(EVENT_BASELINE_START, 0),
        "code_4_count": counts.get(EVENT_STIMULUS_START, 0),
        "code_5_count": counts.get(EVENT_STIMULUS_END, 0),
        "inferred_trial_count": len(trials),
        "first_3_baseline_durations": [
            trial["baseline_duration_seconds"] for trial in trials[:3]
        ],
        "first_3_stimulus_durations": [
            trial["stimulus_duration_seconds"] for trial in trials[:3]
        ],
        "is_consistent": (
            counts.get(EVENT_BASELINE_START, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and counts.get(EVENT_STIMULUS_START, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and counts.get(EVENT_STIMULUS_END, 0) == EXPECTED_DEAP_TRIAL_COUNT
            and len(trials) == EXPECTED_DEAP_TRIAL_COUNT
        ),
    }


def validate_multi_subject_trial_boundaries(
    subject_ids: tuple[int, ...] = (1, 2, 3),
) -> list[dict]:
    """Validate trial-boundary consistency across multiple raw BDF files."""
    return [
        summarize_subject_trial_boundaries(subject_id)
        for subject_id in subject_ids
    ]


def load_participant_ratings(
    participant_id: int,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
) -> list[dict]:
    """Load participant ratings for one DEAP subject from metadata CSV."""
    metadata_path = PROJECT_ROOT / Path(metadata_dir) / "participant_ratings.csv"
    participant_rows = []

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row["Participant_id"]) == participant_id:
                participant_rows.append(row)

    return participant_rows


def inspect_subject_raw(
    subject_id: int = 1,
    original_dir: str | Path = DEFAULT_ORIGINAL_DIR,
    metadata_dir: str | Path = DEFAULT_METADATA_DIR,
) -> dict:
    """Inspect one subject's raw BDF file and related label metadata."""
    bdf_path = PROJECT_ROOT / Path(original_dir) / f"s{subject_id:02d}.bdf"
    header = read_bdf_header(bdf_path)
    ratings = load_participant_ratings(subject_id, metadata_dir)

    label_columns = ["Valence", "Arousal", "Dominance", "Liking", "Familiarity"]
    sampling_rate = header["sampling_rates"][0]
    expected_trial_samples = int(DEAP_TRIAL_DURATION_SECONDS * sampling_rate)

    return {
        **header,
        "trial_count_from_metadata": len(ratings),
        "eeg_channel_count_expected": EEG_CHANNEL_COUNT,
        "label_columns": label_columns,
        "label_shape": (len(ratings), len(label_columns)),
        "expected_trial_duration_seconds": DEAP_TRIAL_DURATION_SECONDS,
        "expected_trial_samples_per_channel": expected_trial_samples,
    }


def print_subject_raw_report(subject_id: int = 1) -> None:
    """Print a compact report for one subject's raw DEAP data structure."""
    info = inspect_subject_raw(subject_id)

    print("DEAP raw data structure check")
    print(f"file name: {info['file_name']}")
    print(f"data type: {info['data_type']}")
    print(
        "data dimension: "
        f"{info['num_signals']} channels x "
        f"{info['total_samples_per_channel']} continuous samples"
    )
    print(f"trial count: {info['trial_count_from_metadata']} metadata trials")
    print(
        "channel count: "
        f"{info['num_signals']} raw channels "
        f"({info['eeg_channel_count_expected']} EEG channels expected by DEAP)"
    )
    print(
        "per-trial duration/sample count: "
        f"{info['expected_trial_duration_seconds']} s, "
        f"{info['expected_trial_samples_per_channel']} samples/channel "
        "(expected stimulus segment; raw BDF is continuous and not segmented here)"
    )
    print(f"label information dimension: {info['label_shape']}")
    print(f"label columns: {', '.join(info['label_columns'])}")
    print(f"first 10 channel labels: {', '.join(info['channel_labels'][:10])}")


def print_subject_event_report(subject_id: int = 1) -> None:
    """Print event-channel and trial-boundary checks for one raw subject."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    event_info = extract_status_events(bdf_path)
    trials = infer_trial_boundaries(event_info["events"])

    print("DEAP raw BDF event structure check")
    print(f"file name: {bdf_path.name}")
    print(
        "event channel: "
        f"{event_info['status_channel_name']} "
        f"(index {event_info['status_channel_index']})"
    )
    print(f"sampling rate: {event_info['sampling_rate']} Hz")
    print(f"extracted rising events: {len(event_info['events'])}")
    print(f"event code counts: {event_info['rising_event_counts']}")
    print("first 20 rising events:")

    for event in event_info["events"][:20]:
        print(
            f"  sample={event['sample']}, "
            f"time={event['time_seconds']:.3f}s, "
            f"code={event['event_code']}"
        )

    print("trial-boundary inference:")
    print(
        f"  code {EVENT_BASELINE_START}: baseline start, "
        f"count={event_info['rising_event_counts'].get(EVENT_BASELINE_START, 0)}"
    )
    print(
        f"  code {EVENT_STIMULUS_START}: stimulus start, "
        f"count={event_info['rising_event_counts'].get(EVENT_STIMULUS_START, 0)}"
    )
    print(
        f"  code {EVENT_STIMULUS_END}: stimulus end, "
        f"count={event_info['rising_event_counts'].get(EVENT_STIMULUS_END, 0)}"
    )
    print(f"  inferred trials: {len(trials)}")
    print("first 5 inferred trials:")

    for trial in trials[:5]:
        print(
            f"  trial={trial['trial']}, "
            f"baseline_start={trial['baseline_start_time']:.3f}s, "
            f"stimulus_start={trial['stimulus_start_time']:.3f}s, "
            f"stimulus_end={trial['stimulus_end_time']:.3f}s, "
            f"baseline={trial['baseline_duration_seconds']:.3f}s, "
            f"stimulus={trial['stimulus_duration_seconds']:.3f}s"
        )

    print("segmentation plan:")
    print(
        "  Each trial is inferred as [code 3, code 5), containing a baseline "
        "segment [code 3, code 4) and a stimulus segment [code 4, code 5)."
    )
    print(
        f"  Expected durations are about {DEAP_BASELINE_DURATION_SECONDS}s "
        f"baseline and {DEAP_TRIAL_DURATION_SECONDS}s stimulus."
    )
    print(
        "  This script only verifies the boundary logic and does not save "
        "segmented trials."
    )
    print("uncertainty:")
    print(
        "  The event-code meaning is inferred from repeated counts and timing "
        "patterns, not from an external marker manual in this repository."
    )


def print_multi_subject_consistency_report(
    subject_ids: tuple[int, ...] = (1, 2, 3),
) -> None:
    """Print Status-event consistency checks for multiple subjects."""
    summaries = validate_multi_subject_trial_boundaries(subject_ids)
    all_consistent = all(summary["is_consistent"] for summary in summaries)

    print("DEAP multi-subject trial-boundary consistency check")
    print("verified facts:")

    for summary in summaries:
        baseline_durations = ", ".join(
            f"{duration:.3f}s"
            for duration in summary["first_3_baseline_durations"]
        )
        stimulus_durations = ", ".join(
            f"{duration:.3f}s"
            for duration in summary["first_3_stimulus_durations"]
        )

        print(f"  subject s{summary['subject_id']:02d}: {summary['file_name']}")
        print(
            "    code counts: "
            f"3={summary['code_3_count']}, "
            f"4={summary['code_4_count']}, "
            f"5={summary['code_5_count']}"
        )
        print(f"    inferred trials: {summary['inferred_trial_count']}")
        print(f"    first 3 baseline durations: {baseline_durations}")
        print(f"    first 3 stimulus durations: {stimulus_durations}")

    print("conclusion based on verified facts:")

    if all_consistent:
        print(
            "  For s01, s02, and s03, event codes 3, 4, and 5 each appear "
            "40 times, so the same trial-boundary rule is consistent across "
            "these checked subjects."
        )
        print(
            "  The fixed rule is: trial=[code 3, code 5), "
            "baseline=[code 3, code 4), stimulus=[code 4, code 5)."
        )
        print(
            "  The function get_trial_boundaries_from_bdf(bdf_path) now "
            "returns the 40 boundary records for one subject without saving "
            "EEG segments."
        )
    else:
        print(
            "  The checked subjects are not fully consistent. Trial splitting "
            "should not be finalized until the mismatched file is inspected."
        )


def print_raw_eeg_trial_extraction_report(subject_id: int = 1) -> None:
    """Cut one subject into raw EEG trial snippets and print shape details."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    summary = summarize_extracted_raw_eeg_trials(extracted)

    baseline_counts_preview = ", ".join(
        str(count) for count in summary["baseline_sample_counts"][:5]
    )
    stimulus_counts_preview = ", ".join(
        str(count) for count in summary["stimulus_sample_counts"][:5]
    )

    print("DEAP raw EEG trial extraction check")
    print(f"file name: {summary['file_name']}")
    print(f"trial count: {summary['trial_count']}")
    print(f"EEG channel count: {summary['eeg_channel_count']}")
    print(f"baseline samples per trial, first 5: {baseline_counts_preview}")
    print(f"stimulus samples per trial, first 5: {stimulus_counts_preview}")
    print(f"baseline array shape, first trial: {summary['baseline_shape_first_trial']}")
    print(f"stimulus array shape, first trial: {summary['stimulus_shape_first_trial']}")
    print(f"all baseline shape: {summary['all_baseline_shape']}")
    print(f"all stimulus shape: {summary['all_stimulus_shape']}")
    print(f"stored sample type: {summary['stored_sample_type']}")
    print(f"estimated in-memory data size: {summary['estimated_memory_mb']:.1f} MB")
    print("implementation note:")
    print(
        "  The current implementation reads only the first 32 EEG channels and "
        "only the verified trial intervals. It does not load peripheral "
        "channels or the Status channel into the extracted trial data."
    )
    print(
        "  The sample counts can vary slightly because boundaries come from "
        "real event sample indices rather than forced 5s/60s rounding."
    )


def print_standardized_trial_length_report(subject_id: int = 1) -> None:
    """Print fixed-length raw EEG trial organization details."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    standardized = standardize_raw_eeg_trial_lengths(extracted)

    print("DEAP fixed-length raw EEG trial organization check")
    print(f"file name: {standardized['file_name']}")
    print(f"target baseline length: {TARGET_BASELINE_SAMPLES} samples")
    print(f"target stimulus length: {TARGET_STIMULUS_SAMPLES} samples")
    print(f"baseline shape: {standardized['baseline_shape']}")
    print(f"stimulus shape: {standardized['stimulus_shape']}")

    if standardized["warnings"]:
        print("warnings:")
        for warning in standardized["warnings"]:
            print(f"  {warning}")
    else:
        print(
            "warnings: none; all trial segments were at least the target "
            "length and were cropped to the fixed shape where needed."
        )

    print("processing note:")
    print(
        "  This step only enforces fixed lengths: 5s baseline and 60s "
        "stimulus at 512 Hz. No filtering, ICA, artifact removal, baseline "
        "correction, feature extraction, classification, or active learning "
        "is performed."
    )


def print_basic_preprocessing_report(subject_id: int = 1) -> None:
    """Run and print the minimal basic preprocessing-chain verification."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"
    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    standardized = standardize_raw_eeg_trial_lengths(extracted)
    preprocessed = run_basic_preprocessing_on_standardized_trials(standardized)
    params = preprocessed["filter_parameters"]

    print("DEAP basic EEG preprocessing chain check")
    print(f"file name: {preprocessed['file_name']}")
    print(
        "filter parameters: "
        f"bandpass={params['bandpass_hz'][0]}-{params['bandpass_hz'][1]} Hz, "
        f"order={params['bandpass_order']}; "
        f"notch={params['notch_hz']} Hz, Q={params['notch_quality_factor']}"
    )
    print("raw fixed-length data:")
    print(f"  baseline shape: {preprocessed['raw_fixed_baseline'].shape}")
    print(f"  stimulus shape: {preprocessed['raw_fixed_stimulus'].shape}")
    print("filtered data after bandpass + notch:")
    print(f"  baseline shape: {preprocessed['filtered_baseline'].shape}")
    print(f"  stimulus shape: {preprocessed['filtered_stimulus'].shape}")
    print("baseline-corrected stimulus data:")
    print(
        "  stimulus shape: "
        f"{preprocessed['baseline_corrected_stimulus'].shape}"
    )
    print("processing note:")
    print(
        "  Baseline correction subtracts each trial/channel's filtered 5s "
        "baseline mean from the corresponding 60s filtered stimulus."
    )
    print(
        "  This report only verifies that the basic preprocessing chain runs. "
        "No ICA, feature extraction, classification, or active learning is "
        "performed."
    )


def smoke_test_preprocess_subject(subject_id: int = 1) -> dict:
    """Run a minimal s01 preprocessing smoke test and print step shapes."""
    subject = load_bdf_subject(subject_id)
    status_trials = extract_trials_from_status(subject["bdf_path"])
    extracted = extract_raw_eeg_trials_from_bdf(subject["bdf_path"])
    fixed = fix_trial_length(extracted)
    preprocessed = run_basic_preprocessing_on_standardized_trials(
        fixed,
        enable_ica=False,
    )
    figure_paths = save_preprocessing_visualizations(preprocessed)

    print("DEAP preprocessing smoke test")
    print(f"subject: s{subject_id:02d}")
    print(f"file name: {subject['bdf_path'].name}")
    print(f"trial boundaries: {len(status_trials['boundaries'])}")
    print(f"raw baseline first-trial shape: {len(extracted['trials'][0]['baseline'])}, "
          f"{len(extracted['trials'][0]['baseline'][0])}")
    print(f"raw stimulus first-trial shape: {len(extracted['trials'][0]['stimulus'])}, "
          f"{len(extracted['trials'][0]['stimulus'][0])}")
    print(f"fixed baseline shape: {fixed['baseline_shape']}")
    print(f"fixed stimulus shape: {fixed['stimulus_shape']}")
    print(f"raw fixed baseline shape: {preprocessed['raw_fixed_baseline'].shape}")
    print(f"raw fixed stimulus shape: {preprocessed['raw_fixed_stimulus'].shape}")
    print(f"filtered baseline shape: {preprocessed['filtered_baseline'].shape}")
    print(f"filtered stimulus shape: {preprocessed['filtered_stimulus'].shape}")
    print(f"ICA baseline shape: {preprocessed['ica_baseline'].shape}")
    print(f"ICA stimulus shape: {preprocessed['ica_stimulus'].shape}")
    print(f"ICA enabled: {preprocessed['ica_info']['enabled']}")
    print(f"ICA removed components: {preprocessed['ica_info']['removed_components']}")
    print(
        "baseline-corrected stimulus shape: "
        f"{preprocessed['baseline_corrected_stimulus'].shape}"
    )
    print("saved preprocessing figures:")
    for figure_path in figure_paths:
        print(f"  {figure_path}")

    expected_shape = (
        EXPECTED_DEAP_TRIAL_COUNT,
        EEG_CHANNEL_COUNT,
        TARGET_STIMULUS_SAMPLES,
    )
    if preprocessed["baseline_corrected_stimulus"].shape != expected_shape:
        raise ValueError(
            "Unexpected final stimulus shape: "
            f"{preprocessed['baseline_corrected_stimulus'].shape}, "
            f"expected {expected_shape}."
        )

    return {
        "subject": subject,
        "status_trials": status_trials,
        "raw_trials": extracted,
        "fixed_trials": fixed,
        "preprocessed": preprocessed,
        "figure_paths": figure_paths,
    }


def main() -> None:
    run_ica_debug_experiments(subject_id=1)


if __name__ == "__main__":
    main()
